import sys

import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers import DMRPPParser, HDFParser
from virtualizarr.tests import requires_network, requires_pydap, slow_test
from virtualizarr.tests.utils import obstore_s3
from virtualizarr.xarray import open_virtual_dataset

urls = [
    (
        "s3://its-live-data/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
        "s3://its-live-data/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc.dmrpp",
    )
    # TODO: later add MUR, SWOT, TEMPO and others by using kerchunk JSON to read refs (rather than reading the whole netcdf file)
]


@slow_test
@requires_network
@requires_pydap
@pytest.mark.parametrize("data_url, dmrpp_url", urls)
def test_NASA_dmrpp(data_url, dmrpp_url):
    store = obstore_s3(
        url=dmrpp_url,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry()
    registry.register("s3://its-live-data/test-space", store)
    with (
        open_virtual_dataset(
            url=dmrpp_url,
            registry=registry,
            parser=DMRPPParser(),
            loadable_variables=[],
        ) as actual,
        open_virtual_dataset(
            url=data_url,
            registry=registry,
            parser=HDFParser(),
            loadable_variables=[],
        ) as expected,
    ):
        xr.testing.assert_identical(actual, expected)


@requires_network
@requires_pydap
@slow_test
@pytest.mark.parametrize("data_url, dmrpp_url", urls)
def test_NASA_dmrpp_load(data_url, dmrpp_url):
    store = obstore_s3(
        url=dmrpp_url,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry()
    registry.register(dmrpp_url, store)
    parser = DMRPPParser()
    manifest_store = parser(url=dmrpp_url, registry=registry)

    with xr.open_dataset(
        manifest_store, engine="zarr", consolidated=False, zarr_format=3
    ) as ds:
        assert ds.load()


def local_registry() -> ObjectStoreRegistry:
    return ObjectStoreRegistry({"file://": LocalStore()})


@requires_pydap
def test_dmrpp_parser_local(dmrpp_simple_url):
    """DMRPPParser reads a local DMR++ file through the registry."""
    manifest_store = DMRPPParser()(url=dmrpp_simple_url, registry=local_registry())

    assert isinstance(manifest_store, ManifestStore)
    temperature = manifest_store._group.arrays["temperature"]
    assert temperature.shape == (25, 53)
    assert temperature.metadata.attributes["units"] == "K"

    # Chunk references point at the data file next to the .dmrpp file.
    expected_path = dmrpp_simple_url.removesuffix(".dmrpp")
    assert temperature.manifest.dict() == {
        "0.0": {"path": expected_path, "offset": 100, "length": 5300}
    }


@requires_pydap
def test_dmrpp_parser_skip_variables(dmrpp_grouped_url):
    """The skip_variables argument is passed through to pydap."""
    registry = local_registry()

    manifest_store = DMRPPParser()(url=dmrpp_grouped_url, registry=registry)
    assert "x" in manifest_store._group.arrays

    manifest_store = DMRPPParser(skip_variables=["x"])(
        url=dmrpp_grouped_url, registry=registry
    )
    assert "x" not in manifest_store._group.arrays


@requires_pydap
@pytest.mark.parametrize(
    "group, expected_vars",
    [
        (None, {"x"}),
        ("/", {"x"}),
        ("obs", {"pressure"}),
        ("/obs", {"pressure"}),
    ],
)
def test_dmrpp_parser_group(dmrpp_grouped_url, group, expected_vars):
    """The group argument selects which DMR++ group becomes the ManifestStore root."""
    manifest_store = DMRPPParser(group=group)(
        url=dmrpp_grouped_url, registry=local_registry()
    )
    assert set(manifest_store._group.arrays) == expected_vars


@requires_pydap
def test_dmrpp_open_virtual_dataset_local(dmrpp_grouped_url):
    """DMRPPParser works through the public open_virtual_dataset API."""
    with open_virtual_dataset(
        url=dmrpp_grouped_url,
        registry=local_registry(),
        parser=DMRPPParser(group="obs"),
        loadable_variables=[],
    ) as vds:
        assert set(vds.data_vars) | set(vds.coords) == {"pressure"}
        assert vds["pressure"].shape == (3,)
        assert vds["pressure"].attrs["units"] == "hPa"


@requires_pydap
def test_dmrpp_reads_stored_values(dmrpp_hdf5_url):
    """Values read through the DMR++ file are the values stored in the HDF5 file."""
    manifest_store = DMRPPParser()(url=dmrpp_hdf5_url, registry=local_registry())
    with xr.open_dataset(
        manifest_store, engine="zarr", consolidated=False, zarr_format=3
    ) as ds:
        np.testing.assert_array_equal(
            ds["grid"].values, np.arange(24, dtype="int32").reshape(4, 6)
        )


@requires_pydap
def test_dmrpp_reads_inlined_values(dmrpp_inline_url):
    """A variable whose values are inlined in the DMR++ file is read from there."""
    manifest_store = DMRPPParser()(url=dmrpp_inline_url, registry=local_registry())
    with xr.open_dataset(
        manifest_store, engine="zarr", consolidated=False, zarr_format=3
    ) as ds:
        np.testing.assert_array_equal(ds["my_dataset"].values, np.arange(10))


def test_dmrpp_without_pydap_names_the_extra(monkeypatch, dmrpp_simple_url):
    """Without pydap the error says which extra to install."""
    monkeypatch.setitem(sys.modules, "pydap", None)
    with pytest.raises(ImportError, match=r"virtualizarr\[dmrpp\]"):
        DMRPPParser()(url=dmrpp_simple_url, registry=local_registry())
