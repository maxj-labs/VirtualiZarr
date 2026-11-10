"""Concatenation of arrays that cannot share one chunk grid."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

import virtualizarr
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ConcatenatedManifestArray, ManifestArray
from virtualizarr.tests import requires_hdf5plugin, requires_icechunk
from virtualizarr.tests.utils import PYTEST_TMP_DIRECTORY_URL_PREFIX

pytestmark = pytest.mark.skipif(
    not hasattr(zarr.codecs, "ArrayConcatCodec"),
    reason="needs a zarr-python with the array_concat codec",
)

NT, NY, NX = 365, 40, 30

# (encoding of the data variable, how it differs from the first)
LAYOUTS = {
    "one_step_chunks": dict(
        chunksizes=(1, NY, NX), zlib=True, complevel=4, shuffle=False
    ),
    "shuffled": dict(chunksizes=(1, NY, NX), zlib=True, complevel=4, shuffle=True),
    "ten_step_chunks": dict(
        chunksizes=(10, 16, 16), zlib=True, complevel=4, shuffle=True
    ),
}


def _write_year(path: Path, year: int, layout: str) -> xr.Dataset:
    rng = np.random.default_rng(year)
    ds = xr.Dataset(
        {
            "tmin": (
                ("time", "y", "x"),
                rng.normal(size=(NT, NY, NX)).astype("float32"),
                {"units": "degrees C"},
            )
        },
        coords={
            "time": ("time", np.arange(NT, dtype="float64") + NT * (year - 2017)),
            "y": ("y", np.arange(NY, dtype="float32")),
            "x": ("x", np.arange(NX, dtype="float32")),
        },
    )
    encoding = {"tmin": {"_FillValue": np.float32(-9999.0), **LAYOUTS[layout]}}
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return ds


@pytest.fixture
def years(tmp_path: Path):
    """Yearly files like Daymet's: the layout changes twice, and 365 steps in chunks
    of ten leave a partial chunk at the end of every file of the last layout."""
    layouts = {
        2017: "one_step_chunks",
        2018: "one_step_chunks",
        2019: "shuffled",
        2020: "ten_step_chunks",
        2021: "ten_step_chunks",
    }
    paths, datasets = [], []
    for year, layout in layouts.items():
        path = tmp_path / f"tmin_{year}.nc"
        datasets.append(_write_year(path, year, layout))
        paths.append(str(path))
    return paths, xr.concat(datasets, dim="time")


def _open_all(paths: list[str]) -> list[xr.Dataset]:
    from virtualizarr.parsers import HDFParser

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    return [
        open_virtual_dataset(
            url=path,
            registry=registry,
            parser=HDFParser(),
            loadable_variables=["time", "y", "x"],
        )
        for path in paths
    ]


@requires_hdf5plugin
def test_mismatched_concat_raises_by_default(years):
    paths, _ = years
    vdss = _open_all(paths)
    with pytest.raises((NotImplementedError, ValueError)):
        xr.concat(vdss, dim="time", coords="minimal", compat="override")


@requires_hdf5plugin
def test_concat_keeps_incompatible_runs_as_pieces(years):
    paths, _ = years
    vdss = _open_all(paths)
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(vdss, dim="time", coords="minimal", compat="override")

    data = vds["tmin"].data
    assert isinstance(data, ConcatenatedManifestArray)
    assert data.shape == (5 * NT, NY, NX)
    assert data.axis == 0
    # 2017 and 2018 merge; 2019 has other codecs; 2020 and 2021 each end in a partial chunk
    assert [piece.shape[0] for piece in data.pieces] == [2 * NT, NT, NT, NT]
    assert vds.vz.nrefs() == 2 * NT + NT + 2 * 37 * 3 * 2

    # compatible arrays still merge into one ManifestArray
    with virtualizarr.set_options(mismatched_concat="pieces"):
        merged = xr.concat(vdss[:2], dim="time", coords="minimal", compat="override")
        assert isinstance(merged["tmin"].data, ManifestArray)
        # a second concatenation along the same axis extends the pieces
        again = xr.concat(
            [vds, vdss[-1]], dim="time", coords="minimal", compat="override"
        )
    assert len(again["tmin"].data.pieces) == 5

    with pytest.raises(NotImplementedError, match="kerchunk"):
        vds.vz.to_kerchunk(format="dict")


@requires_hdf5plugin
@requires_icechunk
def test_icechunk_roundtrip(years, tmp_path: Path):
    import icechunk

    paths, expected = years
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(
            _open_all(paths), dim="time", coords="minimal", compat="override"
        )

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix=PYTEST_TMP_DIRECTORY_URL_PREFIX,
            store=icechunk.local_filesystem_store(PYTEST_TMP_DIRECTORY_URL_PREFIX),
        )
    )
    repo = icechunk.Repository.create(
        storage=icechunk.Storage.new_in_memory(),
        config=config,
        authorize_virtual_chunk_access={PYTEST_TMP_DIRECTORY_URL_PREFIX: None},
    )
    session = repo.writable_session("main")
    vds.vz.to_icechunk(session.store)
    session.commit("daymet-like store")

    store = repo.readonly_session("main").store
    array = zarr.open_array(store, path="tmin", mode="r")
    assert array.shape == (5 * NT, NY, NX)
    assert [codec["name"] for codec in array.metadata.to_dict()["codecs"]] == [
        "array_concat"
    ]

    with xr.open_zarr(store, consolidated=False, chunks=None) as actual:
        assert list(actual.data_vars) == ["tmin"]
        assert actual["tmin"].attrs == expected["tmin"].attrs
        xr.testing.assert_identical(actual["tmin"].load(), expected["tmin"].load())
        # a time series at a point crosses every seam
        np.testing.assert_array_equal(
            actual["tmin"].isel(y=7, x=11).values, expected["tmin"].isel(y=7, x=11)
        )
        # a map on each side of the last seam
        np.testing.assert_array_equal(
            actual["tmin"].isel(time=slice(3 * NT - 2, 3 * NT + 12)).values,
            expected["tmin"].isel(time=slice(3 * NT - 2, 3 * NT + 12)),
        )
