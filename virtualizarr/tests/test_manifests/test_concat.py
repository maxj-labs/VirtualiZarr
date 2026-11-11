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

    zarr.config.set({"array.rectilinear_chunks": True})
    store = repo.readonly_session("main").store
    array = zarr.open_array(store, path="tmin", mode="r")
    assert array.shape == (5 * NT, NY, NX)
    # virtual shards where Icechunk can store them, pieces joined by a codec otherwise
    assert [codec["name"] for codec in array.metadata.to_dict()["codecs"]] in (
        ["switch"],
        ["array_concat"],
    )

    # readers are offered chunks they can use, never one chunk spanning everything
    assert (
        int(
            np.prod(
                array.chunks
                if hasattr(array, "_chunk_grid")
                and array.metadata.chunk_grid.__class__.__name__.startswith("Regular")
                else (1,)
            )
        )
        < array.size
    )
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


def _roundtrip(vds: xr.Dataset) -> "zarr.abc.store.Store":
    import icechunk

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
    session.commit("joined")
    zarr.config.set({"array.rectilinear_chunks": True})
    return repo.readonly_session("main").store


def _write(path: Path, data: np.ndarray, *, t0: int = 0, **encoding) -> xr.Dataset:
    ds = xr.Dataset(
        {"v": (("time", "y", "x"), data)},
        coords={"time": ("time", np.arange(data.shape[0], dtype="float64") + t0)},
    )
    ds.to_netcdf(path, engine="netcdf4", encoding={"v": encoding})
    return ds


@requires_hdf5plugin
@requires_icechunk
def test_same_chunk_shape_is_one_array_of_plain_chunks(tmp_path: Path):
    """A codec change and a cut last chunk need no shards: every source chunk is one
    Zarr chunk, with released Icechunk."""
    rng = np.random.default_rng(0)
    layouts = [
        dict(chunksizes=(4, 6, 5), zlib=True, shuffle=False),
        dict(chunksizes=(4, 6, 5), zlib=True, shuffle=True),  # 10 steps: cut last chunk
        dict(chunksizes=(4, 6, 5), zlib=False),
    ]
    paths, datasets, t0 = [], [], 0
    for i, (n, encoding) in enumerate(zip((8, 10, 7), layouts)):
        data = rng.normal(size=(n, 12, 10)).astype("float32")
        paths.append(str(tmp_path / f"f{i}.nc"))
        datasets.append(_write(Path(paths[-1]), data, t0=t0, **encoding))
        t0 += n
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    from virtualizarr.parsers import HDFParser

    vdss = [
        open_virtual_dataset(
            url=p, registry=registry, parser=HDFParser(), loadable_variables=["time"]
        )
        for p in paths
    ]
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(vdss, dim="time", coords="minimal", compat="override")
    store = _roundtrip(vds)
    array = zarr.open_array(store, path="v", mode="r")
    codec = array.metadata.to_dict()["codecs"][0]
    assert codec["name"] == "switch"
    assert [case["chunks"] for case in codec["configuration"]["cases"]] == [2, 3, 2]
    expected = np.concatenate([d["v"].values for d in datasets])
    np.testing.assert_array_equal(array[:], expected)
    np.testing.assert_array_equal(array[6:20, 3:9, ::2], expected[6:20, 3:9, ::2])


@requires_hdf5plugin
@requires_icechunk
def test_data_type_change(tmp_path: Path):
    """int16 in one file and float32 in the next read as one float32 array."""
    pytest.importorskip("cast_value_rs")
    rng = np.random.default_rng(1)
    a = rng.integers(-500, 500, size=(4, 6, 5)).astype("int16")
    b = rng.normal(size=(4, 6, 5)).astype("float32")
    _write(tmp_path / "a.nc", a, chunksizes=(2, 6, 5), zlib=True)
    _write(tmp_path / "b.nc", b, t0=4, chunksizes=(2, 6, 5), zlib=True)
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    from virtualizarr.parsers import HDFParser

    vdss = [
        open_virtual_dataset(
            url=str(tmp_path / f),
            registry=registry,
            parser=HDFParser(),
            loadable_variables=["time"],
        )
        for f in ("a.nc", "b.nc")
    ]
    with pytest.raises((ValueError, NotImplementedError)):
        xr.concat(vdss, dim="time", coords="minimal", compat="override")
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(vdss, dim="time", coords="minimal", compat="override")
    assert vds["v"].dtype == np.float32
    array = zarr.open_array(_roundtrip(vds), path="v", mode="r")
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array[:], np.concatenate([a.astype("float32"), b]))


@requires_hdf5plugin
@requires_icechunk
def test_stacking_files_without_the_joined_axis(tmp_path: Path):
    """Daily maps with no time axis, stacked along a new one, with a codec change."""
    rng = np.random.default_rng(2)
    maps = [rng.normal(size=(12, 10)).astype("float32") for _ in range(4)]
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    from virtualizarr.parsers import HDFParser

    vdss = []
    for i, m in enumerate(maps):
        path = tmp_path / f"day{i}.nc"
        xr.Dataset({"v": (("y", "x"), m)}).to_netcdf(
            path,
            engine="netcdf4",
            encoding={"v": dict(chunksizes=(6, 5), zlib=True, shuffle=i >= 2)},
        )
        vdss.append(
            open_virtual_dataset(
                url=str(path),
                registry=registry,
                parser=HDFParser(),
                loadable_variables=[],
            )
        )
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(vdss, dim="day", coords="minimal", compat="override")
    assert vds["v"].shape == (4, 12, 10)
    array = zarr.open_array(_roundtrip(vds), path="v", mode="r")
    np.testing.assert_array_equal(array[:], np.stack(maps))


@requires_hdf5plugin
@requires_icechunk
def test_file_without_the_variable_reads_as_its_fill_value(tmp_path: Path):
    """A granule that lacks the variable is a piece that stores nothing. It reads as
    its own fill value, even where that is not the joined array's."""
    from virtualizarr.manifests.plan import plan_joined_array
    from virtualizarr.parsers import HDFParser

    rng = np.random.default_rng(4)
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    data = [rng.normal(size=(4, 6, 5)).astype("float32") for _ in range(2)]
    vdss = []
    for i, d in enumerate(data):
        _write(
            tmp_path / f"g{i}.nc",
            d,
            t0=8 * i,
            chunksizes=(2, 6, 5),
            zlib=True,
            _FillValue=np.float32(-9999.0),
        )
        vdss.append(
            open_virtual_dataset(
                url=str(tmp_path / f"g{i}.nc"),
                registry=registry,
                parser=HDFParser(),
                loadable_variables=["time"],
            )
        )
    first, last = (v["v"].data for v in vdss)
    absent = first.with_fill_value_only(
        np.float32(7.0)
    )  # the middle granule has no `v`
    plan = plan_joined_array([first, absent, last], axis=0)
    assert plan.strategy == "switch"
    assert plan.notes["pieces_with_their_own_fill"] == 1
    assert [case.get("fill_value") for case in plan.cases] == [None, 7.0, None]

    middle = xr.Dataset(
        {"v": (("time", "y", "x"), absent)},
        coords={"time": ("time", np.arange(4.0) + 4)},
    )
    with virtualizarr.set_options(mismatched_concat="pieces"):
        vds = xr.concat(
            [vdss[0], middle, vdss[1]], dim="time", coords="minimal", compat="override"
        )
    array = zarr.open_array(_roundtrip(vds), path="v", mode="r")
    expected = np.concatenate(
        [data[0], np.full((4, 6, 5), 7.0, dtype="float32"), data[1]]
    )
    np.testing.assert_array_equal(array[:], expected)
    np.testing.assert_array_equal(array[2:10, 1:4, ::2], expected[2:10, 1:4, ::2])


@requires_hdf5plugin
@requires_icechunk
@pytest.mark.parametrize("layouts", ["same chunk shape", "chunk shape changes"])
@pytest.mark.parametrize("select", ["position", "key"])
def test_append_to_a_joined_array(tmp_path: Path, layouts: str, select: str):
    """New files, in yet another layout, are appended without touching what is there."""
    import icechunk

    from virtualizarr.parsers import HDFParser
    from virtualizarr.writers.icechunk import _icechunk_takes_parts

    if (
        layouts == "chunk shape changes" or select == "key"
    ) and not _icechunk_takes_parts():
        pytest.skip("needs an Icechunk whose references take a prefix and more ranges")
    rng = np.random.default_rng(5)
    second = (4, 6, 5) if layouts == "same chunk shape" else (2, 12, 5)
    encodings = [
        dict(chunksizes=(4, 6, 5), zlib=True, shuffle=False),
        dict(chunksizes=second, zlib=True, shuffle=True),
        dict(chunksizes=(4, 6, 5), zlib=False),  # appended: a third layout, 10 steps
        dict(chunksizes=(4, 6, 5), zlib=True, shuffle=False),  # appended again
    ]
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    vdss, datasets, t0 = [], [], 0
    for i, (n, encoding) in enumerate(zip((8, 8, 10, 7), encodings)):
        path = tmp_path / f"f{i}.nc"
        datasets.append(
            _write(
                path, rng.normal(size=(n, 12, 10)).astype("float32"), t0=t0, **encoding
            )
        )
        vdss.append(
            open_virtual_dataset(
                url=str(path),
                registry=registry,
                parser=HDFParser(),
                loadable_variables=["time"],
            )
        )
        t0 += n

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
    zarr.config.set({"array.rectilinear_chunks": True})
    with virtualizarr.set_options(mismatched_concat="pieces"):
        first = xr.concat(vdss[:2], dim="time", coords="minimal", compat="override")
    session = repo.writable_session("main")
    with virtualizarr.set_options(switch_select=select):
        first.vz.to_icechunk(session.store)
    session.commit("two files")
    for i in (2, 3):
        session = repo.writable_session("main")
        vdss[i].vz.to_icechunk(session.store, append_dim="time")
        session.commit(f"append file {i}")
        expected = np.concatenate([d["v"].values for d in datasets[: i + 1]])
        with xr.open_zarr(
            repo.readonly_session("main").store, consolidated=False, chunks=None
        ) as ds:
            assert ds.sizes["time"] == expected.shape[0]
            np.testing.assert_array_equal(ds["v"].values, expected)
            np.testing.assert_array_equal(
                ds["time"].values, np.arange(expected.shape[0], dtype="float64")
            )
    array = zarr.open_array(repo.readonly_session("main").store, path="v", mode="r")
    configuration = array.metadata.to_dict()["codecs"][0]["configuration"]
    assert configuration.get("select", "position") == select
    # by position: one case per run of chunks. By key: one per distinct layout, and
    # the first and last files share theirs.
    assert len(configuration["cases"]) == (4 if select == "position" else 3)
