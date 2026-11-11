import asyncio
import dataclasses
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Union, cast

import numpy as np
import xarray as xr
from xarray.backends.zarr import ZarrStore as XarrayZarrStore
from xarray.backends.zarr import encode_zarr_attr_value
from zarr import Array, Group, open_group
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync

from virtualizarr.codecs import extract_codecs, get_codecs
from virtualizarr.manifests import (
    ChunkManifest,
    ConcatenatedManifestArray,
    ManifestArray,
)
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.manifests.utils import (
    check_compatible_encodings,
    check_same_chunk_shapes,
    check_same_codecs,
    check_same_dtypes,
    check_same_ndims,
    check_same_shapes_except_axes,
    check_same_shapes_except_on_concat_axis,
)

if TYPE_CHECKING:
    from icechunk import (
        IcechunkStore,  # type: ignore[import-not-found]
        RepositoryConfig,  # type: ignore[import-not-found]
    )


ENCODING_KEYS = {"_FillValue", "missing_value", "scale_factor", "add_offset"}

VALID_MODES = ("w", "w-", "a")


def _resolve_mode(
    mode: Optional[Literal["w", "w-", "a"]],
    append_dim: Optional[str] = None,
    region: object = None,
) -> Literal["w", "w-", "a", "r+"]:
    """Validate ``mode`` and resolve it to the effective zarr group-open mode."""
    if not isinstance(mode, (type(None), str)):
        raise TypeError(f"mode: expected type Optional[str], but got type {type(mode)}")

    if mode is not None and mode not in VALID_MODES:
        raise ValueError(f"mode: expected one of {VALID_MODES}, but got {mode!r}")

    if append_dim or region:
        if mode in ("w", "w-"):
            raise ValueError(
                f"mode {mode!r} cannot be used together with append_dim or region, "
                "which require opening an existing group"
            )
        # appending or writing to a region requires the group (and arrays) to already exist
        return "r+"

    return mode or "w-"


def virtual_dataset_to_icechunk(
    vds: xr.Dataset,
    store: "IcechunkStore",
    *,
    group: Optional[str] = None,
    mode: Optional[Literal["w", "w-", "a"]] = None,
    append_dim: Optional[str] = None,
    region: Optional[Literal["auto"] | Mapping[str, Literal["auto"] | slice]] = None,
    validate_containers: bool = True,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write an virtual xarray dataset to an Icechunk store.

    Both `icechunk` and `zarr` (v3) must be installed.

    Parameters
    ----------
    vds
        Dataset to write to an Icechunk store. Can contain both "virtual" variables (backed by ManifestArray objects) and "loadable" variables (backed by numpy arrays).
    store
        Store to write the dataset to, which must not be read-only.
    group
        Path to the group in which to store the dataset, defaulting to the root group.
    mode
        How to handle a pre-existing group at the target path:

        - ``"w-"``: create the group, raising a ``ContainsGroupError`` if it already exists.
        - ``"w"``: create the group, overwriting any existing contents at that path.
        - ``"a"``: open the group if it exists (keeping existing arrays), otherwise create it.
          An existing array of the same name must have the same metadata apart from
          attributes, otherwise a ``ValueError`` is raised.
        - ``None`` (default): equivalent to ``"w-"``, unless ``append_dim`` or ``region``
          is given, in which case the existing group is opened.

        ``mode="w"`` and ``mode="w-"`` are incompatible with ``append_dim`` and ``region``.
    append_dim
        Name of the dimension along which to append data. If provided, the dataset must
        have a dimension with this name.
    region
        Optional mapping from dimension names to either a) ``"auto"``, or b) integer
        slices, indicating the region of existing zarr array(s) in which to write
        this dataset's data.

        See ``xarray.Dataset.to_zarr`` documentation for details.
    validate_containers
        If ``True``, raise if any virtual chunks refer to locations that don't
        match any existing virtual chunk container set on this Icechunk repository.

        It is not generally recommended to set this to ``False``, because it can lead to
        confusing runtime results and errors when reading data back.
    last_updated_at
        The time at which the virtual dataset was last updated. When specified, if any
        of the virtual chunks written in this session are modified in storage after this
        time, icechunk will raise an error at runtime when trying to read the virtual
        chunk. When not specified, icechunk will not check for modifications to the
        virtual chunks at runtime.

    Raises
    ------
    ValueError
        If the store is read-only.
    """
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]
        from zarr.storage import StorePath  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'icechunk' and 'zarr' version 3 libraries are required to use this function"
        ) from None

    if not isinstance(store, IcechunkStore):
        raise TypeError(
            f"store: expected type IcechunkStore, but got type {type(store)}"
        )

    if not isinstance(group, (type(None), str)):
        raise TypeError(
            f"group: expected type Optional[str], but got type {type(group)}"
        )

    open_mode = _resolve_mode(mode, append_dim=append_dim, region=region)

    if not isinstance(append_dim, (type(None), str)):
        raise TypeError(
            f"append_dim: expected type Optional[str], but got type {type(append_dim)}"
        )

    if not isinstance(region, (type(None), str, Mapping)):
        raise TypeError(
            "region: expected type Optional[Literal['auto'] | Mapping[str, Literal['auto'] | slice]],"
            f" but got type {type(last_updated_at)}"
        )

    if not isinstance(last_updated_at, (type(None), datetime)):
        raise TypeError(
            "last_updated_at: expected type Optional[datetime],"
            f" but got type {type(last_updated_at)}"
        )

    if store.read_only:
        raise ValueError("supplied store is read-only")

    if append_dim and append_dim not in vds.dims:
        raise ValueError(
            f"append_dim {append_dim!r} does not match any existing dataset dimensions"
        )

    store_path = StorePath(store, path=group or "")

    if validate_containers:
        validate_virtual_chunk_containers(store.session.config, [vds])

    group_object = open_group(
        store_path, mode=open_mode, zarr_format=3, use_consolidated=False
    )

    write_virtual_dataset_to_icechunk_group(
        vds=vds,
        store=store,
        group=group_object,
        append_dim=append_dim,
        region=region,
        last_updated_at=last_updated_at,
    )


def virtual_datatree_to_icechunk(
    vdt: xr.DataTree,
    store: "IcechunkStore",
    *,
    mode: Optional[Literal["w", "w-", "a"]] = None,
    write_inherited_coords: bool = False,
    validate_containers: bool = True,
    last_updated_at: datetime | None = None,
    **kwargs,
) -> None:
    """
    Write an xarray dataset to an Icechunk store.

    Both `icechunk` and `zarr` (v3) must be installed.

    Parameters
    ----------
    vdt
        DataTree to write to an Icechunk store. Can contain both "virtual" variables (backed by ManifestArray objects) and "loadable" variables (backed by numpy arrays).
    store
        Store to write the dataset to, which must not be read-only.
    mode
        How to handle pre-existing groups at the target paths:

        - ``"w-"`` or ``None`` (default): create each group, raising a
          ``ContainsGroupError`` if it already exists.
        - ``"w"``: create each group, overwriting any existing contents at that path.
        - ``"a"``: open each group if it exists (keeping existing arrays), otherwise create it.
          An existing array of the same name must have the same metadata apart from
          attributes, otherwise a ``ValueError`` is raised.
    write_inherited_coords
        If ``True``, replicate inherited coordinates on all descendant nodes of the
        tree. Otherwise, only write coordinates at the level at which they are
        originally defined. This saves disk space, but requires opening the
        full tree to load inherited coordinates.
    validate_containers
        If ``True``, raise if any virtual chunks refer to locations that don't
        match any existing virtual chunk container set on this Icechunk repository.

        It is not generally recommended to set this to ``False``, because it can lead to
        confusing runtime results and errors when reading data back.
    last_updated_at
        The time at which the virtual dataset was last updated. When specified, if any
        of the virtual chunks written in this session are modified in storage after this
        time, icechunk will raise an error at runtime when trying to read the virtual
        chunk. When not specified, icechunk will not check for modifications to the
        virtual chunks at runtime.
    **kwargs
        Additional keyword arguments to be passed to ``xarray.Dataset.vz.to_icechunk``.

    Raises
    ------
    ValueError
        If the store is read-only.
    """
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]
        from zarr.storage import StorePath  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'icechunk' and 'zarr' version 3 libraries are required to use this function"
        ) from None

    if not isinstance(store, IcechunkStore):
        raise TypeError(
            f"store: expected type IcechunkStore, but got type {type(store)}"
        )

    open_mode = _resolve_mode(
        mode, append_dim=kwargs.get("append_dim"), region=kwargs.get("region")
    )

    if not isinstance(last_updated_at, (type(None), datetime)):
        raise TypeError(
            "last_updated_at: expected type datetime,"
            f" but got type {type(last_updated_at)}"
        )

    if store.read_only:
        raise ValueError("supplied store is read-only")

    def node_to_vds(node: xr.DataTree) -> xr.Dataset:
        tree = cast(xr.DataTree, node)  # subtree is typed as Unknown
        at_root = tree is vdt
        return tree.to_dataset(write_inherited_coords or at_root)

    def get_store_path(subtree, vdt) -> StorePath:
        at_root = subtree is vdt
        return StorePath(store, path="" if at_root else subtree.relative_to(vdt))

    # can't just use a dict because StorePath is not hashable
    paths_and_virtual_datasets = [
        (get_store_path(subtree, vdt), node_to_vds(subtree)) for subtree in vdt.subtree
    ]
    virtual_datasets = [pair[1] for pair in paths_and_virtual_datasets]

    if validate_containers:
        validate_virtual_chunk_containers(store.session.config, virtual_datasets)

    # TODO this serial loop could be slow writing lots of groups to high-latency store, see https://github.com/pydata/xarray/issues/9455
    for store_path, vds in paths_and_virtual_datasets:
        group = open_group(
            store_path, mode=open_mode, zarr_format=3, use_consolidated=False
        )

        write_virtual_dataset_to_icechunk_group(
            vds=vds,
            store=store,
            group=group,
            last_updated_at=last_updated_at,
            **kwargs,
        )


# TODO ideally I would be able to just call some Icechunk API to do this (see https://github.com/earth-mover/icechunk/issues/1167)
def validate_virtual_chunk_containers(
    config: "RepositoryConfig", virtual_datasets: Iterable[xr.Dataset]
) -> None:
    """Check that all virtual refs have corresponding virtual chunk containers, before writing any of the refs."""

    manifestarrays = [
        marr
        for dataset in virtual_datasets
        for var in dataset.variables.values()
        for marr in _manifest_arrays(var.data)
    ]

    # get the prefixes of all virtual chunk containers
    if config.virtual_chunk_containers is None:
        # TODO for some reason Icechunk returns None instead of an empty dict if there are zero containers (see https://github.com/earth-mover/icechunk/issues/1168)
        supported_prefixes = set()
    else:
        supported_prefixes = set(config.virtual_chunk_containers.keys())

    # fastpath for common case that no virtual chunk containers have been set
    if manifestarrays and not supported_prefixes:
        raise ValueError("No Virtual Chunk Containers set")

    # check all refs against existing virtual chunk containers
    # passing a tuple to str.startswith runs the loop over prefixes in C
    supported_prefixes_tuple = tuple(supported_prefixes)
    for marr in manifestarrays:
        # TODO this loop over every virtual reference is likely inefficient in python,
        # is there a way to push this down to Icechunk? (see https://github.com/earth-mover/icechunk/issues/1167)
        for ref in marr.manifest.iter_nonempty_paths():
            validate_single_ref(ref, supported_prefixes_tuple)


def validate_single_ref(ref: str, supported_prefixes: tuple[str, ...]) -> None:
    if not ref.startswith(supported_prefixes):
        raise ValueError(
            f"No Virtual Chunk Container set which supports prefix of path {ref}"
        )


def write_virtual_dataset_to_icechunk_group(
    vds: xr.Dataset,
    store: "IcechunkStore",
    group: Group,
    append_dim: Optional[str] = None,
    region: Optional[Literal["auto"] | Mapping[str, Literal["auto"] | slice]] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    if region is not None:
        vds, region = validate_and_autodetect_region(group, vds, region)

    virtual_variables = {
        name: var
        for name, var in vds.variables.items()
        if isinstance(var.data, (ManifestArray, ConcatenatedManifestArray))
    }

    loadable_variables = {
        name: var
        for name, var in vds.variables.items()
        if name not in virtual_variables
    }

    # First write all the non-virtual variables
    if loadable_variables:
        mode: Literal["a", "r+"] | None
        if append_dim is None and region is None:
            mode = "a"
        else:
            # let xarray set it automatically, 'a' for append_dim and 'r+' for region
            mode = None
        loadable_ds = xr.Dataset(loadable_variables)
        loadable_ds.to_zarr(  # type: ignore[call-overload]
            store,
            group=group.name,
            zarr_format=3,
            consolidated=False,
            mode=mode,
            append_dim=append_dim,
            region=region,
        )

    # Then write the virtual variables to the same group
    for name, var in virtual_variables.items():
        if isinstance(var.data, ConcatenatedManifestArray):
            write_concatenated_variable_to_icechunk(
                store=store,
                group=group,
                name=name,  # type: ignore[arg-type]
                var=var,
                append_dim=append_dim,
                region=region,
                last_updated_at=last_updated_at,
            )
            continue
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,  # type: ignore[arg-type]
            var=var,
            append_dim=append_dim,
            region=region,
            last_updated_at=last_updated_at,
        )

    # finish by writing group-level attributes
    # note: group attributes must be set after writing individual variables else it gets overwritten
    update_attributes(group, vds.attrs, coords=vds.coords)


def update_attributes(
    zarr_node: Array | Group, attrs: dict, coords=None, encoding=None
):
    """Update metadata attributes of one Zarr node (array or group), to match how xarray does it."""

    zarr_node.update_attributes(
        {k: encode_zarr_attr_value(v) for k, v in attrs.items()}
    )

    # preserve info telling xarray which variables are coordinates upon re-opening
    if isinstance(zarr_node, Group) and coords:
        zarr_node.update_attributes(
            {"coordinates": " ".join(list(coords))},
        )

    # preserve variable-level encoding
    if isinstance(zarr_node, Array) and encoding:
        for k, v in encoding.items():
            if k in ENCODING_KEYS:
                zarr_node.attrs[k] = encode_zarr_attr_value(v)


def validate_and_autodetect_region(
    group: Group,
    vds: xr.Dataset,
    region: Literal["auto"] | Mapping[str, Literal["auto"] | slice],
) -> tuple[xr.Dataset, dict[str, slice]]:
    """
    Convert regions like `"auto"` and `{"dim": "auto"}` into concrete `dict[str, slice]`
    in a way which is maximally compatible with xarray's `to_zarr`.
    The method itself is quite complicated, we call into xarray
    so we do not have to reimplement the logic.

    This function uses xarray's internal private functions and can break at any time.
    """
    xarray_store = XarrayZarrStore(
        zarr_group=group,
        append_dim=None,
        write_region=region,
        consolidate_on_close=False,
        close_store_on_close=False,
        mode="r+",
    )
    vds = xarray_store._validate_and_autodetect_region(vds)
    xarray_store.close()
    return vds, cast(dict[str, slice], xarray_store._write_region)


def num_chunks(
    array,
    axis: int,
) -> int:
    return array.shape[axis] // array.chunks[axis]


def resize_array(
    arr: "Array",
    manifest_array: "ManifestArray",
    append_axis: int,
) -> None:
    new_shape = list(arr.shape)
    new_shape[append_axis] += manifest_array.shape[append_axis]
    arr.resize(tuple(new_shape))


def get_axis(
    dims: list[str],
    dim_name: Optional[str],
) -> int:
    if dim_name is None:
        raise ValueError("dim_name must be provided")
    return dims.index(dim_name)


def check_compatible_arrays(
    ma: "ManifestArray",
    existing_array: "Array",
    append_axis: int | None,
    except_axes: list[int] | None = None,
):
    arrays: List[Union[ManifestArray, Array]] = [ma, existing_array]
    check_same_dtypes([arr.dtype for arr in arrays])
    check_same_codecs([get_codecs(arr) for arr in arrays])
    check_same_chunk_shapes([arr.metadata.chunks for arr in arrays])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    if append_axis is not None:
        check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)
    if except_axes is not None:
        check_same_shapes_except_axes(arr_shapes, except_axes)


def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: xr.Variable,
    append_dim: Optional[str] = None,
    region: Optional[Mapping[str, slice]] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """Write a single virtual variable into an icechunk store"""

    if append_dim and append_dim in var.dims and _is_joined_array(group.get(name)):
        append_to_joined_array(store, group, name, var, append_dim, last_updated_at)
        return

    ma = cast(ManifestArray, var.data)
    metadata = ma.metadata

    dims: list[str] = cast(list[str], list(var.dims))
    chunk_offsets: list[int]
    existing_num_chunks = 0
    if append_dim and append_dim in dims:
        # TODO: MRP - zarr, or icechunk zarr, array assignment to a variable doesn't work to point to the same object
        # for example, if you resize an array, it resizes the array but not the bound variable.
        if not isinstance(group[name], Array):
            raise ValueError("Expected existing array to be a zarr.core.Array")
        append_axis = get_axis(dims, append_dim)

        # check if arrays can be concatenated
        check_compatible_arrays(ma, group[name], append_axis)  # type: ignore[arg-type]
        check_compatible_encodings(var.encoding, group[name].attrs)

        # determine number of existing chunks along the append axis
        existing_num_chunks = num_chunks(
            array=group[name],
            axis=append_axis,
        )
        chunk_offsets = [
            existing_num_chunks if dim == append_dim else 0 for dim in dims
        ]

        # resize the array
        resize_array(
            group[name],  # type: ignore[arg-type]
            manifest_array=ma,
            append_axis=append_axis,
        )
    elif region is not None:
        existing_array = group[name]
        if not isinstance(existing_array, Array):
            raise ValueError(
                f"Expected {name!r} to be a zarr.core.Array, got {type(existing_array)}"
            )
        check_compatible_arrays(
            ma,
            existing_array,
            append_axis=None,
            except_axes=[
                get_axis(dims, region_dim)
                for region_dim in region.keys()
                if region_dim in dims
            ],
        )
        check_compatible_encodings(var.encoding, existing_array.attrs)

        chunk_offsets = []
        for dim, chunk_size in zip(dims, existing_array.chunks):
            if dim not in region:
                chunk_offsets.append(0)
                continue
            dim_region = region[dim]
            start = dim_region.start if dim_region.start is not None else 0
            if start % chunk_size != 0:
                raise ValueError(
                    f"Trying to write variable {name!r} to region "
                    + f"{region!r} in dimension {dim!r}, but it is not aligned to whole "
                    + f"chunks of size {chunk_size!r}"
                )
            chunk_offsets.append(start // chunk_size)
    else:
        chunk_offsets = [0 for _ in dims]
        filters, serializer, compressors = extract_codecs(metadata.inner_codecs)
        array_kwargs = dict(
            shape=metadata.shape,
            chunks=metadata.chunks,
            shards=metadata.shards,
            dtype=metadata.data_type.to_native_dtype(),
            filters=filters,
            compressors=compressors,
            serializer=serializer,
            dimension_names=var.dims,
            fill_value=metadata.fill_value,
        )
        existing = group.get(name)
        if isinstance(existing, Array):
            _check_existing_array_matches(existing, metadata, dims)
        arr = group.require_array(name=name, **array_kwargs)

        update_attributes(arr, var.attrs, encoding=var.encoding)

    write_manifest_to_icechunk(
        store=store,
        group=group,
        arr_name=name,
        manifest=ma.manifest,
        chunk_index_offsets=tuple(chunk_offsets),
        last_updated_at=last_updated_at,
    )


PIECES_GROUP = "_pieces"
"""The child group that holds the pieces of a group's concatenated variables."""


def _reported_chunks(cma: ConcatenatedManifestArray) -> tuple[int, ...]:
    """
    The chunk shape an array joined by the `array_concat` codec reports to readers.

    The array stores no chunks, so this is a hint. A common multiple of the pieces'
    chunk sizes lets a reader that works chunk by chunk read each stored chunk once;
    where that exceeds the array, the largest of the pieces' chunk sizes is used.
    """
    import math

    from virtualizarr.manifests.utils import manifest_chunk_shape

    shapes = [manifest_chunk_shape(piece.metadata) for piece in cma.pieces]
    out = []
    for d, extent in enumerate(cma.shape):
        sizes = [shape[d] for shape in shapes]
        lcm = math.lcm(*sizes)
        out.append(min(extent, lcm if lcm <= extent else max(sizes)))
    return tuple(out)


def _is_joined_array(node: object) -> bool:
    """Whether a node is an array written from a plan (its codec is `switch`)."""
    if not isinstance(node, Array):
        return False
    codecs = node.metadata.to_dict().get("codecs") or ()
    return (
        len(codecs) == 1
        and isinstance(codecs[0], dict)
        and codecs[0].get("name") == "switch"
    )


def _grid_edges(chunk_grid: dict, shape: tuple[int, ...]) -> list[list[int]]:
    """A chunk grid's edges along each dimension, from its JSON."""
    configuration = chunk_grid["configuration"]
    if chunk_grid["name"] == "regular":
        return [
            [c] * (n // c) + ([n % c] if n % c else [])
            for n, c in zip(shape, configuration["chunk_shape"])
        ]
    out = []
    for spec, n in zip(configuration["chunk_shapes"], shape):
        if isinstance(spec, int):
            out.append([spec] * (n // spec) + ([n % spec] if n % spec else []))
        else:
            edges: list[int] = []
            for item in spec:
                edges += [item] if isinstance(item, int) else [item[0]] * item[1]
            out.append(edges)
    return out


def append_to_joined_array(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: xr.Variable,
    append_dim: str,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Append a virtual variable to an array that was written from a plan.

    The new data is planned to fit the existing array (the same strategy, and the same
    chunks or shards off the joined axis); its chunks are added after the existing ones
    along the axis and its cases after the existing cases. Nothing already written
    changes. New data that does not fit raises ``NotImplementedError``: the whole record
    then has to be planned and written again.
    """
    import icechunk
    import zarr
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.metadata.v3 import ArrayV3Metadata

    from virtualizarr.manifests.plan import plan_joined_array

    existing = cast(Array, group[name])
    axis = get_axis(cast(list[str], list(var.dims)), append_dim)
    meta = json.loads(json.dumps(existing.metadata.to_dict(), default=str))
    codec = meta["codecs"][0]["configuration"]
    select = codec.get("select", "position")
    if select == "position" and codec["axis"] != axis:
        raise NotImplementedError(
            f"The array {name!r} is joined along axis {codec['axis']}, not along {append_dim!r}."
        )
    grid = _grid_edges(meta["chunk_grid"], existing.shape)
    strategy = (
        "shards"
        if any(
            case["codecs"][0].get("name") == "sharding_indexed"
            for case in codec["cases"]
        )
        else "switch"
    )
    data = var.data
    pieces = (
        list(data.pieces)
        if isinstance(data, ConcatenatedManifestArray)
        else [cast(ManifestArray, data)]
    )
    plan = plan_joined_array(
        pieces,
        axis,
        dtype=existing.dtype,
        fill_value=existing.fill_value,
        like={"strategy": strategy, "chunk_grid": grid},
        select=select,
    )
    if (plan.strategy == "shards" or select == "key") and not _icechunk_takes_parts():
        raise NotImplementedError(
            "this Icechunk cannot store a reference with a prefix or several ranges"
        )

    existing_chunks = len(grid[axis])
    cases = codec["cases"]
    for case in plan.cases:
        if select == "key":
            # cases are named by their content: the two tables merge by union
            if not any(c["key"] == case["key"] for c in cases):
                cases.append(case)
        elif {k: v for k, v in cases[-1].items() if k != "chunks"} == {
            k: v for k, v in case.items() if k != "chunks"
        }:
            cases[-1]["chunks"] += case["chunks"]
        else:
            cases.append(case)
    grid[axis] = grid[axis] + plan.chunk_grid[axis]
    meta["shape"] = [
        n + (plan.shape[axis] if d == axis else 0) for d, n in enumerate(existing.shape)
    ]
    meta["chunk_grid"] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": grid},
    }
    key_prefix = (name if group.name == "/" else f"{group.name}/{name}").lstrip("/")
    if last_updated_at is None:
        last_updated_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    with zarr.config.set({"array.rectilinear_chunks": True}):
        new_metadata = ArrayV3Metadata.from_dict(meta)
        document = new_metadata.to_buffer_dict(default_buffer_prototype())["zarr.json"]
        sync(store.set(f"{key_prefix}/zarr.json", document))
        refs = []
        for ref in plan.refs:
            index = list(ref["index"])
            index[axis] += existing_chunks
            refs.append(
                icechunk.VirtualChunkSpec(
                    **{**ref, "index": index}, last_updated_at_checksum=last_updated_at
                )
            )
        failed = store.set_virtual_refs(key_prefix, refs, validate_containers=True)
        _write_literals(store, key_prefix, plan.literals, axis, existing_chunks)
    if failed:
        raise ValueError(
            f"{len(failed)} references of variable {name!r} are outside the "
            "repository's virtual chunk containers."
        )


def _write_literals(
    store: "IcechunkStore",
    key_prefix: str,
    literals: list[dict],
    axis: int = 0,
    offset: int = 0,
) -> None:
    """Write a plan's literal chunks: the few bytes that name the case of a chunk which
    stores nothing else."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    proto = default_buffer_prototype()
    for literal in literals:
        index = list(literal["index"])
        index[axis] += offset
        key = f"{key_prefix}/c/" + "/".join(str(i) for i in index)
        sync(store.set(key, proto.buffer.from_bytes(literal["data"])))


def _icechunk_takes_parts() -> bool:
    """Whether virtual references can carry a prefix and several byte ranges."""
    import icechunk

    try:
        icechunk.VirtualChunkSpec(
            index=[0],
            location="s3://b/k",
            offset=0,
            length=1,
            prefix=b"",
            more_ranges=None,
        )
    except TypeError:
        return False
    return True


def _manifest_arrays(data: object) -> tuple[ManifestArray, ...]:
    """The ManifestArrays behind a variable's data: itself, its pieces, or none."""
    if isinstance(data, ManifestArray):
        return (data,)
    if isinstance(data, ConcatenatedManifestArray):
        return data.pieces
    return ()


def write_concatenated_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: xr.Variable,
    append_dim: Optional[str] = None,
    region: Optional[Mapping[str, slice]] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write a variable whose data is a ConcatenatedManifestArray into an icechunk store.

    The variable becomes one ordinary array, planned by
    [plan_joined_array][virtualizarr.manifests.plan.plan_joined_array]: its ``switch``
    codec decodes each run of chunks its own way, and where the pieces' chunk shapes
    disagree each chunk is a virtual shard. Shards need an Icechunk whose virtual
    references take a prefix and several ranges; without it, or when the pieces cannot
    be planned, every piece becomes an array at ``_pieces/<name>/<i>`` and the variable
    an array whose ``array_concat`` codec reads them.
    """
    import zarr

    from virtualizarr.manifests.plan import plan_joined_array

    if region is not None:
        raise NotImplementedError(
            f"Variable {name!r} is a ConcatenatedManifestArray, which cannot be written to a region."
        )
    if append_dim is not None:
        if append_dim not in var.dims:
            return
        if not _is_joined_array(group.get(name)):
            raise NotImplementedError(
                f"Variable {name!r} cannot be appended as pieces: the existing array {name!r} "
                "is not a joined array (it has no `switch` codec)."
            )
        append_to_joined_array(store, group, name, var, append_dim, last_updated_at)
        return
    if isinstance(group.get(name), Array):
        raise NotImplementedError(
            f"Variable {name!r} is a ConcatenatedManifestArray and the array {name!r} "
            "already exists. Write to a new group or store."
        )
    cma = cast(ConcatenatedManifestArray, var.data)
    try:
        from zarr.codecs import SwitchCodec  # type: ignore[attr-defined]

        from virtualizarr.options import OPTIONS

        plan = plan_joined_array(
            cma.pieces, cma.axis, dtype=cma.dtype, select=OPTIONS["switch_select"]
        )
        if (
            plan.strategy == "shards" or plan.select == "key"
        ) and not _icechunk_takes_parts():
            raise NotImplementedError(
                "this Icechunk cannot store a reference with a prefix or several ranges"
            )
    except (ImportError, NotImplementedError) as err:
        import warnings

        warnings.warn(
            f"Variable {name!r} is written as pieces joined by the `array_concat` codec: {err}.",
            UserWarning,
            stacklevel=2,
        )
        plan = None
    if plan is not None:
        import icechunk

        if last_updated_at is None:
            last_updated_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        # the grid is rectilinear, which zarr-python still gates behind a setting
        with zarr.config.set({"array.rectilinear_chunks": True}):
            arr = group.require_array(
                name=name,
                shape=plan.shape,
                chunks=plan.chunk_grid,
                dtype=plan.dtype,
                filters=None,
                compressors=None,
                serializer=SwitchCodec(
                    axis=plan.axis,
                    cases=plan.cases,
                    select=plan.select,
                    key_bytes=plan.key_bytes,
                ),
                dimension_names=var.dims,
                fill_value=plan.fill_value,
            )
            key_prefix = name if group.name == "/" else f"{group.name}/{name}"
            failed = store.set_virtual_refs(
                key_prefix.lstrip("/"),
                [
                    icechunk.VirtualChunkSpec(
                        **ref, last_updated_at_checksum=last_updated_at
                    )
                    for ref in plan.refs
                ],
                validate_containers=True,
            )
            if failed:
                raise ValueError(
                    f"{len(failed)} references of variable {name!r} are outside the "
                    "repository's virtual chunk containers."
                )
            _write_literals(store, key_prefix.lstrip("/"), plan.literals)
            update_attributes(arr, var.attrs, encoding=var.encoding)
        return

    try:
        from zarr.codecs import ArrayConcatCodec  # type: ignore[attr-defined]
    except ImportError as err:
        raise ImportError(
            f"Variable {name!r} joins arrays that cannot share one chunk grid, which needs "
            "the `switch` or `array_concat` codec. The installed zarr-python has neither."
        ) from err

    pieces_group = group.require_group(f"{PIECES_GROUP}/{name}")
    for i, piece in enumerate(cma.pieces):
        write_virtual_variable_to_icechunk(
            store=store,
            group=pieces_group,
            name=str(i),
            var=xr.Variable(var.dims, piece, attrs=var.attrs, encoding=var.encoding),
            last_updated_at=last_updated_at,
        )

    first = cma.pieces[0].metadata
    arr = group.require_array(
        name=name,
        shape=cma.shape,
        chunks=_reported_chunks(cma),
        dtype=first.data_type.to_native_dtype(),
        filters=None,
        compressors=None,
        serializer=ArrayConcatCodec(
            axis=cma.axis,
            pieces=[
                {"path": f"../{PIECES_GROUP}/{name}/{i}", "size": piece.shape[cma.axis]}
                for i, piece in enumerate(cma.pieces)
            ],
        ),
        dimension_names=var.dims,
        fill_value=first.fill_value,
    )
    update_attributes(arr, var.attrs, encoding=var.encoding)


def _check_existing_array_matches(
    existing: Array, metadata: ArrayV3Metadata, dimension_names: list[str]
) -> None:
    """
    Raise if an existing array has different metadata, apart from attributes, than ``metadata``.

    Raises
    ------
    ValueError
        If any metadata differs. zarr's ``require_array`` would keep the stored
        metadata, so the new refs would be decoded with the wrong codecs or read under
        the wrong dimension names.
    """
    # Compare the JSON form rather than the dataclasses: before zarr 3.3.0 the generated
    # __eq__ treats two NaN fill values as unequal (zarr-python#2929), whereas to_dict
    # serializes NaN to the string "NaN".
    # TODO: compare the metadata objects directly once zarr>=3.3.0 is the minimum version.
    requested = dataclasses.replace(
        metadata, attributes={}, dimension_names=tuple(dimension_names)
    ).to_dict()
    stored = dataclasses.replace(existing.metadata, attributes={}).to_dict()
    differing = sorted(
        key
        for key in requested.keys() | stored.keys()
        if requested.get(key) != stored.get(key)
    )
    if differing:
        raise ValueError(
            f"Array {existing.path!r} already exists with different "
            f"{', '.join(differing)}. Writing new references under the stored metadata "
            "would decode them incorrectly; use mode='w' to replace the group instead."
        )


def write_manifest_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    arr_name: str,
    manifest: ChunkManifest,
    chunk_index_offsets: tuple[int, ...],
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write all the chunks (virtual and/or inlined) for one array manifest at once.

    Virtual chunks are written as virtual chunks, and inlined chunks are written as native
    (which Icechunk may then choose to inline in its manifests).
    """

    if group.name == "/":
        key_prefix = arr_name
    else:
        key_prefix = f"{group.name}/{arr_name}"

    if last_updated_at is None:
        # Icechunk rounds timestamps to the nearest second, but filesystems have higher precision,
        # so we need to add a buffer, so that if you immediately read data back from this icechunk store,
        # and the referenced data was literally just created (<1s ago),
        # you don't get an IcechunkError warning you that your referenced chunk has changed.
        # In practice this should only really come up in synthetic examples, e.g. tests and docs.
        last_updated_at = datetime.now(timezone.utc) + timedelta(seconds=1)

    paths_flat = manifest._paths.flatten()

    if manifest._inlined:
        # Write inlined chunks first, then erase them from the paths array so the
        # virtual-refs write below doesn't see the INLINED_CHUNK_PATH sentinel
        # (which Icechunk's `.set_virtual_refs_arr` would reject as a malformed URL).
        # Use of zarr's `sync` here is to avoid a serial high-latency loop over chunks.
        # Would prefer if zarr-python had a public API for setting many chunks at once concurrently.
        sync(
            write_inlined_chunks_as_native(
                store=store,
                key_prefix=key_prefix,
                inlined=manifest._inlined,
                chunk_index_offsets=chunk_index_offsets,
            )
        )
        virtual_paths = np.where(paths_flat == INLINED_CHUNK_PATH, "", paths_flat)
    else:
        virtual_paths = paths_flat

    # Cheap numpy-level check so we can skip the .tolist() allocation and the
    # Python->Rust call entirely when no position holds a real virtual ref
    # (e.g. an all-inlined or all-missing manifest).
    if (virtual_paths != "").any():
        # Pass flat per-chunk arrays (or a list) to Rust in one call, avoiding Python-side
        # per-chunk dict construction. Empty paths are skipped on the Rust side.
        store.set_virtual_refs_arr(
            array_path=key_prefix,
            chunk_grid_shape=manifest.shape_chunk_grid,
            locations=virtual_paths.tolist(),
            offsets=manifest._offsets.flatten(),
            lengths=manifest._lengths.flatten(),
            validate_containers=False,
            arr_offset=chunk_index_offsets if any(chunk_index_offsets) else None,
            checksum=last_updated_at,
        )


async def write_inlined_chunks_as_native(
    store: "IcechunkStore",
    key_prefix: str,
    inlined: Mapping[tuple[int, ...], bytes],
    chunk_index_offsets: tuple[int, ...],
) -> None:
    """Write each inlined chunk as a native chunk at its zarr chunk key.

    Icechunk's ``Key::parse`` only accepts the standard zarr v3 ``/``-separated
    ``c/i0/i1/...`` chunk-key form; the manifest's stored chunk_key_encoding
    uses ``.`` (the ManifestStore's internal convention), so we encode with a
    slash-separated encoding here regardless.
    """
    encoding = DefaultChunkKeyEncoding(separator="/")
    prototype = default_buffer_prototype()
    has_offset = any(chunk_index_offsets)
    coros = []
    for chunk_idx, data in inlined.items():
        shifted_idx = (
            tuple(c + o for c, o in zip(chunk_idx, chunk_index_offsets))
            if has_offset
            else chunk_idx
        )
        encoded_chunk_key = encoding.encode_chunk_key(shifted_idx)
        coros.append(
            store.set(
                f"{key_prefix}/{encoded_chunk_key}",
                prototype.buffer.from_bytes(data),
            )
        )
    await asyncio.gather(*coros)
