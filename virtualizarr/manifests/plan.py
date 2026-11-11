"""Plan one Zarr array from pieces whose layouts do not match."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from virtualizarr.manifests.utils import manifest_chunk_shape

if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray

MISSING = 2**64 - 1
KEY_BYTES = 8


def _edges(n: int, c: int) -> list[int]:
    return [c] * (n // c) + ([n % c] if n % c else [])


@dataclass
class JoinedArrayPlan:
    """How a set of pieces becomes one Zarr array.

    Attributes
    ----------
    strategy
        ``"switch"``: every source chunk is one Zarr chunk on a rectilinear grid, and a
        ``switch`` codec picks the codec chain per run of chunks and crops chunks stored
        larger than their place. ``"shards"``: the pieces' chunk shapes disagree, so each
        Zarr chunk is a virtual shard, a literal ``sharding_indexed`` index followed by
        the source chunks it holds.
    refs
        One dict per Zarr chunk, with the arguments of ``icechunk.VirtualChunkSpec``.
    """

    strategy: str
    axis: int
    shape: tuple[int, ...]
    dtype: np.dtype
    fill_value: Any
    chunk_grid: list[list[int]]
    cases: list[dict]
    refs: list[dict] = field(default_factory=list)
    source_chunks: int = 0
    notes: dict = field(default_factory=dict)
    select: str = "position"
    key_bytes: int = KEY_BYTES


def plan_joined_array(
    pieces: Sequence["ManifestArray"],
    axis: int,
    dtype: np.dtype | None = None,
    shard_length_along_axis: int | None = None,
    fill_value: Any = None,
    like: dict | None = None,
    select: str = "position",
) -> JoinedArrayPlan:
    """
    Plan one array from pieces joined along `axis`.

    The pieces may differ in codecs, chunk shape, fill value and data type, and a piece
    may end in a chunk that is cut short. They must agree on their extent off the axis.

    Parameters
    ----------
    select
        How the `switch` codec finds a chunk's case. ``"position"``: from the chunk's
        place along the joined axis; nothing is stored per chunk, and released Icechunk
        can hold the array unless it needs shards. ``"key"``: from a key in front of
        the chunk's bytes, which every reference carries as a prefix; chunks describe
        themselves and the case table has one entry per distinct layout, but the store
        must take references with a prefix.
    shard_length_along_axis
        For virtual shards, how far a shard reaches along the joined axis, when the
        pieces' chunks along it allow more than one choice. A shard may then hold
        source chunks of several files (daily maps grouped by the week, say), which
        needs an Icechunk whose references can name several objects.

    fill_value
        The joined array's fill value. By default the most common among the pieces.
    like
        To plan pieces that will be appended to an existing joined array: that array's
        ``{"strategy": ..., "chunk_grid": [...]}``. The plan then uses the same
        strategy and the same chunks (or shards) off the joined axis.

    Raises
    ------
    NotImplementedError
        If the pieces cannot be planned (their extents differ off the axis, or a piece
        that would go inside a shard is itself sharded).
    """
    from zarr.core.dtype import get_data_type_from_native_dtype

    ndim = pieces[0].ndim
    others = [d for d in range(ndim) if d != axis]
    if any(p.shape[d] != pieces[0].shape[d] for p in pieces for d in others):
        raise NotImplementedError("the pieces' extents differ off the joined axis")
    shape = list(pieces[0].shape)
    shape[axis] = sum(p.shape[axis] for p in pieces)
    chunks = [tuple(manifest_chunk_shape(p.metadata)) for p in pieces]

    dtype = np.dtype(
        dtype if dtype is not None else np.result_type(*[p.dtype for p in pieces])
    )
    zdtype = get_data_type_from_native_dtype(dtype)
    fills = [dtype.type(p.metadata.fill_value) for p in pieces]
    keys = [f.tobytes() for f in fills]
    # the most common fill value; among equals, the first piece's
    array_fill = fills[max(range(len(keys)), key=lambda i: (keys.count(keys[i]), -i))]
    if fill_value is not None:
        array_fill = dtype.type(fill_value)

    notes: dict = {
        "distinct_chunk_shapes": sorted(set(chunks)),
        "distinct_codec_chains": len(
            {str(p.metadata.to_dict()["codecs"]) for p in pieces}
        ),
        "distinct_data_types": sorted({str(p.dtype) for p in pieces}),
        "pieces_with_their_own_fill": 0,
    }
    cases: list[dict] = []

    def add_case(n: int, case: dict) -> bytes:
        """Record a case for `n` more chunks. Returns the key its chunks begin with,
        which is empty when cases are found by position."""
        if select == "key":
            # the key names the case by its content, so equal cases share one entry
            canonical = json.dumps(case, sort_keys=True, separators=(",", ":"))
            key = hashlib.sha256(canonical.encode()).digest()[:KEY_BYTES]
            if not any(c["key"] == key.hex() for c in cases):
                cases.append({"key": key.hex(), **case})
            return key
        if cases and {k: v for k, v in cases[-1].items() if k != "chunks"} == case:
            cases[-1]["chunks"] += n
        else:
            cases.append({"chunks": n, **case})
        return b""

    def codecs_of(p: "ManifestArray") -> list[dict]:
        """The piece's chain, converting its stored data type to the array's first."""
        chain = [
            dict(c) if isinstance(c, dict) else c
            for c in p.metadata.to_dict()["codecs"]
        ]
        if p.dtype != dtype:
            configuration: dict = {
                "data_type": p.metadata.data_type.to_json(zarr_format=3)
            }
            if p.dtype.kind in "iu" and dtype.kind == "f" and np.isnan(array_fill):
                # the array's fill value must have a counterpart in the stored type
                configuration["scalar_map"] = {
                    "encode": [["NaN", int(p.metadata.fill_value)]]
                }
            chain = [{"name": "cast_value", "configuration": configuration}, *chain]
        return json.loads(json.dumps(chain, default=str))

    def own_fill(p: "ManifestArray", fill: Any) -> dict:
        """A case-level fill value, where the piece leaves chunks unstored and its fill
        is not the array's. (Where every chunk is stored the fill is never read.)"""
        if fill.tobytes() == array_fill.tobytes() or bool(
            (p.manifest._paths != "").all()
        ):
            return {}
        if select == "key":
            # a chunk that stores nothing has no reference to carry a key
            raise NotImplementedError(
                "a piece leaves chunks unstored and its fill value is not the array's; "
                "that needs select='position'"
            )
        notes["pieces_with_their_own_fill"] += 1
        return {"fill_value": zdtype.to_json_scalar(fill, zarr_format=3)}

    def paths_of(p: "ManifestArray") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = p.manifest
        return np.asarray(m._paths), np.asarray(m._offsets), np.asarray(m._lengths)

    grid: list[list[int]] = [[] for _ in range(ndim)]
    same_off_axis = all(c[d] == chunks[0][d] for c in chunks for d in others)
    if like is not None:
        if like["strategy"] == "switch" and not (
            same_off_axis
            and all(
                like["chunk_grid"][d] == _edges(shape[d], chunks[0][d]) for d in others
            )
        ):
            raise NotImplementedError(
                "the new pieces' chunks differ off the joined axis from the existing array's, "
                "which stores plain chunks; the whole record has to be planned again"
            )
        same_off_axis = like["strategy"] == "switch"
    if same_off_axis:
        for d in others:
            grid[d] = _edges(shape[d], chunks[0][d])
        out = JoinedArrayPlan(
            "switch",
            axis,
            tuple(shape),
            dtype,
            array_fill,
            grid,
            cases,
            notes=notes,
            select=select,
        )
        base = 0
        for p, c, fill in zip(pieces, chunks, fills):
            along = _edges(p.shape[axis], c[axis])
            encoded: list[int | None] = list(c)
            if p.shape[axis] % c[axis] == 0 and select == "position":
                # every chunk is whole: as large as its place. (By key the shape is
                # spelled out, so pieces with and without a cut chunk share a case.)
                encoded[axis] = None
            key = add_case(
                len(along),
                {
                    "codecs": codecs_of(p),
                    "encoded_chunk_shape": encoded,
                    **own_fill(p, fill),
                },
            )
            grid[axis] += along
            paths, offsets, lengths = paths_of(p)
            for idx in zip(*np.nonzero(paths != "")):
                index = [int(i) for i in idx]
                index[axis] += base
                ref: dict = {
                    "index": index,
                    "location": str(paths[idx]),
                    "offset": int(offsets[idx]),
                    "length": int(lengths[idx]),
                }
                if key:
                    ref["prefix"] = key
                out.refs.append(ref)
            out.source_chunks += int((paths != "").sum())
            base += len(along)
        return out

    # The chunk shapes disagree: shards sized to a common multiple where that is small,
    # and to the whole extent otherwise.
    for p in pieces:
        if any(
            (c.get("name") if isinstance(c, dict) else None) == "sharding_indexed"
            for c in p.metadata.to_dict()["codecs"]
        ):
            raise NotImplementedError(
                "a sharded piece cannot go inside a virtual shard"
            )
    lcm_axis = math.lcm(*[c[axis] for c in chunks])
    shard_axis = lcm_axis if lcm_axis <= min(p.shape[axis] for p in pieces) else None
    if shard_axis is not None and shard_length_along_axis:
        if shard_length_along_axis % lcm_axis:
            raise ValueError(
                f"shard_length_along_axis must be a multiple of {lcm_axis}"
            )
        shard_axis = shard_length_along_axis
    shard = [0] * ndim
    for d in others:
        lcm = math.lcm(*[c[d] for c in chunks])
        shard[d] = lcm if lcm <= shape[d] else shape[d]
        if like is not None:
            # keep the existing array's shards; they must hold whole source chunks
            shard[d] = like["chunk_grid"][d][0]
            if like["chunk_grid"][d] != _edges(shape[d], shard[d]) or (
                shard[d] != shape[d] and any(shard[d] % c[d] for c in chunks)
            ):
                raise NotImplementedError(
                    "the new pieces' chunks do not fit the existing array's shards; "
                    "the whole record has to be planned again"
                )
        grid[d] = _edges(shape[d], shard[d])
    notes["shard_shape"] = [
        ("one per piece" if shard_axis is None else shard_axis)
        if d == axis
        else shard[d]
        for d in range(ndim)
    ]
    out = JoinedArrayPlan(
        "shards",
        axis,
        tuple(shape),
        dtype,
        array_fill,
        grid,
        cases,
        notes=notes,
        select=select,
    )
    base = 0
    for p, c, fill in zip(pieces, chunks, fills):
        length = list(shard)
        length[axis] = shard_axis or p.shape[axis]
        encoded_shape = [math.ceil(length[d] / c[d]) * c[d] for d in range(ndim)]
        per_shard = tuple(encoded_shape[d] // c[d] for d in range(ndim))
        along = _edges(p.shape[axis], length[axis])
        n_shards = tuple(len(along) if d == axis else len(grid[d]) for d in range(ndim))
        index_nbytes = 16 * int(np.prod(per_shard))
        sharding = {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": list(c),
                "codecs": codecs_of(p),
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}}
                ],
                "index_location": "start",
            },
        }
        # Neighbours in a virtual shard are separate source chunks, so ranges are never merged.
        key = add_case(
            len(along),
            {
                "codecs": [sharding],
                "encoded_chunk_shape": encoded_shape,
                "coalesce_max_gap_bytes": 0,
                **own_fill(p, fill),
            },
        )
        grid[axis] += along
        paths, offsets, lengths = paths_of(p)
        for s in np.ndindex(n_shards):
            index = np.full((*per_shard, 2), MISSING, dtype="<u8")
            parts: list[tuple[str, int, int]] = []
            pos = index_nbytes
            for inner in np.ndindex(per_shard):
                src = tuple(s[d] * per_shard[d] + inner[d] for d in range(ndim))
                if all(x < n for x, n in zip(src, paths.shape)) and paths[src]:
                    n = int(lengths[src])
                    index[inner] = (pos, n)
                    parts.append((str(paths[src]), int(offsets[src]), n))
                    pos += n
            if parts:
                where = [int(i) for i in s]
                where[axis] += base
                first = parts[0][0]
                out.refs.append(
                    {
                        "index": where,
                        "location": first,
                        "offset": parts[0][1],
                        "length": parts[0][2],
                        # the key, when there is one, then the shard index, whose
                        # offsets count from after the key
                        "prefix": key + index.tobytes(),
                        # a part in another object than the first names it; the rest say None
                        "more_ranges": [
                            (None if loc == first else loc, o, n)
                            for loc, o, n in parts[1:]
                        ]
                        or None,
                    }
                )
                out.source_chunks += len(parts)
        base += len(along)
    return out
