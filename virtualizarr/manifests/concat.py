"""Concatenation of ManifestArrays that cannot share one chunk grid."""

from typing import TYPE_CHECKING, Any, Callable, Sequence, Union

import numpy as np

from virtualizarr.manifests.utils import (
    check_same_dtypes,
    check_same_ndims,
    check_same_shapes_except_on_concat_axis,
    get_codecs,
    manifest_chunk_shape,
)

if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray


class ConcatenatedManifestArray:
    """
    Several ManifestArrays joined end to end along one axis, each kept as it is.

    A [ManifestArray][virtualizarr.manifests.ManifestArray] describes one Zarr array, so
    all of its chunks have one shape, one codec chain and one fill value. Arrays that
    differ in any of those cannot be merged into one. This class joins them without
    merging: every piece stays a ManifestArray, with its own chunk grid and codecs.

    It is written to Icechunk as one array per piece plus one array, under the
    variable's name, whose ``array_concat`` codec reads the pieces. That codec is not
    part of released zarr-python.

    It is created by concatenating ManifestArrays inside
    ``virtualizarr.set_options(mismatched_concat="pieces")``, not directly. It can be
    concatenated again along the same axis, and it cannot be indexed, stacked or
    broadcast.

    Parameters
    ----------
    pieces
        The arrays, in order. They have the same data type and the same shape on every
        axis but `axis`.
    axis
        The axis along which the pieces are joined.
    """

    def __init__(self, pieces: Sequence["ManifestArray"], axis: int) -> None:
        from virtualizarr.manifests.array import ManifestArray

        pieces = tuple(pieces)
        if len(pieces) < 2:
            raise ValueError("A ConcatenatedManifestArray has at least two pieces.")
        for piece in pieces:
            if not isinstance(piece, ManifestArray):
                raise TypeError(
                    f"Every piece must be a ManifestArray. Got {type(piece)}."
                )
        check_same_dtypes([piece.dtype for piece in pieces])
        check_same_ndims([piece.ndim for piece in pieces])
        if not 0 <= axis < pieces[0].ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for arrays of {pieces[0].ndim} dimensions."
            )
        check_same_shapes_except_on_concat_axis([piece.shape for piece in pieces], axis)
        self._pieces = pieces
        self._axis = axis

    @property
    def pieces(self) -> tuple["ManifestArray", ...]:
        return self._pieces

    @property
    def axis(self) -> int:
        return self._axis

    @property
    def dtype(self) -> np.dtype:
        return self._pieces[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        shape = list(self._pieces[0].shape)
        shape[self._axis] = sum(piece.shape[self._axis] for piece in self._pieces)
        return tuple(shape)

    @property
    def ndim(self) -> int:
        return self._pieces[0].ndim

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes_virtual(self) -> int:
        """The number of bytes that hold these virtual references in memory."""
        return sum(piece.nbytes_virtual for piece in self._pieces)

    def __repr__(self) -> str:
        sizes = [piece.shape[self._axis] for piece in self._pieces]
        return (
            f"ConcatenatedManifestArray<shape={self.shape}, dtype={self.dtype}, "
            f"axis={self._axis}, pieces={len(self._pieces)}, sizes={sizes}>"
        )

    def __array_function__(self, func, types, args, kwargs) -> Any:
        from virtualizarr.manifests.array import ManifestArray
        from virtualizarr.manifests.array_api import (
            MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS,
        )

        if func not in (np.concatenate, np.result_type):
            return NotImplemented
        if not all(
            issubclass(t, (ConcatenatedManifestArray, ManifestArray, np.ndarray))
            for t in types
        ):
            return NotImplemented
        return MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        """Defined so that xarray treats this class as a duck array."""
        from virtualizarr.manifests.array_api import _isnan

        if ufunc == np.isnan:
            return _isnan(self.shape)
        return NotImplemented

    def __array__(
        self, dtype: np.typing.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        raise NotImplementedError(
            "ConcatenatedManifestArray holds virtual references to chunks in archival "
            "files and cannot be converted into a numpy array. Write the virtual dataset "
            "to an Icechunk store and reopen it to read values."
        )

    def astype(
        self, dtype: np.dtype, /, *, copy: bool = True
    ) -> "ConcatenatedManifestArray":
        """Cannot change the dtype, but xarray calls this even when it is a no-op."""
        if not np.issubdtype(self.dtype, dtype):
            raise NotImplementedError()
        return self

    def __getitem__(self, key: Any, /) -> "ConcatenatedManifestArray":
        keys = key if isinstance(key, tuple) else (key,)
        if all(k is Ellipsis or k == slice(None) for k in keys):
            return self
        raise NotImplementedError(
            "A ConcatenatedManifestArray cannot be indexed. Index the datasets before "
            "concatenating them."
        )

    def rename_paths(
        self, new: str | Callable[[str], str]
    ) -> "ConcatenatedManifestArray":
        """Rename the paths to chunks in every piece's manifest."""
        return ConcatenatedManifestArray(
            [piece.rename_paths(new) for piece in self._pieces], axis=self._axis
        )


def _same_fill_value(a: Any, b: Any) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _can_extend(run: list["ManifestArray"], nxt: "ManifestArray", axis: int) -> bool:
    """Whether `nxt` can be merged onto the end of `run` as one Zarr array."""
    first, last = run[0], run[-1]
    if get_codecs(first) != get_codecs(nxt):
        return False
    if manifest_chunk_shape(first.metadata) != manifest_chunk_shape(nxt.metadata):
        return False
    if not _same_fill_value(first.metadata.fill_value, nxt.metadata.fill_value):
        return False
    # a partial chunk is only valid at the end of an array
    return last.shape[axis] % manifest_chunk_shape(last.metadata)[axis] == 0


def concatenate_keeping_pieces(
    arrays: Sequence[Union["ManifestArray", ConcatenatedManifestArray]],
    axis: int,
    merge: Callable[[list["ManifestArray"], int], "ManifestArray"],
) -> Union["ManifestArray", ConcatenatedManifestArray]:
    """
    Concatenate arrays, merging each run of compatible neighbours and keeping the runs
    that cannot be merged as separate pieces.

    Parameters
    ----------
    arrays
        The arrays to join, in order.
    axis
        The (non-negative) axis to join them along.
    merge
        Concatenates ManifestArrays that share a chunk grid, codecs and fill value.

    Returns
    -------
    A ManifestArray when everything merged into one run, else a ConcatenatedManifestArray.
    """
    flat: list["ManifestArray"] = []
    for arr in arrays:
        if isinstance(arr, ConcatenatedManifestArray):
            if arr.axis != axis:
                raise NotImplementedError(
                    f"Cannot concatenate along axis {axis} an array whose pieces are "
                    f"joined along axis {arr.axis}."
                )
            flat.extend(arr.pieces)
        else:
            flat.append(arr)

    check_same_dtypes([arr.dtype for arr in flat])
    check_same_ndims([arr.ndim for arr in flat])
    check_same_shapes_except_on_concat_axis([arr.shape for arr in flat], axis)

    runs: list[list["ManifestArray"]] = [[flat[0]]]
    for arr in flat[1:]:
        if _can_extend(runs[-1], arr, axis):
            runs[-1].append(arr)
        else:
            runs.append([arr])

    merged = [run[0] if len(run) == 1 else merge(run, axis) for run in runs]
    if len(merged) == 1:
        return merged[0]
    return ConcatenatedManifestArray(merged, axis=axis)
