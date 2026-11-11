"""Global options, set for a block with ``with set_options(...)`` or for the session."""

from typing import Any, Literal, TypedDict


class _Options(TypedDict):
    mismatched_concat: Literal["raise", "pieces"]


OPTIONS: _Options = {"mismatched_concat": "raise"}

_VALID: dict[str, tuple[Any, ...]] = {"mismatched_concat": ("raise", "pieces")}


class set_options:
    """
    Set VirtualiZarr options, for the session or inside a ``with`` block.

    Parameters
    ----------
    mismatched_concat
        What concatenation does when the arrays cannot share one chunk grid, because their
        codecs, chunk shapes or fill values differ, or because an array before the last one
        ends in a partial chunk.

        - ``"raise"`` (default): raise an error.
        - ``"pieces"``: keep each run of compatible arrays as a piece of a
          [ConcatenatedManifestArray][virtualizarr.manifests.ConcatenatedManifestArray].
          It is written to Icechunk as one array: on a rectilinear chunk grid with a
          ``switch`` codec where the pieces agree on their chunk shape off the joined
          axis, and as virtual shards where they do not (see
          [plan_joined_array][virtualizarr.manifests.plan.plan_joined_array]).

    Examples
    --------
    >>> with set_options(mismatched_concat="pieces"):  # doctest: +SKIP
    ...     vds = xr.concat([vds_2019, vds_2020], dim="time")
    """

    def __init__(self, **kwargs: Any) -> None:
        self._old: dict[str, Any] = {}
        for name, value in kwargs.items():
            if name not in OPTIONS:
                raise ValueError(
                    f"{name!r} is not an option. The options are {sorted(OPTIONS)}."
                )
            if value not in _VALID[name]:
                raise ValueError(
                    f"{name} must be one of {_VALID[name]}. Got {value!r}."
                )
            self._old[name] = OPTIONS[name]  # type: ignore[literal-required]
        OPTIONS.update(kwargs)  # type: ignore[typeddict-item]

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        OPTIONS.update(self._old)  # type: ignore[typeddict-item]
