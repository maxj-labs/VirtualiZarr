"""Global options, set for a block with ``with set_options(...)`` or for the session."""

from typing import Any, Literal, TypedDict


class _Options(TypedDict):
    mismatched_concat: Literal["raise", "pieces"]
    switch_select: Literal["position", "key"]
    switch_key_bytes: int
    switch_parameters: bool


OPTIONS: _Options = {
    "mismatched_concat": "raise",
    "switch_select": "position",
    "switch_key_bytes": 8,
    "switch_parameters": True,
}

_VALID: dict[str, tuple[Any, ...]] = {
    "mismatched_concat": ("raise", "pieces"),
    "switch_select": ("position", "key"),
    "switch_key_bytes": (1, 2, 4, 8, 16, 32),
    "switch_parameters": (True, False),
}


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

    switch_select
        How a joined array's ``switch`` codec finds each chunk's case:
        ``"position"`` (default) from the chunk's place along the joined axis, or
        ``"key"`` from a key that every chunk reference carries as a prefix, so that a
        chunk describes itself (see
        [plan_joined_array][virtualizarr.manifests.plan.plan_joined_array]).

    switch_key_bytes
        The length of the key when ``switch_select="key"``, 8 by default. A key is the
        leading bytes of the hash of its case, whatever its length; a shorter key saves
        space in every reference and makes it likelier that two cases share a key, which
        is refused when the array is planned.

    switch_parameters
        With ``switch_select="key"``, whether files that differ only in the numbers of a
        ``scale_offset`` codec share one case, each chunk reference carrying its own
        numbers after the key (default True).

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
