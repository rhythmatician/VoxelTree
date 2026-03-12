"""Small helpers for emitting progress notifications to stdout.

These functions are intentionally lightweight so that non-GUI users are not
forced to depend on ``tqdm`` or any extra libraries.  The GUI listens for
percentage values anywhere in the child process output and converts them to
circular progress indicators.

Usage examples::

    # simple in-place call inside a loop
    for i, x in enumerate(things):
        ...
        progress.report(i, len(things))

    # or wrap an iterable to get implicit reporting
    for i, item in progress.wrap(enumerable, desc="Doing thing"):
        ...

The printed lines look like ``"[PROGRESS] 42%"`` which is recognised by
``RunWorker``'s regex and results in a progress signal to the GUI.
"""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def report(current: int, total: int) -> None:
    """Write a progress message for ``current`` of ``total``.

    The message is printed only when ``total > 0``; the percentage is rounded
    to the nearest integer and clamped to [0,100].  A trailing newline is
    included so that ``RunWorker`` can capture it as a distinct line.
    """
    if total <= 0:
        return
    pct = int((current / total) * 100)
    pct = max(0, min(100, pct))
    print(f"[PROGRESS] {pct}%")


def wrap(iterable: Iterable[T], *, desc: str | None = None) -> Iterator[T]:
    """Yield items from *iterable*, printing periodic progress reports.

    ``desc`` is ignored by this simple helper, but included for API compatibility
    with ``tqdm`` so that callers can swap between them easily.
    """
    # convert to sequence if possible to determine length
    try:
        total = len(iterable)  # type: ignore[arg-type]
    except Exception:
        total = 0
    for idx, item in enumerate(iterable):
        if total:
            report(idx, total)
        yield item
    if total:
        report(total, total)
