"""Tee stdout/stderr to a timestamped log file under a run's artifacts dir.

The sweep/compare/train scripts print useful run details (streamed doc
counts per language, training progress, chosen vocab size) that would
otherwise only live in a terminal's scrollback. Teeing them into
`<out-dir>/logs/` keeps that record alongside the CSVs/report the run
produced, so e.g. "how many docs did the last comparison run actually
see" is answerable from artifacts/ instead of "re-run and watch closely".
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_to_log(out_dir: Path, name: str) -> Iterator[Path]:
    """Mirror stdout+stderr to `out_dir/logs/{name}_{UTC timestamp}.log`
    for the duration of the `with` block, in addition to the console."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{name}_{timestamp}.log"

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(orig_stdout, log_file)
        sys.stderr = _Tee(orig_stderr, log_file)
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
    print(f"Wrote {log_path}")
