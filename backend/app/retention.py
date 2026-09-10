from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .store import Store

log = logging.getLogger("fp.backend.retention")

_INTERVAL_S = 3600


async def _sweep(store: Store, overlay_dir: Path, retention_days: int) -> None:
    cutoff = time.time() - retention_days * 86400
    removed_paths = await store.prune_older_than(cutoff)
    files_deleted = 0
    for rel in removed_paths:
        p = overlay_dir / Path(rel).name
        try:
            p.unlink(missing_ok=True)
            files_deleted += 1
        except OSError as exc:  # pragma: no cover
            log.warning("could not delete overlay %s: %s", p, exc)
    if removed_paths:
        log.info("retention: pruned %d rows, %d overlay files", len(removed_paths), files_deleted)


async def retention_loop(store: Store, overlay_dir: Path, retention_days: int) -> None:
    while True:
        try:
            await _sweep(store, overlay_dir, retention_days)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            log.exception("retention sweep failed")
        await asyncio.sleep(_INTERVAL_S)
