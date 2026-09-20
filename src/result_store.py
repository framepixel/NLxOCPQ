#!/usr/bin/env python3
"""Content-addressed store for OCPQ root result sets.

The 36-condition matrix costs 40-100 GPU-hours. Re-deriving a metric afterwards must not
require re-running it, so every root result set produced during a run is persisted here in
the reduced form the metric actually consumes.

**Why not keep the raw CLI exports.** Measured on real gold queries, an ocpq_cli result file
runs 20-470x larger than the extracted IDs -- one 53,399-situation query produced a 745 MB
export whose extracted root set is 152 KB gzipped. Keeping the raw files for the full matrix
would exceed the whole cluster storage budget; keeping the extracted sets costs ~100-500 MB.

**What is stored.** Only the root situation set, as a sorted list of
`[[event_indices], [object_indices], violated]` -- exactly the third component of
`eval_testset._extract_result_ids`.

`event_ids` and `object_ids` are deliberately *not* stored. In `_extract_result_ids` they are
accumulated in the same loop, under the same guards, as the situation keys, so each is exactly
the union of the corresponding component across situations. Deriving them on load makes the
situation set the minimal sufficient statistic for the whole similarity triple, and removes any
chance of the three drifting apart in storage.

The per-situation `violated` flag is kept rather than aggregated to a count. It is what allows
the result metric to be decomposed later -- population F1 (flag dropped: did the query select
the right rows?) versus violation F1 (flagged situations only: did it catch the right
violations?) -- without re-executing anything. That decomposition matters because for a
compliance query the trivial floor is `1 - violation_rate` (median error 0.0000 across the 154
compliance entries), so a rarely-violated rule compresses the entire scoring range into the top
few percent.

**Child nodes are not stored.** They cost 1.9x-132x the root (one entry: 7,270,674 child
situations behind a 53,399-situation root) and the comparison they would enable is not well
defined -- child *i* of a prediction is not child *i* of gold when tree shapes differ, which is
why the metric is root-only in the first place.
"""
import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1

# Above this many situations a blob records cardinality and `truncated: true` instead of the
# members. Gold never comes close (the largest entry in the testset is 53,399); the cap exists
# only to bound a pathological prediction, which ocpq_cli itself caps at MAX_NUM_BINDINGS =
# 10,000,000. A truncated blob is reported as *unavailable* on recompute rather than scored,
# because a comparison against a partial set is meaningless.
DEFAULT_MAX_SITUATIONS = 200_000


def query_key(ocel_path: str, query: Any) -> str:
    """Content address for a `(log, query)` pair.

    Mirrors `eval_testset._hash_query_key` -- the same canonical form, so a blob and the
    ground-truth cache agree on what counts as the same query. Kept here rather than imported
    so this module stays usable without pulling in the evaluation harness.
    """
    payload = json.dumps(query, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{ocel_path}::{digest}"


def blob_id(ocel_path: str, query: Any) -> str:
    """Filesystem-safe hash of `query_key` (which contains path separators)."""
    return hashlib.sha256(query_key(ocel_path, query).encode("utf-8")).hexdigest()


def blob_path(store_dir, ocel_path: str, query: Any) -> Path:
    return _path_for_id(store_dir, blob_id(ocel_path, query))


def _path_for_id(store_dir, bid: str) -> Path:
    # Two-level fan-out: a flat directory would hold one entry per distinct query across the
    # whole matrix, which is slow to list and unpleasant on shared cluster filesystems.
    return Path(store_dir) / bid[:2] / f"{bid}.json.gz"


def _encode_situations(situations) -> list:
    """Canonical, sorted, JSON-ready form of a situation set.

    Sorting makes a blob byte-stable for a given set, so an idempotent re-write produces an
    identical file and two shards that computed the same result cannot disagree.
    """
    encoded = []
    for situation in situations:
        events, objects, violated = situation
        encoded.append([list(events), list(objects), bool(violated)])
    encoded.sort(key=lambda item: (item[0], item[1], item[2]))
    return encoded


def _decode_situations(encoded) -> set:
    """Inverse of `_encode_situations`, restoring the exact key type the metric compares.

    `_binding_entity_indices` yields tuples, and `_set_metrics` compares set members by
    equality, so the round trip must restore tuples rather than lists -- a list is unhashable
    and a mismatched type would silently score every comparison as disjoint.
    """
    return {
        (tuple(events), tuple(objects), bool(violated))
        for events, objects, violated in encoded
    }


def store(
    store_dir,
    ocel_path: str,
    query: Any,
    ids: Optional[tuple],
    bindings_skipped: Optional[bool] = None,
    max_situations: int = DEFAULT_MAX_SITUATIONS,
) -> Optional[str]:
    """Persist one root result set. Returns the blob id, or None when there was nothing to store.

    Idempotent: an existing blob is left alone, so this is safe under `--resume` and under
    concurrent shards. Writes go through a temp file and `os.replace`, so a reader never observes
    a partial blob and two writers racing on the same id leave one valid file.

    Args:
        store_dir: Root of the store.
        ocel_path: Log the query ran against; part of the content address.
        query: The query that produced `ids`.
        ids: The `(event_ids, object_ids, situations)` triple from `_extract_result_ids`.
            None (query did not execute) stores nothing.
        bindings_skipped: The CLI's truncation flag, carried through so a recomputed metric can
            still identify results that were incomplete when produced.
        max_situations: Cap above which members are dropped and `truncated` is set.
    """
    if not ids:
        return None
    situations = ids[2]
    if situations is None:
        return None

    bid = blob_id(ocel_path, query)
    path = _path_for_id(store_dir, bid)
    if path.exists():
        return bid

    n = len(situations)
    truncated = n > max_situations
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ocel_path": ocel_path,
        "query_sha256": query_key(ocel_path, query).rsplit("::", 1)[-1],
        "n_situations": n,
        "bindings_skipped": bindings_skipped,
        "truncated": truncated,
        "situations": [] if truncated else _encode_situations(situations),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per writer, not per process: two threads in one process share a PID, so a
    # PID-keyed temp name lets them clobber each other's partial file and one of the two
    # `os.replace` calls then fails on Windows with a sharing violation.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(payload, f, separators=(",", ":"))
        try:
            os.replace(tmp_path, path)
        except OSError:
            # Lost a race: another writer published the same blob first, and on Windows
            # replacing a file that is open for reading raises. The content is identical by
            # construction (same content address, canonical encoding), so the other writer's
            # copy is just as good -- succeed rather than fail a run over it.
            if not path.exists():
                raise
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return bid


def load_blob(store_dir, ocel_path: str, query: Any) -> Optional[dict]:
    """Read a blob's raw payload, or None when it is absent or unreadable."""
    path = blob_path(store_dir, ocel_path, query)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load(store_dir, ocel_path: str, query: Any) -> Optional[tuple]:
    """Reconstruct the `(event_ids, object_ids, situations)` triple `_extract_result_ids` returns.

    Returns None when the blob is missing or truncated. Truncated must not be scored: a
    comparison against a partial set would be silently wrong, which is worse than absent.
    """
    return load_with_meta(store_dir, ocel_path, query)[0]


def load_with_meta(store_dir, ocel_path: str, query: Any) -> tuple:
    """`(ids, bindings_skipped)` for a stored result, or `(None, None)` when unusable.

    Exists for the evaluation's resume path, which needs the truncation flag alongside the ids
    to rebuild a ground-truth cache entry without re-executing the query. `load` alone drops the
    flag, and re-deriving it would mean reading the blob twice.
    """
    payload = load_blob(store_dir, ocel_path, query)
    if payload is None or payload.get("truncated"):
        return None, None
    situations = _decode_situations(payload.get("situations") or [])
    event_ids: set = set()
    object_ids: set = set()
    for events, objects, _violated in situations:
        event_ids.update(events)
        object_ids.update(objects)
    return (event_ids, object_ids, situations), payload.get("bindings_skipped")


def union_into(dest_dir, source_dirs) -> int:
    """Merge shard stores into one. Returns the number of blobs copied.

    Safe by construction: the filename is a hash of the content address, so an id present in two
    shards refers to the same `(log, query)` pair and the blobs agree. Existing blobs are left
    alone rather than compared, which keeps the merge O(new blobs).
    """
    dest_dir = Path(dest_dir)
    copied = 0
    for source in source_dirs:
        source = Path(source)
        if not source.is_dir():
            continue
        for src_path in source.glob("*/*.json.gz"):
            dst_path = dest_dir / src_path.parent.name / src_path.name
            if dst_path.exists():
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = dst_path.with_name(f"{dst_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            tmp_path.write_bytes(src_path.read_bytes())
            try:
                os.replace(tmp_path, dst_path)
            except OSError:
                if not dst_path.exists():
                    raise
                tmp_path.unlink(missing_ok=True)
                continue
            copied += 1
    return copied


def stats(store_dir) -> dict:
    """Blob count and on-disk size, for reporting what a run's store actually cost."""
    store_dir = Path(store_dir)
    if not store_dir.is_dir():
        return {"blobs": 0, "bytes": 0}
    total = 0
    count = 0
    for path in store_dir.glob("*/*.json.gz"):
        total += path.stat().st_size
        count += 1
    return {"blobs": count, "bytes": total}
