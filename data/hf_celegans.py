"""Download and ingest the homogenized C. elegans activity corpus.

Source: Simeon et al. 2024 (arXiv:2411.12091),
https://huggingface.co/datasets/qsimeon/celegans_neural_data (MIT).

Long-format parquet rows: ``source_dataset, raw_data_file, worm, neuron,
slot, is_labeled_neuron, ..., calcium_data[], time_in_seconds[], ...``
(common resampling dt = 0.333 s -> ~3 Hz; standard normalization).

Usage::

    python -m brain_moe_pinn.data.hf_celegans inspect          # schema only
    python -m brain_moe_pinn.data.hf_celegans download \\
        --out /path/worm_data_short.parquet                    # full corpus
    python -m brain_moe_pinn.data.hf_celegans ingest \\
        --parquet /path/worm_data_short.parquet \\
        --out /path/celegans_hf_ladder [--limit N]

The ingest step writes the canonical ladder layout consumed by
``SpeciesSignalDataset``/``train.py --data``: per-worm ``calcium/<id>.npy``
(N, T), ``calcium_ids/<id>.txt`` (neuron names), ``calcium_mask`` where
traces were absent/NaN, and a manifest with subject/origin/rate metadata.
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HF_REPO = "qsimeon/celegans_neural_data"
HF_FILE = "worm_data_short.parquet"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"
RESAMPLE_DT_S = 0.333


class RangeReader:
    """Minimal seekable HTTP reader (range requests) for parquet metadata."""

    closed = False

    def __init__(self, url: str, size: int, chunk: int = 8 << 20):
        self.url, self.size, self.chunk = url, size, chunk
        self.pos = 0
        self.cache: Dict[int, bytes] = {}

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 2:
            self.pos = max(0, self.size + offset)
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = max(0, offset)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readable(self) -> bool:
        return True

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(self.size, start + length) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while len(out) < n:
            start = self.pos + len(out)
            base = start - (start % self.chunk)
            data = self.cache.get(base)
            if data is None:
                data = self._fetch(base, self.chunk)
                self.cache[base] = data
            piece = data[start - base: start - base + n - len(out)]
            out += piece
            if not piece:
                break
        self.pos += len(out)
        return bytes(out)


def remote_size(url: str = HF_URL) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers.get("Content-Length", 0))


def inspect(url: str = HF_URL) -> None:
    """Print the parquet schema using only footer range reads (no download)."""
    import pyarrow.parquet as pq
    size = remote_size(url)
    print(f"[hf] {HF_FILE} remote size: {size / 1e6:.1f} MB")
    pf = pq.ParquetFile(RangeReader(url, size))
    print(pf.schema_arrow)
    print("num_row_groups:", pf.num_row_groups)
    md = pf.metadata
    print("rows (approx):", md.num_rows if md is not None else "n/a")


def download(out: Path, url: str = HF_URL) -> Path:
    """Stream-download the corpus parquet with a progress report."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=600) as resp, open(out, "wb") as f:
        downloaded = 0
        while True:
            chunk = resp.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            print(f"\r[hf] {downloaded / 1e6:.1f} / {expected / 1e6:.1f} MB",
                  end="", flush=True)
    print()
    if downloaded != expected:
        raise RuntimeError(
            f"size mismatch: got {downloaded}, expected {expected}")
    print(f"[hf] downloaded {out}")
    return out


def ingest(parquet: Path, out_root: Path, limit: Optional[int] = None,
           keep_unlabeled: bool = False) -> None:
    """Convert the long-format parquet into the canonical ladder layout.

    Rows with ``is_labeled_neuron == False`` carry a *recording-local slot*
    index as their ``neuron`` name, not a cell identity: across the corpus 103
    such tokens recur in more than one worm (one in 268 worms), so keeping them
    would make union-alignment merge unrelated cells. They are dropped by
    default (~23% of rows, 9,919/42,798); pass ``keep_unlabeled=True`` only for
    single-worm analysis.
    """
    import pyarrow.parquet as pq
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(
        parquet,
        columns=["source_dataset", "raw_data_file", "worm", "neuron", "slot",
                 "calcium_data", "time_in_seconds", "max_timesteps",
                 "is_labeled_neuron"])
    rows = table.to_pylist()
    print(f"[hf] {len(rows)} (worm, neuron) rows")

    if not keep_unlabeled:
        labeled = [r for r in rows if r["is_labeled_neuron"]]
        dropped = len(rows) - len(labeled)
        if dropped:
            print(f"[hf] dropped {dropped} unlabeled-slot rows "
                  f"({100.0 * dropped / max(1, len(rows)):.1f}%); their "
                  f"'neuron' field is a per-worm slot index, not an identity")
        rows = labeled

    by_worm: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        key = (row["worm"], row["source_dataset"])
        by_worm.setdefault(key, []).append(row)

    # resolve duplicate worm ids across source datasets
    seen = {}
    for (worm, _origin) in by_worm:
        seen[worm] = seen.get(worm, 0) + 1
    duplicate = {w for w, n in seen.items() if n > 1}

    order = sorted(by_worm)
    if limit is not None:
        order = order[:max(1, int(limit))]

    # Provenance: which raw release file each (dataset, worm) came from.
    source_series: Dict[Tuple[str, str], str] = {}
    for key, entries in by_worm.items():
        series = {r.get("raw_data_file") or "" for r in entries}
        source_series[key] = ";".join(sorted(s for s in series if s))

    def emit(key, entries):
        worm, origin = key
        # Channel order is the recording's own slot order, so the ladder keeps
        # the array layout of the release (ties broken by name for stability).
        neuron_rows = sorted(entries, key=lambda r: (r.get("slot") or 0,
                                                     r["neuron"]))
        names = [r["neuron"] for r in neuron_rows]
        t_max = max((r["max_timesteps"] or 0) for r in neuron_rows)
        if t_max < 1:
            t_max = max(len(r["time_in_seconds"] or [])
                        for r in neuron_rows) or 1
        matrix = np.full((len(neuron_rows), t_max), np.nan, dtype=np.float32)
        for i, r in enumerate(neuron_rows):
            data = np.asarray(r["calcium_data"] or [], dtype=np.float32)
            n = min(len(data), t_max)
            if n:
                matrix[i, :n] = data[:n]
        mask = np.isfinite(matrix)
        matrix = np.nan_to_num(matrix, nan=0.0)
        sample_id = f"{origin}__{worm}" if worm in duplicate else worm
        d = out_root / "calcium"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{sample_id}.npy", matrix)
        ids_dir = out_root / "calcium_ids"
        ids_dir.mkdir(parents=True, exist_ok=True)
        (ids_dir / f"{sample_id}.txt").write_text("\n".join(names) + "\n")
        if not mask.all():
            mdir = out_root / "calcium_mask"
            mdir.mkdir(parents=True, exist_ok=True)
            np.save(mdir / f"{sample_id}.npy", mask)
        return sample_id, worm, origin

    from .readers import _write_rows
    manifest_rows = []
    for key in order:
        sample_id, worm, origin = emit(key, by_worm[key])
        manifest_rows.append({
            "sample_id": sample_id,
            "subject": f"{origin}::{worm}" if worm in duplicate else worm,
            "session": "1",
            "origin": origin,
            "condition": "",
            "rate_hz": str(round(1.0 / RESAMPLE_DT_S, 4)),
            "dt_s": str(RESAMPLE_DT_S),
            "channels": str(len(by_worm[key])),
            "labeled_only": "0" if keep_unlabeled else "1",
            "source_series": source_series.get(key, ""),
        })
    _write_rows(out_root / "manifest.csv", manifest_rows)
    print(f"[hf] ingested {len(order)} worms -> {out_root}")
    print(f"[hf] dt={RESAMPLE_DT_S}s (~{round(1.0/RESAMPLE_DT_S, 2)} Hz), "
          f"standard normalization; masks mark NaN/absent entries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="print remote schema (no download)")
    dl = sub.add_parser("download", help="download the corpus parquet")
    dl.add_argument("--out", default="worm_data_short.parquet")
    ing = sub.add_parser("ingest", help="parquet -> canonical ladder layout")
    ing.add_argument("--parquet", required=True)
    ing.add_argument("--out", required=True)
    ing.add_argument("--limit", type=int, default=None,
                     help="ingest at most N worms (smoke runs)")
    ing.add_argument("--keep-unlabeled", action="store_true",
                     help="keep unlabeled-slot rows (per-worm placeholders "
                          "with no cross-worm identity; off by default)")
    args = parser.parse_args()
    if args.command == "inspect":
        inspect()
    elif args.command == "download":
        download(Path(args.out))
    else:
        ingest(Path(args.parquet), Path(args.out), limit=args.limit,
               keep_unlabeled=args.keep_unlabeled)


if __name__ == "__main__":
    main()
