"""Ingest collected mouse corpora into the canonical data ladder.

Source manifests cover DANDI NWB, extracted Dryad/Figshare arrays, and
OpenNeuro NIfTI/BIDS runs.  No network download is implicit.  For a BIDS run
that is already on disk, the convenience command is::

    python -m brain_moe_pinn.data.ingest_mouse bids-fmri \
        --bids-root /data/openneuro/ds004402 \
        --subject 01 --out /data/data_ladder/mouse_crossrepo

For federated calcium/voltage/fMRI/behavior ingestion, use a source manifest::

    python -m brain_moe_pinn.data.ingest_mouse ingest \
        --source-manifest /data/mouse_sources.json \
        --out /data/data_ladder/mouse_crossrepo
"""

from __future__ import annotations

import argparse
import json

from .corpus_pipeline import (
    ingest_bids_fmri_session,
    validate_ladder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("ingest", help="source manifest -> canonical ladder")
    source.add_argument("--source-manifest", required=True)
    source.add_argument("--out", required=True)
    source.add_argument("--overwrite", action="store_true")
    bids = sub.add_parser("bids-fmri", help="ingest one OpenNeuro BIDS BOLD run")
    bids.add_argument("--bids-root", required=True)
    bids.add_argument("--subject", required=True)
    bids.add_argument("--session")
    bids.add_argument("--task")
    bids.add_argument("--run")
    bids.add_argument("--space")
    bids.add_argument("--atlas")
    bids.add_argument("--tr", type=float, dest="tr_s")
    bids.add_argument("--max-channels", type=int, default=400)
    bids.add_argument("--sample-id")
    bids.add_argument("--origin", default="OpenNeuro")
    bids.add_argument("--out", required=True)
    bids.add_argument("--overwrite", action="store_true")
    validate = sub.add_parser("validate", help="validate canonical ladder files")
    validate.add_argument("--root", required=True)
    validate.add_argument("--modalities", nargs="*")
    args = parser.parse_args()
    if args.command == "ingest":
        from .corpus_pipeline import ingest_source_manifest
        ingest_source_manifest(
            args.source_manifest, args.out, species="mouse",
            overwrite=args.overwrite)
    elif args.command == "bids-fmri":
        ingest_bids_fmri_session(
            args.bids_root, args.out, subject=args.subject,
            session=args.session, task=args.task, run=args.run,
            space=args.space, atlas=args.atlas, tr_s=args.tr_s,
            max_channels=args.max_channels, sample_id=args.sample_id,
            origin=args.origin, overwrite=args.overwrite)
    else:
        print(json.dumps(validate_ladder(
            args.root, modalities=args.modalities or None), indent=2))


if __name__ == "__main__":
    main()
