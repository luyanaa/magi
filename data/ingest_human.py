"""Ingest local human EEG, MEG, and fMRI datasets into the canonical ladder.

The command never downloads a dataset.  Acquire a BIDS/OpenNeuro, Cam-CAN,
HCP, or other catalogued source first, then ingest the local files::

    python -m brain_moe_pinn.data.ingest_human bids-session \
        --bids-root /data/openneuro/ds006040 \
        --subject 01 --task rest --modalities eeg fmri \
        --eeg-rate 256 --atlas /data/atlas.nii.gz \
        --origin OpenNeuro:ds006040 \
        --out /data/data_ladder/human_bids

For heterogeneous sources, use a JSON source manifest.  Its modality entries
can use ``format: mne`` for EDF/BDF/FIF/BrainVision/EEGLAB/CTF recordings and
``format: nifti`` for BOLD; ``SpeciesSignalDataset`` then windows them in
physical seconds at training time.
"""

from __future__ import annotations

import argparse
import json

from .corpus_pipeline import (
    ingest_bids_session,
    ingest_source_manifest,
    validate_ladder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    source = sub.add_parser("ingest", help="source manifest -> canonical ladder")
    source.add_argument("--source-manifest", required=True)
    source.add_argument("--out", required=True)
    source.add_argument("--overwrite", action="store_true")

    bids = sub.add_parser(
        "bids-session", help="ingest local BIDS EEG/MEG/fMRI files")
    bids.add_argument("--bids-root", required=True)
    bids.add_argument("--subject", required=True)
    bids.add_argument("--modalities", nargs="+", required=True,
                      choices=("eeg", "meg", "fmri"))
    bids.add_argument("--session")
    bids.add_argument("--task")
    bids.add_argument("--run")
    bids.add_argument("--eeg-path")
    bids.add_argument("--meg-path")
    bids.add_argument("--fmri-path")
    bids.add_argument("--eeg-rate", type=float,
                      help="optional EEG resampling target in Hz")
    bids.add_argument("--meg-rate", type=float,
                      help="optional MEG resampling target in Hz")
    bids.add_argument("--eeg-channel-types", nargs="+")
    bids.add_argument("--meg-channel-types", nargs="+")
    bids.add_argument("--eeg-picks", nargs="+")
    bids.add_argument("--meg-picks", nargs="+")
    bids.add_argument("--atlas", help="atlas NIfTI for ROI extraction")
    bids.add_argument("--tr", type=float, dest="tr_s")
    bids.add_argument("--space", help="BIDS fMRI space entity")
    bids.add_argument("--max-fmri-channels", type=int, default=400)
    bids.add_argument("--condition")
    bids.add_argument("--cross-modal-label", type=float, choices=(0.0, 1.0))
    bids.add_argument("--origin", default="OpenNeuro")
    bids.add_argument("--sample-id")
    bids.add_argument("--out", required=True)
    bids.add_argument("--overwrite", action="store_true")

    validate = sub.add_parser("validate", help="validate canonical ladder files")
    validate.add_argument("--root", required=True)
    validate.add_argument("--modalities", nargs="*")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest_source_manifest(
            args.source_manifest, args.out, species="human",
            overwrite=args.overwrite)
    elif args.command == "bids-session":
        paths = {
            modality: path
            for modality, path in {
                "eeg": args.eeg_path,
                "meg": args.meg_path,
                "fmri": args.fmri_path,
            }.items()
            if path
        }
        ingest_bids_session(
            args.bids_root,
            args.out,
            subject=args.subject,
            modalities=args.modalities,
            session=args.session,
            task=args.task,
            run=args.run,
            source_paths=paths,
            atlas=args.atlas,
            tr_s=args.tr_s,
            space=args.space,
            max_fmri_channels=args.max_fmri_channels,
            eeg_target_rate_hz=args.eeg_rate,
            meg_target_rate_hz=args.meg_rate,
            eeg_channel_types=args.eeg_channel_types,
            meg_channel_types=args.meg_channel_types,
            eeg_picks=args.eeg_picks,
            meg_picks=args.meg_picks,
            condition=args.condition,
            sample_id=args.sample_id,
            origin=args.origin,
            cross_modal_label=args.cross_modal_label,
            overwrite=args.overwrite,
        )
    else:
        print(json.dumps(validate_ladder(
            args.root, modalities=args.modalities or None), indent=2))


if __name__ == "__main__":
    main()
