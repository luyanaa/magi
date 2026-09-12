"""Ingest collected zebrafish corpora into the canonical data ladder.

The source manifest can point to DANDI NWB sessions, extracted Dryad/Figshare
NPZ/MAT/CSV files, or inline smoke-test arrays.  The command never downloads
data; acquire and extract a source first, then run::

    python -m brain_moe_pinn.data.ingest_zebrafish ingest \
        --source-manifest /data/zebrafish_sources.json \
        --out /data/data_ladder/zebrafish_crossrepo

Validate the result before training::

    python -m brain_moe_pinn.data.ingest_zebrafish validate \
        --root /data/data_ladder/zebrafish_crossrepo
"""

from .corpus_pipeline import _cli


if __name__ == "__main__":
    _cli("zebrafish")
