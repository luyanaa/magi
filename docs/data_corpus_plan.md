# Data Corpus Plan and Registry

*Living document — updated 2026-09. Purpose: record the open-data landscape for
the species data ladder (C. elegans → zebrafish → mouse → human), choose
corpora, and track acquisition/ingestion status. Numbers below are as
reported by the linked sources at the time of writing; always verify counts,
formats, and licenses at download.*

---

## 1. Ladder adequacy model (decision, 2026-09)

The ladder does **not** require equal subject counts across stages — open data
cannot provide them and the shared-dynamics + replay + adaptation design does
not need them. Adequacy is instead judged on four axes:

| Axis | Definition | C. elegans | Zebrafish | Mouse |
|---|---|---|---|---|
| Brain coverage | fraction of the nervous system sampled per subject, region-annotated | whole nervous system (~250/302 neurons) | whole brain (10⁴–10⁵ cells → region-parcellated) | cortex-wide + whole-brain fMRI |
| Subject count | usable animals after harmonization | ~900 (homogenized HF) | target ≥100 (federated across studies) | target ≥100 (federated across modalities) |
| Temporal scale | rate × duration per subject | 10 Hz × ~10 min | 1–5 Hz × min–h | 2P ~30 Hz; widefield 10–50 Hz; fMRI ~0.5 Hz |
| Conditions | behavior/stimulus labels | stimuli (salt, etc.) | tail tracking, stimuli | task (IBL), visual stimuli (Allen) |

**Ladder currency**: region/cell-trace matrices `(C, T)` per subject, with a
consistent parcellation per species (CCF for mouse, mapZebrain/Z-Brain for
zebrafish, neuron-name sets for C. elegans). Raw volumes/pixels are ingested
only through that reduction; the loader contract is
`{modality: (B, C, T)}` + manifest metadata.

---

## 2. C. elegans

### 2.1 Primary corpus (target): homogenized neural activity + connectomes

| Field | Value |
|---|---|
| Source | [arXiv:2411.12091](https://arxiv.org/abs/2411.12091) — Simeon, Kashyap, Kording, Boyden 2024 |
| Activity data | [HF: qsimeon/celegans_neural_data](https://huggingface.co/datasets/qsimeon/celegans_neural_data) |
| Code | [github.com/qsimeon/worm-graph](https://github.com/qsimeon/worm-graph) |
| Reported scale | ~900 worms from 12 neuroimaging experiments; ~250 uniquely labeled neurons; resampled to common rate; consistent neuron ordering |
| Connectivity | 9 connectome annotations (3 EM studies + propagation study) as graphs |
| Why it fits | Solves the per-worm channel-subset problem (see §2.2); common ordering → dense `(250, T)` batches; graph data → future structure-conditioned dynamics |
| Caveats | Calcium low-pass filtering/temporal lags; heterogeneous protocols — stimulus metadata is essential; check HF license before use |
| Status | **Scripts ready (2026-09), corpus not yet downloaded.** `python -m brain_moe_pinn.data.hf_celegans inspect` (footer-only schema check: verified 42,798 (worm, neuron) rows, single row group, long-format `calcium_data`/`time_in_seconds` with resample dt = 0.333 s) → `download --out worm_data_short.parquet` (669.5 MB) → `ingest --parquet ... --out ...` (canonical ladder: per-worm (N, T), neuron ids, masks for NaN gaps, manifest with origin/rate) |

### 2.2 Local pilot (acquired): salt-stimulus recordings

| Field | Value |
|---|---|
| Location | `cleandata_smoothened2/` (Google Drive) — 24 worms, `<id>_ratio.csv` (T×N time-major) + `<id>_uniqNames.csv` |
| Scale | 24 worms; observed `T=6000`, `N=146…~226` (per-worm subsets differ) |
| Stimulus | salt (per `sample_name_list.txt`) |
| Ingestion (2026-09 upgrade) | `data/ingest_c_elegans.py` → canonical ladder via `emit_sample`: `calcium/<id>.npy (N,T)`, `calcium_ids/<id>.txt`, `calcium_mask` (NaN cells), manifest (`subject/origin/condition/rate_hz`). Options: `--metadata gKDR-GMM/metadata` + `--filter-known` (keep only channels in the canonical name pool) or `--apply-names` (positional id mapping) |
| ID semantics (verified) | Mixed within worms: canonical names (e.g. `ADAL`) plus recording-local numeric ids; `--filter-known` drops the latter (e.g. worm 10: −78, worm 11: −54, worm 12: −82 channels) making cross-worm **name-union alignment** valid; 299-name pool from `conneurons.csv` (159 in `globalNames.csv`), `multiconmatrix.csv` rows are 299-valued profiles |
| Sanity | `tools/real_data_sanity.py` passed (batch-1; forward/backward, correlation criteria); loader P0-P2 features (union alignment, masks, manifest splits) covered by tests |

---

## 3. Zebrafish (target: ≥100 larvae, federated)

### 3.1 Candidate corpora

| Dataset | Source | Reported scale | Format / access | Notes |
|---|---|---|---|---|
| Portugues-lab whole-brain studies (e.g. [Lavian et al. 2025](https://github.com/portugueslab/Lavian_et_al_2025); OMR studies e.g. Kist & Portugues 2019) | [portugueslab.com](https://portugueslab.com/publications.html) | cohorts ~10–30 larvae/condition; whole-brain cellular, thousands of neurons, HDF5 | per-paper GitHub/Zenodo ([e.g. Zenodo 10281421](https://zenodo.org/records/10281421)); some contact-gated | Cell traces + coordinates + region labels + behavior (tail tracking); best structural fit |
| DANDI dandisets (Haesemeyer thermoregulation [000697–699, 707–708](https://dandiarchive.org/); forebrain/midbrain [000235/000236](https://dandiarchive.org/); Ahrens glia/behavior [000350](https://dandiarchive.org/); mesoscale pipelines [000244](https://dandiarchive.org/)) | [dandiarchive.org](https://dandiarchive.org/) | varies; whole-brain or regional, cellular | NWB (DANDI standard) | Needs NWB → `(C,T)` importer; per-dandiset check |
| Z-Brain atlas / [ZebraFishExplorer](https://zebrafishexplorer.zib.de/about) + [mapZebrain](https://mapzebrain.org/) | ZIB / Portugues lab | atlas: 294 regions, 4,000+ traced neurons | `.mat`/`.h5`/SWC + registration tools (ANTs) | Parcellation/registration substrate, not a bulk trace dump |
| Free-swimming whole-brain (Kim/Kim 2017 Nature Methods) | via journals/DANDI | n≈4–7 larvae per study (technical) | — | Reference for naturalistic behavior; low n |
| **ZAPBench** (Zebrafish Activity Prediction Benchmark) | [github.com/google-research/zapbench](https://github.com/google-research/zapbench) / [arXiv:2503.02618](https://arxiv.org/pdf/2503.02618) (ICLR 2025) | 4D light-sheet recordings of **>70,000 neurons** in larval zebrafish; motion-stabilized, voxel + cell segmentations | Apache 2.0; segmentation/trace data + baselines | Whole-brain cellular forecasting benchmark — best-in-class target for dynamics evaluation; cell-level C is beyond raw-channel ladder → use segmentation subset/region reduction |
| Whole-brain under oriented grating stimuli | [Zenodo 19486022](https://zenodo.org/records/19486022) | larval zebrafish whole-brain + stimuli | Zenodo | Visual-stimulus analysis track |
| Whole-brain connectomic resource (intact 7 dpf larva) | [bioRxiv 2025.06.10.658982](https://www.biorxiv.org/content/10.1101/2025.06.10.658982v1.full-text) | >40,000 neurons annotated; 30M synapses, molecular labels | vEM + confocal | Future structure-conditioned dynamics (zebrafish connectome) |
| Community dataset co-registration | [Sprague et al. 2025](https://www.sciencedirect.com/science/article/pii/S2667237524003540) | unifies community whole-brain imaging datasets | repository + trained cell-ID models | Standardization path across labs |
| Larval behavior videos | [Zenodo 7807968](https://zenodo.org/records/7807968) | 376.8 GB behavior video | Zenodo | Behavior-modality pool (Mullen et al. 2023) |
| Whole-brain **voltage** imaging | [Nature Methods 2026 (s41592-026-03179-7)](https://www.nature.com/articles/s41592-026-03179-7) / [bioRxiv 2023.12.15.571964](https://www.biorxiv.org/content/10.1101/2023.12.15.571964v1.full-text) | whole-brain voltage, ~35 s trials (200 s continuous shown) | per-paper | The ladder's "voltage" modality for zebrafish — emerging, few specimens |
| Gene-expression + activity co-mapping (WARP) | [bioRxiv 2026.02.07.704095](https://www.biorxiv.org/content/10.64898/2026.02.07.704095v1.full-text) | whole-brain, behaving larva, neuron-type identification | per-paper | Future heterogeneity/metadata axis |

### 3.2 Plan
1. Pick 1–2 Portugues-lab head-fixed datasets (optomotor/decision tasks) with open deposits; register cells to Z-Brain regions → region traces `(C≈294, T)`.
2. Add Haesemeyer thermoregulation dandisets (many fish per condition) via an NWB importer as a second corpus (temperature gradient = natural perturbation track).
3. Target federated total ≥100 larvae; manifest must carry stimulus/condition labels.

---

## 4. Mouse (target: ≥100 subjects, federated by modality)

### 4.1 Candidate corpora

| Modality | Dataset | Source | Reported scale | Format / access | Notes |
|---|---|---|---|---|---|
| 2P calcium (visual cortex) | Allen Brain Observatory — Visual Coding | [portal](https://portal.brain-map.org/explore/circuits) / AWS `s3://allen-brain-observatory` | ~243 mice, ~60–63k neurons, hundreds of sessions (3×1 h each) | AllenSDK | Session-specific neuron sets → use CCF region-parcellated ROI traces for cross-session consistency |
| Ephys + behavior | IBL Brain-Wide Map | [internationalbrainlab.com](https://www.internationalbrainlab.com/brainwide-map) | 139 mice, 459 sessions, ~621k units | ONE API / ALF | Behavior + multi-area spikes; binned region-rate “voltage” proxy |
| Ephys + visual behavior | Allen Neuropixels Visual Behavior | [DANDI 000713](https://dandiarchive.org/dandiset/000713) | ~300,000 mouse neurons, task | NWB/DANDI | Largest NP release; complements IBL |
| Ephys (derived) | Curated SWR corpus from public NP datasets | [Sci Data 2025 (s41597-025-06115-0)](https://www.nature.com/articles/s41597-025-06115-0) | multi-dataset curation | DANDI/ONE | Aggregation pattern for derived corpora |
| 2P calcium (dense, rule task) | Tseng et al. 2022 / Harvey lab | [Harvey lab DANDI](https://harveylab.hms.harvard.edu/resources/dandi.html) | **8 mice, 286 sessions, 273,770 neurons**, 6 regions | NWB | Dense-session champion; foundation-model pretraining target (Tseng 2022) |
| Widefield + ephys (decision) | Nature 2025 prior-information study | [s41586-025-09226-1](https://www.nature.com/articles/s41586-025-09226-1) | 2,289 region-sessions; widefield 32 dorsal regions + NP | per-paper | In-brain multimodal sync at scale |
| fMRI (awake, multi) | OpenNeuro [ds007100](https://openneuro.org/datasets/ds007100/versions/1.0.3); 14T awake (eLife preprint, 38 mice) | OpenNeuro / eLife | cohort sizes to verify at download | BIDS | Awake-state fMRI pool beyond n=10 |
| Widefield calcium | IBL widefield | [IBL widefield docs](https://docs.internationalbrainlab.org/notebooks_external/loading_widefield_data.html) | sessions subset (search `widefieldU.images.npy`) | ONE API, ALF, SVD-decomposed (`widefieldSVT.haemoCorrected.npy`) | ROI/SVD mode, not raw pixels |
| Widefield calcium + task | [DANDI 001712](https://dandiarchive.org/dandiset/001712) (“IBL Widefield”, Cntnap2 KO decision task) | DANDI | NWB | Standardized mesoscale + behavior; check subject count at download |
| **Widefield calcium + behavior (operant lever-pull)** | [Kondo et al., Sci Data 2025](https://www.nature.com/articles/s41597-025-05482-y) | **25 mice, 364/375 sessions**; dorsal-cortex wide-field + behavior sensors + 3×100 Hz cameras (body/face/eye) | Sci Data repository | Strong primary mouse widefield candidate: many sessions/mouse, multimodal behavior |
| Dual-color mesoscopic ACh + calcium (awake mice) | [DANDI 001172](https://dandiarchive.org/dandiset/001172) (HnaskoLab/Lotfi 2025) | mesoscale, awake | NWB | Neuromodulator+calcium; future for state/arousal conditioning |
| 2P ↔ ephys calibration | [Allen ophys/ephys calibration](https://portal.brain-map.org/our-research/circuits-behavior/ophys-ephys-calibration-data) | simultaneous GCaMP6s/f + spiking, L2/3 | Allen | Calcium→spike calibration mapping (voltage proxy validation) |
| fMRI (awake, longitudinal) | [Longitudinal rs-fMRI habituation (PMC12956629)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12956629) | awake mice, longitudinal sessions | per-paper | Habituation dynamics; check cohort n |
| fMRI (awake) | Gutierrez-Barragan et al. (Curr Biol 2022) | [Mendeley 10.17632/np2fx99hn6.2](https://doi.org/10.17632/np2fx99hn6.2) (CC BY 4.0) | n=10 awake (n=19 anesthetized: [354f8dc8xh.2](https://doi.org/10.17632/354f8dc8xh.2)) | NIfTI etc. | Drop first ~120 scans (thermal equilibration); parcellate to CCF regions |
| fMRI (large cohorts) | IIT Gozzi lab — e.g. Autism Mouse Brain Connectome collection (600+ mice, 20 etiologies) | [fnimg.iit.it/datasets-code](https://fnimg.iit.it/datasets-code) | >600 mice (multi-center); aging n≈82 | per-collection | Multi-etiologies add confounds; use healthy controls subset for ladder base |

### 4.2 Plan
1. Primary: Allen 2P Visual Coding → CCF region-parcellated ROI traces (per-session `(C_CCF, T)`) as the calcium channel with strong subject count (~200+).
2. Behavior/ephys: IBL (139 mice) as the “voltage/behavior” channel; binned region rates.
3. Widefield: IBL widefield or DANDI 001712 (Cntnap2 → wild-type subset) — region/SVD traces.
4. fMRI: Gutierrez-Barragan awake (n=10) as the cross-modal bridge; treat larger IIT collections (healthy-control subset) as optional scale-up.
5. Note: no open corpus pairs fMRI + widefield + behavior in the same animals at scale → mouse stage entries are **async cross-subject modalities**; use soft-contrastive/async alignment (labels required).

---

## 5. Scale parity with the human catalog (EEG-Datasets.md §D.1)

Human catalog classes (from `EEG-Datasets.md`): scalp EEG ~90,500+ subjects / ~668M tokens; ECoG/sEEG ~1,128 / ~2.7B; MEG ~1,000; fMRI ~122,000+ subjects; sync EEG-fMRI ~363; dense-sampling subjects (6-100+ h); task cohorts n≈20-2,500.

Animal open data cannot match the top subject-count classes (no TUH/UKB equivalents exist for mice/zebrafish), but it **matches or exceeds the other classes** via sessions and units per subject:

| Human class (scale) | Animal counterpart found (rounds 1-3) | Parity |
|---|---|---|
| Very large-n EEG/fMRI (10⁴-10⁵ subjects) | None. C. elegans HF ~900 worms is the closest whole-brain "large-n"; mouse EEG is n≈9-20 per study | **Gap: 1-2 orders**; big-n role goes to C. elegans + federated mouse fMRI (AMBC >600 mice, disease; healthy subset smaller) |
| Large-n task/ephys cohorts (MOABB ~2,500; MEG ~1,000; ECoG ~1,128) | Allen Neuropixels Visual Behavior ~300k neurons ([DANDI 000713](https://dandiarchive.org/dandiset/000713)); IBL 139 mice/459 sessions; Allen 2P ~243 mice; aggregated 2024-25 releases | **Comparable in units/sessions**, not subjects (~400-600 mice aggregated) |
| Dense per-subject (CNeuroMod 6×100+h; MyConnectome; NSD 8×30-40 sessions) | Tseng/Harvey 2P: **8 mice × 286 sessions, 273,770 neurons** ([Harvey lab DANDI](https://harveylab.hms.harvard.edu/resources/dandi.html)); Kondo widefield 25 mice × 364 sessions; Allen 2P 3×1 h/mouse | **Animals exceed humans** (sessions/units per subject) |
| Whole-brain single-specimen extreme | ZAPBench: **~70,000 neurons, one larva, same-specimen connectome in progress** ([blog](https://research.google/blog/improving-brain-models-with-zapbench)); C. elegans HF 250-neuron × 900 | Structural (function + connectome same brain) — new class |
| fMRI large-n (UKB 43k; HCP ~1.2k) | Mouse fMRI: awake n=10-38; ds007100 awake mice (check cohort); aging n≈82; AMBC >600 (disease) | **Gap ~2 orders**; federate + accept |
| Sync multimodal (EEG-fMRI ~363; iEEG+fMRI ~10) | Mouse: widefield+ephys Nature 2025 **2,289 region-sessions**; Allen ophys/ephys calibration; DANDI 001172 ACh+calcium; Kondo widefield+behavior+3 cameras; mouse iEEG+widefield DANDI 001211/001543 | **Animals exceed humans** (sync is cheaper at animal scale) |
| "Voltage" fast modality (EEG/MEG ms-scale) | **Whole-brain voltage imaging in larval zebrafish** ([Nature Methods 2026](https://www.nature.com/articles/s41592-026-03179-7); [bioRxiv 2023.12.15.571964](https://www.biorxiv.org/content/10.1101/2023.12.15.571964v1.full-text)) | Emerging; few specimens, ms-scale whole brain |
| Neuron-type/heterogeneity | WARP gene-expression + activity co-mapping, behaving larva ([bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.02.07.704095v1.full-text)) | Future conditioning/metadata axis |

**Implication**: adequacy axes (§1) should count **units and sessions per subject** alongside subjects; ladder stages optimize different axes per class (C. elegans = large-n; zebrafish = single-specimen whole-brain + structure; mouse = dense-session multimodal; human = large-n low-density). Revised targets absorb this: mouse "≥100 subjects" is reachable federated (Allen 243 + IBL 139 + Kondo 25 + fMRI cohorts); zebrafish "≥100 larvae" remains the binding constraint (federation needed), with ZAPBench providing unmatched depth per larva.

## 6. Integration requirements (loader/ingestion upgrades)

Recorded as code tasks independent of any single corpus:

1. **Manifest-driven `SpeciesSignalDataset`** — subject keys, per-sample files, rate/`dt`, stimulus, neuron/region file; index-free alignment.
2. **Leave-subject-out grouping** in `build_species_dataloaders` (currently `random_split`).
3. **Per-channel masks** for absent cells/regions (loss criteria already support masks).
4. **NWB/DANDI importer** (zebrafish dandisets, mouse widefield DANDI) and **HuggingFace datasets importer** (C. elegans homogenized) → ladder layout.
5. **Region parcellation stage** (CCF/mapZebrain) → `(region, T)` traces; raw cellular/volumetric data only through this stage (or full-cell only where C ≈ ≤1e3, e.g. C. elegans).
6. **TDE-RICA motif comparability** — the ladder's cross-species analysis baseline is the free-run metrics suite + TDE-RICA-style components; keep per-stage spectral/marginal targets species-specific.

### 6.1 Requirements derived from the corpus analysis (2026-09)

| Corpus property (examples from §2-5) | What the loader/abstraction must support |
|---|---|
| Multi-session dense designs (Tseng 8 mice × 286 sessions; Kondo 25 × 364; Allen 3×1 h; worm 1 session) | **Nested subject → session hierarchy**; windowed sampling per session; deterministic **subject-grouped** train/val/test splits (LSO), not `random_split` |
| Per-sample channel sets (worm subsets 146–226; Allen session-specific neurons; ZAPBench cell sets) | Per-sample **channel identity** (`channel_ids`), union alignment, **per-channel masks**; batch only when channel space is compatible (else batch-1 or aligned batch) |
| Cell-resolved vs region corpora (ZAPBench 70k cells; CCF/mapZebrain regions) | Lazy **parcellation transform** at load time (cell→region via labels), so one corpus serves both regimes without duplicate files |
| Heterogeneous raw formats (NWB/DANDI, HDF5/Zenodo, ALF/ONE, NIfTI/BIDS, HF/parquet, CSV) | Thin **source readers** that emit one canonical layout (index + arrays); the loader never sees source formats |
| Rate/dt varies per session/species (10 Hz worm vs 0.5 Hz fMRI vs ms voltage) | Per-sample `rate_hz`/`dt` metadata; **windows in seconds**, not frames; dt feeds `latent_dt` |
| Stimulus/condition/task labels (salt, gratings, temperatures, lever-pull, trials) | Condition keys per sample; **leave-condition-out** capability; trial-array slicing (35 s trials, 250-trial sessions) |
| Behavior/aux channels (tail tracking, 100 Hz video features, task events) | Auxiliary time series in the batch dict; categorical or low-dim continuous |
| Same-specimen structure-function (C. elegans HF graphs; ZAPBench connectome) | Optional per-sample **static graph** (adjacency over channel ids) alongside signals |
| Derived/curated multi-origin corpora (SWR NP corpus; community co-registration) | **Provenance** column; splits/origin-aware reporting |
| NaN/photobleach/invalid frames | Mask convention end-to-end: storage (NaN or sibling mask), dataset output, **TotalLoss recon masking**, metric reporting |

### 6.2 Target abstraction (sketch)

```python
# canonical index (CSV) — one row per sample/window source:
# sample_id, subject, session, origin, modality_file[,...],
# channel_id_file, mask_file, rate_hz, dt_s, duration_s,
# condition, split_hint

# dataset returns (and collate batches):
{
  "calcium": (B, C, T),            # or channel-aligned union
  "calcium_channel_ids": ...,      # identity for alignment/parcellation
  "calcium_mask": (B, C, T) bool,  # absent cells / invalid frames
  "behavior": (B, T, D) | labels,
  "graph": (B, C, C) | None,       # structure conditioning
  "sample_id": ..., "subject": ..., "condition": ...,
}
# reserved keys ride through trainer/forward_modalities untouched;
# recon terms consume "<modality>_mask" from targets.
```

**Modality roles (implemented, 2026-09):** every modality carries a
role — `signal` (encoder + recon), `control` (stimulus: salt steps, drifting
gratings, temperature gradients, task events), `aux` (behavior/labels),
`graph` (structure). Loaders return all roles; trainers route `signal` to
`forward_modalities`, `control` to the model's `perturbation` input
(per-step reduction for rollouts), and keep `aux`/`condition` for losses and
evaluation. This is the "one data" contract for stimulus-bearing corpora.

**Status (2026-09): P0-P2 implemented in `data/species_dataset.py`,
`data/readers.py`, trainer and `--data` profiles; covered by
`tests/test_manifest_dataset.py` (7 tests).** Manifest rows (subject/session/
origin/condition/rate/dt), group-preserving splits (subject/condition/origin/
none, optional test split), per-channel masks threaded into `TotalLoss`
recon terms, seconds windows with time-padding collate, union alignment with
max-channel guard, load-time region aggregation, trial windows, static
graphs, source readers (`emit_sample`, HDF5/NWB adapters, ladder summary),
and data-profile options (seq_seconds, split_by, region_map, align_channels,
...). Corpus scripts are run-ready: `data/hf_celegans.py`
(inspect/download/ingest for the homogenized HF corpus, footer-only schema
check verified) and the upgraded `data/ingest_c_elegans.py` (gKDR metadata
name filtering, masks, canonical manifest). Remaining work is executing the
downloads/ingestions and building the ZAPBench/Allen/Zebrafish adapters on
top of `data/readers.py`. Stimulus control is wired end-to-end (roles → perturbation reduction → conditioned gates; `noise_mode` policy `off|rollout|train|always`); conditioned diffusion (P2.2) remains deferred.

---

## 7. Search log

### Round 2 (Tavily CLI `tvly search`, 2026-09-10; unauthenticated rate limit)
Keywords used (basic depth, 4 results each): `zebrafish whole-brain calcium imaging dataset repository download traces`, `zebrafish larva brain recording open dataset zenodo hdf5 behaviour`, `zebrafish functional imaging timeseries dataset forecasting benchmark`, `zebrafish brain dynamics DANDI NWB larva imaging dataset`, `mouse brain dynamics open dataset widefield 2-photon ephys behavior combined`, `mouse widefield calcium imaging open data cortex sessions number mice`, `mouse electrophysiology large open dataset Allen IBL mice sessions`, `mouse fMRI BOLD resting-state open dataset awake subjects`, `mouse calcium imaging open dataset DANDI 2024 2025 sessions`, `human intracranial EEG ECoG open dataset DANDI sessions`, `simultaneous EEG fMRI open dataset paired recordings`, `comparative whole-brain imaging open datasets neuroscience`.

Key additions from this round are integrated above: **ZAPBench** (>70k neurons, Apache-2.0 forecasting benchmark), **Zenodo oriented-grating whole-brain dataset**, **larval connectome resource**, **community co-registration (Sprague 2025)**, **Kondo widefield+behavior (25 mice/364 sessions)**, **DANDI 001172 dual-color ACh+calcium**, **Allen ophys/ephys calibration**, **longitudinal awake mouse fMRI**. Human-stage finds (DANDI:000623, RAM ECoG, sleep ECoG n=39, EEG-fMRI corpora) are catalogued in `EEG-Datasets.md` per project convention.

### Round 3 (Tavily CLI, 2026-09-10; scale-focused)
Keywords: `mouse EEG dataset large number of mice open animal electroencephalogram cohort`, `largest mouse Neuropixels electrophysiology dataset number of mice open 2024 2025`, `largest zebrafish whole-brain imaging dataset number of larvae recording hours`, `mouse fMRI dataset large cohort number of mice resting state 2025 open`, `DANDI mouse calcium imaging total sessions dataset aggregate number`, `mouse widefield imaging number of mice sessions open dataset 2025`, `zebrafish larvae whole brain calcium dataset 100 fish aggregated`, `animal brain recording consortium aggregated open datasets sessions mice hours`.

Key additions (integrated in §5 and the corpus tables): Allen Neuropixels Visual Behavior (~300k neurons; [DANDI 000713](https://dandiarchive.org/dandiset/000713)); Tseng/Harvey 2P 8 mice × 286 sessions / 273,770 neurons; Kondo widefield n=25/364 sessions; awake-mouse fMRI OpenNeuro [ds007100](https://openneuro.org/datasets/ds007100/versions/1.0.3) (cohort to verify) + 14T awake study (38 mice); Nature 2025 widefield+ephys decision study (2,289 region-sessions); whole-brain voltage imaging in larval zebrafish (Nature Methods 2026); WARP gene-expression co-mapping (bioRxiv 2026); curated SWR Neuropixels corpus (Sci Data 2025); ZAPBench same-specimen connectome note. Mouse scalp-EEG remains tiny (n≈9-20/study) — no large-n EEG-class exists for rodents.

Verify counts/formats/licenses at download; unauthenticated Tavily results may miss gated deposits (e.g., contact-only Portugues-lab raw data).

## 8. Status summary

| Stage | Primary target | Acquisition status | Bottleneck |
|---|---|---|---|
| C. elegans | HF homogenized (~900 worms) | Not acquired (local salt pilot: 24 worms, ingested) | HF download + license; loader alignment |
| Zebrafish | ZAPBench subset/region-reduced + Portugues/Zenodo grating + Haesemeyer DANDI (≥100 larvae) | None acquired | Trace extraction from >70k-cell benchmark; NWB importer; region registration |
| Mouse | Kondo widefield+behavior and/or Allen 2P ROI + IBL + awake fMRI (≥100 subjects federated) | None acquired | CCF parcellation scripts; async cross-modal labels; fMRI n small |
| Human | EEG/ECoG/fMRI (see EEG-Datasets.md / data_loader) | Not acquired | DANDI downloads; paired next-step targets |

**Human stage**: covered by `EEG-Datasets.md` (repo root) — EEG/ECoG/iEEG/fMRI catalogs incl. DANDI iEEG+fMRI recordings (e.g. DANDI:000623 movie-watching iEEG+fMRI, n≈51), RAM-project ECoG (251 subjects/700 sessions), multicenter sleep ECoG (n=39), and simultaneous EEG-fMRI corpora (NKI naturalistic viewing n=22, NEMAR sleep n=33). Not duplicated here.

**Near-term milestones**
1. Loader manifest + LSO upgrade (unblocks everything).
2. C. elegans HF ingestion (largest single win: 900 worms, consistent 250-neuron order).
3. Zebrafish pilot: ZAPBench subset/region-reduced traces (best-in-class whole-brain forecasting target) + one Portugues-lab/Zenodo grating-stimuli deposit → region traces; Haesemeyer dandisets as second corpus.
4. Mouse pilot: Kondo widefield+behavior (25 mice, 364 sessions) or Allen ROI traces; IBL behavior/ephys; awake-fMRI parcellation.
