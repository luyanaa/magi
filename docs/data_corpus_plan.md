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
| Status | **Downloaded and ingested (2026-09).** `../worm_data_short.parquet` is 669.5 MB; `data/hf_celegans.py ingest` emitted 919 worms (42,798 `(worm, neuron)` rows, 9,919 unlabeled-slot rows dropped) into `../data_ladder/c_elegans_hf` with per-worm calcium, masks, neuron ids, manifest rate `~3.003 Hz` (`dt=0.333 s`). |

### 2.2 Local pilot (acquired): salt-stimulus recordings

| Field | Value |
|---|---|
| Location | `cleandata_smoothened2/` (Google Drive) — 24 worms, `<id>_ratio.csv` (T×N time-major) + `<id>_uniqNames.csv` |
| Scale | 24 worms; observed `T=6000`, `N=146…~226` (per-worm subsets differ) |
| Stimulus | salt (per `sample_name_list.txt`) |
| Ingestion (2026-09 fix) | `data/ingest_c_elegans.py --timing-xlsx cleandata_smoothened2/stimulation_timing.xlsx --metadata gKDR-GMM/metadata --filter-known` → canonical ladder: `calcium/<id>.npy (N,T)`, `calcium_ids/<id>.txt`, `calcium_mask` (NaN cells), `stimulus/<id>.npy` + `stimulus_trials/<id>.json` (salt drive from `generatesalt.m`, 30 s half-periods), manifest with **per-sample** `rate_hz`/`dt_s` plus `animal`/`sample_name`/`anesthesia`/stimulus provenance. Channel selection = canonical names (`--filter-known`) AND the reference autocorrelation filter `--qc-autocorr 0.3 --qc-lag 20`; `--overwrite` re-ingests in place; `--apply-names` maps positional ids *before* the name filter |
| ID semantics (verified) | Mixed within worms: canonical names (e.g. `ADAL`) plus recording-local numeric ids; `--filter-known` drops the latter making cross-worm **name-union alignment** valid; 299-name pool from `conneurons.csv` (159 in `globalNames.csv`; `globalNames ⊆ conneurons`, so the repo pool equals the reference pool). End-to-end on the pilot: 4,621 channels → 3,021 named → **1,625** after the reference autocorr QC (worst worm 95 → 10) |
| Rates (verified) | `frames/sec` differs **per sample** (3.69-5.72, median 4.08) and the metadata's `total duration` confirms it (`6000/4.1215 = 1455.8 s` for sample 1); a single 10 Hz assumption is wrong by 1.75-2.71x. The manifest carries the per-sample value and the model integrates with each batch's own `dt` |
| Animal identity | 3 animals were imaged twice (15, 17, 23; different anaesthesia state) and 9/24 samples are anaesthetised: `subject` is the **animal** id (not the CSV index) so subject-grouped splits cannot place the same animal in train and val, and `anesthesia` is a manifest column |
| Sanity | `tools/real_data_sanity.py --config configs/species/c_elegans.json --root <ladder>` runs on the real ladder: 60 s windows (244 frames at 4.08 Hz), per-sample `dt`, forward/backward with finite gradients, recon scores reported next to persistence/channel-mean baselines. Estimators: `diagnostics/species_parameters.py` (per-neuron tau median 1.50 s, IQR [1.05, 2.79]; AR(2) gain 2.6% ⇒ first-order observation model adequate; coupling split symmetric 0.60 / directed 0.40; zero-lag network R² 0.51) |

### 2.3 Unified C. elegans training root (acquired; local)

The three source adapters remain separate, then join through
`data/merge_c_elegans.py` into
`../data_ladder/c_elegans_unified/manifest.csv`:

| Source | Rows | Clock | Control |
|---|---:|---:|---|
| Toyoshima/gKDR-GMM salt | 24 worms | 3.69–5.72 Hz per recording | scalar salt waveform |
| Randi PumpProbe | 113 recordings | 2 Hz (`dt=0.5 s`) | recording-local, target-gated opto |
| HuggingFace activity | 919 worms | `dt=0.333 s` (`~3.003 Hz`) | absent; no intervention |

The unified manifest has **1,056 calcium rows** and 137 control-bearing rows.
The model-facing `stimulus` contract is width 282: feature 0 is the source
drive, while features 1–281 are Randi recording-local target gates. Toyoshima
controls are stored as scalar source references and padded by the training
partition; HuggingFace rows omit the control field. Source-qualified subjects,
recording-local channel identities, `align_channels=false`, and batch size 1
are intentional—no cross-recording neuron identity is asserted. The profile
uses 15-second physical windows and `resample` over three 5-second rollout
segments, preserving the supplied Randi pulse widths.

The default unified root is manifest-federated rather than a second large
copy. Stage all three canonical roots with the manifest on a remote host, or
run the merge command with `--materialize` when a self-contained bundle is
required. Remote training has not been launched; only local loader and
forward/backward smoke checks are complete.

---

## 3. Zebrafish (target: ≥100 larvae, federated)

### 3.1 Candidate corpora

| Dataset | Source | Reported scale | Format / access | Notes |
|---|---|---|---|---|
| Portugues-lab whole-brain studies (e.g. [Lavian et al. 2025](https://github.com/portugueslab/Lavian_et_al_2025); OMR studies e.g. Kist & Portugues 2019) | [portugueslab.com](https://portugueslab.com/publications.html) | cohorts ~10–30 larvae/condition; whole-brain cellular, thousands of neurons, HDF5 | per-paper GitHub/Zenodo ([e.g. Zenodo 10281421](https://zenodo.org/records/10281421)); some contact-gated | Cell traces + coordinates + region labels + behavior (tail tracking); best structural fit |
| DANDI dandisets (Haesemeyer thermoregulation [000697–699, 707–708](https://dandiarchive.org/); forebrain/midbrain [000235/000236](https://dandiarchive.org/); Ahrens glia/behavior [000350](https://dandiarchive.org/); mesoscale pipelines [000244](https://dandiarchive.org/)) | [dandiarchive.org](https://dandiarchive.org/) | varies; whole-brain or regional, cellular | NWB (DANDI standard) | Needs NWB → `(C,T)` importer; per-dandiset check |
| Z-Brain atlas / [ZebraFishExplorer](https://zebrafishexplorer.zib.de/about) + [mapZebrain](https://mapzebrain.org/) | ZIB / Portugues lab | atlas: 294 regions, 4,000+ traced neurons | `.mat`/`.h5`/SWC + registration tools (ANTs) | Parcellation/registration substrate, not a bulk trace dump |
| Free-swimming whole-brain (Kim/Kim 2017 Nature Methods) | via journals/DANDI | n≈4–7 larvae per study (technical) | — | Reference for naturalistic behavior; low n |
| **ZAPBench** (Zebrafish Activity Prediction Benchmark) | [github.com/google-research/zapbench](https://github.com/google-research/zapbench) / [arXiv:2503.02618](https://arxiv.org/abs/2503.02618) (ICLR 2025) | 4D light-sheet recordings of **>70,000 neurons** in larval zebrafish (2048×1328×72 volumes × 7,879 at 914 ms); motion-stabilized, voxel + cell segmentations; 9 stimulus conditions | CC-BY 4.0 data / Apache 2.0 code; `gs://zapbench-release/volumes/20240930/{traces,stimuli_features,segmentation}` | Whole-brain cellular forecasting benchmark. **Adapter implemented** (`data/zapbench.py`): one session per condition (subject = animal, session = condition), 26-d stimulus bank as a control-role modality, `--region-bins`/`--max-cells` reduction, mandatory `--rate-hz`; the condition `taxis` is the benchmark holdout → `split_by="condition"` |
| Whole-brain under oriented grating stimuli | [Zenodo 19486022](https://zenodo.org/records/19486022) | larval zebrafish whole-brain + stimuli | Zenodo | Visual-stimulus analysis track |
| Whole-brain connectomic resource (intact 7 dpf larva) | [bioRxiv 2025.06.10.658982](https://www.biorxiv.org/content/10.1101/2025.06.10.658982v1.full-text) | >40,000 neurons annotated; 30M synapses, molecular labels | vEM + confocal | Future structure-conditioned dynamics (zebrafish connectome) |
| Community dataset co-registration | [Sprague et al. 2025](https://www.sciencedirect.com/science/article/pii/S2667237524003540) | unifies community whole-brain imaging datasets | repository + trained cell-ID models | Standardization path across labs |
| Larval behavior videos | [Zenodo 7807968](https://zenodo.org/records/7807968) | 376.8 GB behavior video | Zenodo | Behavior-modality pool (Mullen et al. 2023) |
| Whole-brain **voltage** imaging | [Nature Methods 2026 (s41592-026-03179-7)](https://www.nature.com/articles/s41592-026-03179-7) / [bioRxiv 2023.12.15.571964](https://www.biorxiv.org/content/10.1101/2023.12.15.571964v1.full-text) | whole-brain voltage, ~35 s trials (200 s continuous shown) | per-paper | The ladder's "voltage" modality for zebrafish — emerging, few specimens |
| Gene-expression + activity co-mapping (WARP) | [bioRxiv 2026.02.07.704095](https://www.biorxiv.org/content/10.64898/2026.02.07.704095v1.full-text) | whole-brain, behaving larva, neuron-type identification | per-paper | Future heterogeneity/metadata axis |
| DANDI [000235](https://dandiarchive.org/dandiset/000235/0.230316.1600) / [000236](https://dandiarchive.org/dandiset/000236/0.230316.2031) / [000237](https://dandiarchive.org/dandiset/000237/0.230316.1655) / [000238](https://dandiarchive.org/dandiset/000238/0.230316.1519) thermoregulation family | [DANDI](https://dandiarchive.org/) | 000235/236/237/238: 8/9/8/6 assets; 30.6/39.3/30.1/25.9 GB | NWB; `TwoPhotonSeries` plus `BehavioralTimeSeries` | Forebrain, midbrain, hindbrain, and reticulospinal calcium dynamics under random-wave temperature stimuli |
| DANDI [001337](https://dandiarchive.org/dandiset/001337/0.250814.1917) / [001339](https://dandiarchive.org/dandiset/001339/0.250814.1916) fixed thermal stimulation | [DANDI](https://dandiarchive.org/) | 223/82 assets; 28.8/4.55 GB | NWB; two-photon population imaging | Medulla and trigeminal-ganglion hot/cold stimulus-response dynamics |
| DANDI [001453](https://dandiarchive.org/dandiset/001453/0.250518.1950) visuomotor strategies | [DANDI](https://dandiarchive.org/) | 37 assets; 3.82 GB; 25 subjects | NWB; current summary lists `BehavioralTimeSeries` only | Useful behavior/control candidate, but do not assume neural traces from the title |
| DANDI [001569 draft](https://dandiarchive.org/dandiset/001569/draft) visual/photo stimulation | [DANDI](https://dandiarchive.org/) | Draft; 13 assets reported by the version endpoint | Draft NWB candidate | Calcium fluorescence during visual and photo stimulation; not a released version |
| Dryad — [Inhibition drives habituation of a larval zebrafish visual response](https://datadryad.org/stash/dataset/doi:10.5061/dryad.jdfn2z3fc) | [Data Dryad](https://datadryad.org/) | 31.12 GB; larval zebrafish | ZIP; calcium imaging plus behavioral data | Repeated dark-flash Ca2+ imaging; adaptation and habituation dynamics; strongest Dryad zebrafish learning track |
| Dryad — [Zebrafish retinal ganglion cells asymmetrically encode spectral and temporal information](https://datadryad.org/stash/dataset/doi:10.5061/dryad.7sqv9s4pm) | [Data Dryad](https://datadryad.org/) | 590.62 MB; larval zebrafish | CSV/IBW plus MATLAB | Two-photon hyperspectral visual stimulation; spectral/temporal response dynamics; processed rather than NWB |
| Figshare — [Calcium imaging of spontaneous activity in larval zebrafish tectum](https://figshare.com/articles/dataset/Calcium_imaging_of_spontaneous_activity_in_larval_zebrafish_tectum/6943265) | [Figshare](https://figshare.com/) | 9.90 MB; 2 larvae | ZIP; GCaMP6s dF/F | Optic-tectum spontaneous activity; smallest zebrafish pilot; CC BY 4.0 |
| PLOS Figshare — [Neurotransmitter-mediated activity spatially controls neuronal migration](https://plos.figshare.com/articles/dataset/Neurotransmitter-mediated_activity_spatially_controls_neuronal_migration_in_the_zebrafish_cerebellum/5756700) | [PLOS Figshare](https://plos.figshare.com/) | Supplementary files | TIFF/XLSX/AVI | Calcium-transient and migration assays; auxiliary developmental dynamics, not a canonical continuous trace corpus |

### Cross-repository dynamics shortlist

The current search found strong zebrafish dynamics records in DANDI, Dryad, and Figshare. The exact `zebrafish` keyword query on OpenNeuro returned no confirmed dataset, so OpenNeuro is not currently a zebrafish source in this registry.

Use [000237 Hindbrain](https://dandiarchive.org/dandiset/000237/0.230316.1655) or the compact Figshare tectum dataset for initial zebrafish loader tests; use Dryad habituation when adaptation across repeated stimuli is the target.


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
| DANDI [000206](https://dandiarchive.org/dandiset/000206/0.220103.2119) visual cortical activity | [DANDI](https://dandiarchive.org/) | 1 asset, 118 MB, 1 mouse | NWB; imaging plus `SpatialSeries`/position | Smallest complete calcium/behavior pilot |
| DANDI [000039](https://dandiarchive.org/dandiset/000039/0.230223.1216) contrast tuning | [DANDI](https://dandiarchive.org/) | 100 assets, 22.6 GB | NWB; `TwoPhotonSeries`, `Units`, behavior | Visual-stimulus calcium dynamics with electrophysiology metadata |
| DANDI [000017](https://dandiarchive.org/dandiset/000017/0.240329.1926) distributed coding | [DANDI](https://dandiarchive.org/) | 39 assets, 14.7 GB | NWB; units, pupil, behavioral events/epochs | Choice, action, and engagement dynamics |
| DANDI [001695](https://dandiarchive.org/dandiset/001695/0.260319.2023) hippocampal-cortical dynamics | [DANDI](https://dandiarchive.org/) | 22 assets, 3.09 GB, 6 mice | NWB; Neuropixels/SiNAPS, LFP, units, spatial position | Best medium-size ephys/spatial-behavior pilot |
| DANDI [001425](https://dandiarchive.org/dandiset/001425/0.250705.0947) BraiDyn-BC | [DANDI](https://dandiarchive.org/) | 1,838 assets, 8.06 TB, 25 mice | NWB plus video; widefield/one-photon imaging and behavior | Longitudinal motor learning; high-value but too large for first download |
| DANDI [000021](https://dandiarchive.org/dandiset/000021/0.251116.2246) Allen Visual Coding Neuropixels | [DANDI](https://dandiarchive.org/) | 214 assets, 477.6 GB, 32 mice | NWB; LFP and sorted units | Large visual-ephys benchmark |
| DANDI [000409](https://dandiarchive.org/dandiset/000409/0.260309.1324) IBL Brain Wide Map | [DANDI](https://dandiarchive.org/) | 2,048 assets, 49.7 TB, 139 mice | NWB; units, position, behavior | Whole-brain scale-up after the loader is stable |
| DANDI [000003](https://dandiarchive.org/dandiset/000003/0.260218.2052) hippocampal granule/mossy cells | [DANDI](https://dandiarchive.org/) | 101 assets, 2.56 TB | NWB; LFP, units, position, maze behavior | Rich hippocampal dynamics; heavy acquisition |
| DANDI [000048 draft](https://dandiarchive.org/dandiset/000048/draft) 2P calcium/ephys calibration | [DANDI](https://dandiarchive.org/) | Draft; 1 asset, 590 MB | Draft NWB; simultaneous fluorescence and spiking | Useful calcium-to-spike calibration candidate; not a released version |
| fMRI + high-resolution behavior | OpenNeuro [ds004402](https://openneuro.org/datasets/ds004402) | 10 mice, 8 sessions, 1,515 files, 17.23 GB | BIDS MRI; odor-discrimination task | Strong mouse fMRI/behavior pairing for dynamic-state modeling |
| Resting-state fMRI | OpenNeuro [ds007100](https://openneuro.org/datasets/ds007100/versions/1.0.3) | 82 mice, 330 sessions, 6,865 files, 472.90 GB | BIDS MRI; awake resting state | Largest directly verified OpenNeuro mouse dynamics pool in this search; parcellate to regions |
| Longitudinal BOLD rs-fMRI | OpenNeuro [ds006663](https://openneuro.org/datasets/ds006663/versions/1.0.3) | 69 mice, PND30/PND90 sessions, 4,310 files, 18.03 GB | BIDS MRI; BOLD rs-fMRI plus structural/DWI | Longitudinal developmental axis; separate age/session effects from neural state |
| Optogenetic fMRI | OpenNeuro [ds001541](https://openneuro.org/datasets/ds001541/versions/1.1.3) | 16 mice, 9 sessions, 323 files, 1.17 GB | BIDS MRI; DRN optogenetic task | Stimulus-locked perturbation track; compact fMRI pilot |
| Whisker-stimulation fMRI | OpenNeuro [ds005496](https://openneuro.org/datasets/ds005496/versions/1.0.1) | 6 mice, 12 sessions, 191 files, 11.90 GB | BIDS MRI; whisker-stimulation task | Sensory-evoked fMRI dynamics with repeated sessions |
| Simultaneous 2P voltage/calcium + LFP | Dryad [Cecchetto et al.](https://datadryad.org/stash/dataset/doi:10.5061/dryad.dbrv15f23) | 160.94 MB | ZIP; awake/anesthetized barrel cortex | Best compact cross-modal mouse candidate; spontaneous and whisker-evoked signals |
| Longitudinal 2P calcium | Dryad [Long-term stability of cortical ensembles](https://datadryad.org/stash/dataset/doi:10.5061/dryad.cfxpnvx5m) | 4.07 GB; six mouse identifiers in file names | `.mat` | Same layer-2/3 visual-cortex cells tracked over weeks; longitudinal ensemble dynamics |
| Voltage imaging during learning | Dryad [Emerging experience-dependent dynamics](https://datadryad.org/stash/dataset/doi:10.5061/dryad.h18931zmm) | 266.06 GB | ZIP; S1 voltage imaging plus behavior | Direct adaptation/learning dynamics; too large for the first pilot |
| Calcium imaging + behavior | Dryad [Amplitude modulations of cortical sensory responses](https://datadryad.org/stash/dataset/doi:10.5061/dryad.tb2rbnzxv) | 1.61 GB | `.mat`; calcium traces and synchronized task data | Visual-cortex/retrosplenial evidence-accumulation track; condensed data |
| Widefield calcium + ephys + optogenetics | Dryad [Separable gain control of ongoing and evoked activity](https://datadryad.org/stash/dataset/doi:10.5061/dryad.931zcrjgk) | 2.52 MB | `.mat`; condensed derived data | Very small pilot for ongoing-versus-evoked gain; not a raw recording corpus |
| Simultaneous calcium imaging + electrophysiology | Figshare [A comparison of neuronal population dynamics](https://figshare.com/articles/dataset/Raw_data_for_A_comparison_of_neuronal_population_dynamics_measured_with_calcium_imaging_and_electrophysiology_/12792587) | 2.37 GB; 2 ZIP files | Raw ZIP; V1 calcium/ephys plus ALM task calcium | Strongest Figshare mouse neural-dynamics candidate; CC BY 4.0 |

### Cross-repository dynamics shortlist

OpenNeuro contributes standardized BIDS fMRI time series and behavior, while Dryad and Figshare contribute smaller but more heterogeneous `.mat`, ZIP, CSV/IBW, and derived-data records. These sources should remain separate ingestion adapters rather than being forced into one raw-file format.

Use Dryad [Cecchetto et al.](https://datadryad.org/stash/dataset/doi:10.5061/dryad.dbrv15f23) or Figshare [12792587](https://figshare.com/articles/dataset/Raw_data_for_A_comparison_of_neuronal_population_dynamics_measured_with_calcium_imaging_and_electrophysiology_/12792587) for calcium/electrophysiology alignment; use OpenNeuro [ds004402](https://openneuro.org/datasets/ds004402) or [ds007100](https://openneuro.org/datasets/ds007100/versions/1.0.3) for BOLD dynamics.


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
name filtering, masks, canonical manifest). Zebrafish and mouse source
adapters now live in `data/corpus_pipeline.py`, `data/ingest_zebrafish.py`,
and `data/ingest_mouse.py`; remaining work is executing downloads/ingestions
against the selected records. Stimulus control is wired end-to-end (roles →
perturbation reduction → conditioned gates; `noise_mode` policy
`off|rollout|train|always`); conditioned diffusion (P2.2) remains deferred.

### 6.3 Implemented zebrafish and mouse pipeline

The two new species pipelines use one canonical ladder while keeping source
adapters separate. DANDI NWB, Dryad/Figshare extracted arrays, and OpenNeuro
BIDS files are converted before training; `SpeciesSignalDataset` never reads
repository-specific formats.

```mermaid
flowchart LR
    accTitle: Cross Species Data Pipeline
    accDescr: Source-specific zebrafish and mouse records are acquired explicitly, converted to the canonical ladder, validated, windowed in seconds, split by subject, and routed by modality role into training loaders.

    source_manifest([📋 Source manifest]) --> acquire[📥 Acquire and extract]
    acquire --> source_adapter[🔌 Run source adapter]
    source_adapter --> canonical_ladder[(💾 Canonical ladder)]
    canonical_ladder --> validate_ladder[🧪 Validate arrays and masks]
    validate_ladder --> seconds_windows[⚙️ Window in seconds]
    seconds_windows --> subject_split[👥 Split by subject]
    subject_split --> role_routing[🎯 Route signal and control]
    role_routing --> train_loaders([✅ Train and validation loaders])

    classDef input fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#3b0764
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class source_manifest input
    class acquire,source_adapter,validate_ladder,seconds_windows,subject_split,role_routing process
    class canonical_ladder,train_loaders success
```

#### Zebrafish

| Source family | Adapter | Canonical roles | Required metadata |
|---|---|---|---|
| DANDI NWB thermoregulation and thermal stimulation | `format: nwb` | `calcium` signal; optional `stimulus` control; optional `behavior` aux | NWB `series`, per-session rate or timestamps, subject/session/condition |
| Dryad/Figshare MAT/NPZ/CSV/TSV exports | `format: array` | `calcium` or `voltage` signal; optional control/aux arrays | `key`, explicit `orientation`, rate or `dt`, optional channel IDs |
| ZAPBench light-sheet release | Existing `data/zapbench.py` adapter | `calcium` signal; `stimulus` control | release clock, condition split, cell/region reduction |

Use `configs/data/zebrafish_crossrepo.json` for the federated ladder. The
profile treats behavior as `aux`, stimulus as `control`, uses 30-second
physical windows, and keeps the manifest clock authoritative because the
collected records do not share one sampling rate.

```bash
python -m brain_moe_pinn.data.ingest_zebrafish ingest \
  --source-manifest /data/zebrafish_sources.json \
  --out /data/data_ladder/zebrafish_crossrepo
python -m brain_moe_pinn.data.ingest_zebrafish validate \
  --root /data/data_ladder/zebrafish_crossrepo
```

#### Mouse

| Source family | Adapter | Canonical roles | Spatial policy |
|---|---|---|---|
| DANDI NWB calcium/ephys/widefield | `format: nwb` | optical/electrical modalities as signals; behavior as aux | preserve ROI/channel IDs when supplied |
| Dryad/Figshare MAT/NPZ/CSV/ZIP exports | extract first, then `format: array` | calcium, voltage, widefield, behavior | explicit orientation; no silent channel identity |
| OpenNeuro BIDS fMRI | `format: nifti` or `bids-fmri` convenience command | `fmri` signal | atlas labels produce parcels; without an atlas use deterministic variance top-k and record that provenance |

Use `configs/data/mouse_crossrepo.json` for federated calcium, voltage,
widefield, fMRI, and behavior. The mouse species profile now uses
`sample_rate_hz_source: "manifest"`; 30 Hz is only the nominal optical clock,
while OpenNeuro TR and per-recording electrical/optical rates remain in each
manifest row.

```bash
python -m brain_moe_pinn.data.ingest_mouse bids-fmri \
  --bids-root /data/openneuro/ds004402 \
  --subject 01 --task odor --atlas /data/atlas.nii.gz \
  --out /data/data_ladder/mouse_crossrepo
python -m brain_moe_pinn.data.ingest_mouse ingest \
  --source-manifest /data/mouse_sources.json \
  --out /data/data_ladder/mouse_crossrepo
python -m brain_moe_pinn.data.ingest_mouse validate \
  --root /data/data_ladder/mouse_crossrepo
```

#### Source manifest contract

Each source manifest has one row per subject/session source. Relative paths
are resolved against the manifest file; modality arrays are converted to
`(C,T)` `float32`, finite-value masks are persisted, and optional channel IDs
are written beside the array.

```json
{
  "species": "mouse",
  "samples": [{
    "sample_id": "ds004402__sub-01__ses-01",
    "subject": "sub-01",
    "session": "ses-01",
    "origin": "OpenNeuro:ds004402",
    "condition": "odor_discrimination",
    "modalities": {
      "fmri": {
        "format": "nifti",
        "path": "sub-01/ses-01/func/sub-01_task-odor_bold.nii.gz",
        "tr_s": 1.0,
        "atlas": "atlas.nii.gz"
      },
      "calcium": {
        "format": "nwb",
        "path": "session.nwb",
        "series": "RoiResponseSeries",
        "orientation": "tc"
      }
    }
  }]
}
```

The implementation is in `data/corpus_pipeline.py`, with species commands in
`data/ingest_zebrafish.py` and `data/ingest_mouse.py`. Square arrays require an
explicit orientation; NWB volumes with more than two dimensions require an
explicit reduction mode; no adapter downloads data implicitly.

The previous integration status is updated accordingly: source adapters now
cover NWB, NIfTI/BIDS, and extracted array records; remaining work is
acquisition, source-manifest authoring for each selected record, and running
the ingestion against the downloaded corpora.

### 6.3.1 Human EEG/MEG/fMRI ingestion

The human catalog in `EEG-Datasets.md` spans open BIDS/OpenNeuro records,
clinical EDF/BrainVision exports, FIF-based MEG, and restricted repositories.
The project therefore uses one local, source-manifest boundary rather than
dataset-specific download code:

| Source | Adapter | Canonical output | Required provenance |
|---|---|---|---|
| Scalp EEG (HBN, ABCD, TUH, LEMON, MOABB, sleep/BCI records) | `format: mne` via MNE; EDF/BDF/FIF/BrainVision/EEGLAB | `eeg/<sample>.npy` `(C,T)` plus `eeg_ids/` and optional `eeg_mask/` | subject/session/task, source path, channel names, measured/resampled rate |
| Whole-head MEG (Cam-CAN, HCP MEP, ds000117, OMEGA) | `format: mne` via MNE; FIF/CTF | `meg/<sample>.npy` `(C,T)` plus `meg_ids/` and optional `meg_mask/` | subject/session/task, selected MEG types, source path, measured/resampled rate |
| BOLD fMRI (UKB/HCP/ABCD/ABIDE/Narratives and clinical or naturalistic sets) | `format: nifti` via nibabel; optional atlas NIfTI | `fmri/<sample>.npy` `(R,T)` plus `fmri_ids/` and optional `fmri_mask/` | subject/session/task, TR, atlas or deterministic voxel policy |

`data/ingest_human.py` exposes `ingest`, `bids-session`, and `validate`.
`bids-session` writes one row shared by selected modalities, but preserves
`eeg_rate_hz`, `meg_rate_hz`, and `fmri_rate_hz` independently.  It never
infers synchronization from matching BIDS entities: pass
`cross_modal_label=1` only for a verified simultaneous recording and `0` for
an intentionally asynchronous pair.  `format: mne` accepts explicit channel
types or channel picks; it does not silently drop channels when
`max_channels` is exceeded.  EEG/MEG filters and resampling are opt-in source
spec fields.  fMRI atlas extraction is preferred; the no-atlas
`max_channels` path is deterministic variance top-k voxel selection and is
recorded as provenance, not a cross-study region space.

For a non-BIDS or multi-source session, the same contract is expressed
directly in JSON.  Relative paths are resolved against the manifest directory:

```json
{
  "species": "human",
  "samples": [{
    "sample_id": "ds006040__sub-01__task-rest",
    "subject": "sub-01",
    "session": "ses-01",
    "origin": "OpenNeuro:ds006040",
    "condition": "rest",
    "cross_modal_label": 1,
    "modalities": {
      "eeg": {
        "format": "mne",
        "path": "sub-01_task-rest_eeg.edf",
        "modality": "eeg",
        "target_rate_hz": 256,
        "channel_types": ["eeg"],
        "l_freq": 0.1,
        "h_freq": 100,
        "notch_freqs": [50, 60]
      },
      "meg": {
        "format": "mne",
        "path": "sub-01_task-rest_meg.fif",
        "modality": "meg",
        "target_rate_hz": 256,
        "channel_types": ["meg"]
      },
      "fmri": {
        "format": "nifti",
        "path": "sub-01_task-rest_bold.nii.gz",
        "atlas": "schaefer400.nii.gz",
        "tr_s": 2.0
      }
    }
  }]
}
```


```bash
python -m brain_moe_pinn.data.ingest_human bids-session \
  --bids-root /data/openneuro/ds006040 --subject 01 \
  --task rest --modalities eeg fmri --eeg-rate 256 \
  --atlas /data/atlas.nii.gz --origin OpenNeuro:ds006040 \
  --out /data/data_ladder/human_bids
python -m brain_moe_pinn.data.ingest_human validate \
  --root /data/data_ladder/human_bids --modalities eeg meg fmri
```

Acquisition, NDA/DUA approval, and archive-specific extraction remain
operator responsibilities.  MNE and nibabel are optional runtime
dependencies; fMRIPrep/SPM-produced NIfTI derivatives can be consumed after
their spatial and confound decisions have been documented.  The canonical
training profile is `configs/data/human_bids.json`, which uses subject-grouped
splits and seconds-based windows.

### 6.4 Optogenetic intervention control

The Randi et al. *Neural signal propagation atlas of C. elegans* corpus is
not another stationary stimulus track. It is an intervention dataset:
the experimenter selects a target neuron, applies a time-localized light
drive, and measures the downstream propagation in the neural signals. The
target identity and the light waveform must therefore enter the control SDE
as separate causal factors.

At the data boundary, the optogenetic control vector is encoded as

$$
u_t = [a_t,\; a_t e_i],
$$

where $a_t$ is the light waveform/intensity and $e_i$ is a one-hot target code
from a fixed within-species vocabulary. Multiplying the target code by
$a_t$ preserves the invariant $u_t=0$ when the light is off. Protocol
features such as wavelength or pulse state may be appended as additional
declared control channels, but the current adapter's canonical representation
is the waveform plus target-gated code. The controlled SDE then receives the
intervention per latent step:

$$
dz_t = [f_\theta(z_t) + B_\theta(z_t)u_t]\,dt
       + \Sigma_\theta(z_t)\,dW_t.
$$

The measured propagation response is never copied into `u_t`; it remains in
the future calcium/voltage observation and in the next-window target. This
prevents post-stimulation activity from leaking into the intervention.

The source manifest declares the target vocabulary explicitly:

```json
{
  "opto": {
    "format": "array",
    "path": "randi/opto_waveform.npy",
    "orientation": "ct",
    "control_kind": "optogenetic",
    "target_id": "AVAL",
    "target_vocab": "wormbase_neuron_order.txt",
    "target_scope": "neuron",
    "gate_channel": 0,
    "active_threshold": 0.0,
    "rate_hz": 4.0
  }
}
```

`data/corpus_pipeline.py::encode_optogenetic_control` appends the
drive-gated target code to the waveform. `ingest_source_manifest` writes the
result as one `opto` control modality and records target scope, target IDs,
and vocabulary size in the manifest. A missing or unknown target is an error;
the adapter does not silently replace a target with a generic stimulus.

Use `control_reduction="peak"` for sparse pulses so a short light event is not
diluted by a long-window mean. Set `rollout_steps` so each SDE segment resolves
the pulse onset/offset; use `resample` for long, continuously varying light
drives. The `opto` contribution to `perturbation_dim` must equal the waveform
feature count plus the target-vocabulary width (plus any protocol features);
the total model width must also include every other active control modality.
An active experiment profile must list `opto` in `data.modalities`, assign
`roles.opto = "control"`, and set `features.perturbation_dim` after the
source vocabulary is fixed. The species JSONs declare the optogenetic
contract as metadata only; they do not guess a target vocabulary or resize
model weights.

The same contract applies to zebrafish and mouse, but target vocabularies are
species-specific. A registered zebrafish neuron, mouse cell, or atlas region
is a conditioning identity, not a cross-species homology claim. If cellular
registration is unavailable, parcellate the target to the declared atlas
before encoding. Optogenetic fMRI records use the same intervention vector,
while the measured BOLD response remains downstream of the existing
TR-aware HRF/emission path.

The first validation path is causal and intervention-specific:

1. held-out target neuron/region: predict propagation for a target identity
   not used in training;
2. sham/off windows: verify the encoded control is exactly zero and the
   zero-control SDE is unchanged;
3. target permutation: swap target codes while keeping the waveform fixed
   and measure the downstream propagation change;
4. temporal ablation: compare `peak` against `mean` to quantify pulse
   dilution and verify onset/offset resolution;
5. protocol holdout: hold out wavelength/intensity/pulse protocol separately
   from the target holdout.

The downloaded Randi text export is now handled by
`data/ingest_randi.py`. It validates the 0.5 s source clock, emits a
15-second three-level pulse from the notebook's event files, masks
out-of-range fluorescence frame-by-frame, preserves source labels and event
JSON, and writes a 281-position target vocabulary. The supplied labels are
partially populated and contain duplicates, so this adapter declares targets
as `recording_local_cell` rather than making an unsupported cross-recording
neuron-identity claim. Use `configs/species/c_elegans_randi.json` with
`configs/data/c_elegans_randi.json`; `opto` has width 282 (waveform plus 281
target-gated features) and sparse controls use `peak` reduction.

---

## 7. Search log

### Round 2 (Tavily CLI `tvly search`, 2026-09-10; unauthenticated rate limit)
Keywords used (basic depth, 4 results each): `zebrafish whole-brain calcium imaging dataset repository download traces`, `zebrafish larva brain recording open dataset zenodo hdf5 behaviour`, `zebrafish functional imaging timeseries dataset forecasting benchmark`, `zebrafish brain dynamics DANDI NWB larva imaging dataset`, `mouse brain dynamics open dataset widefield 2-photon ephys behavior combined`, `mouse widefield calcium imaging open data cortex sessions number mice`, `mouse electrophysiology large open dataset Allen IBL mice sessions`, `mouse fMRI BOLD resting-state open dataset awake subjects`, `mouse calcium imaging open dataset DANDI 2024 2025 sessions`, `human intracranial EEG ECoG open dataset DANDI sessions`, `simultaneous EEG fMRI open dataset paired recordings`, `comparative whole-brain imaging open datasets neuroscience`.

Key additions from this round are integrated above: **ZAPBench** (>70k neurons, Apache-2.0 forecasting benchmark), **Zenodo oriented-grating whole-brain dataset**, **larval connectome resource**, **community co-registration (Sprague 2025)**, **Kondo widefield+behavior (25 mice/364 sessions)**, **DANDI 001172 dual-color ACh+calcium**, **Allen ophys/ephys calibration**, **longitudinal awake mouse fMRI**. Human-stage finds (DANDI:000623, RAM ECoG, sleep ECoG n=39, EEG-fMRI corpora) are catalogued in `EEG-Datasets.md` per project convention.

### Round 3 (Tavily CLI, 2026-09-10; scale-focused)
Keywords: `mouse EEG dataset large number of mice open animal electroencephalogram cohort`, `largest mouse Neuropixels electrophysiology dataset number of mice open 2024 2025`, `largest zebrafish whole-brain imaging dataset number of larvae recording hours`, `mouse fMRI dataset large cohort number of mice resting state 2025 open`, `DANDI mouse calcium imaging total sessions dataset aggregate number`, `mouse widefield imaging number of mice sessions open dataset 2025`, `zebrafish larvae whole brain calcium dataset 100 fish aggregated`, `animal brain recording consortium aggregated open datasets sessions mice hours`.

Key additions (integrated in §5 and the corpus tables): Allen Neuropixels Visual Behavior (~300k neurons; [DANDI 000713](https://dandiarchive.org/dandiset/000713)); Tseng/Harvey 2P 8 mice × 286 sessions / 273,770 neurons; Kondo widefield n=25/364 sessions; awake-mouse fMRI OpenNeuro [ds007100](https://openneuro.org/datasets/ds007100/versions/1.0.3) (cohort to verify) + 14T awake study (38 mice); Nature 2025 widefield+ephys decision study (2,289 region-sessions); whole-brain voltage imaging in larval zebrafish (Nature Methods 2026); WARP gene-expression co-mapping (bioRxiv 2026); curated SWR Neuropixels corpus (Sci Data 2025); ZAPBench same-specimen connectome note. Mouse scalp-EEG remains tiny (n≈9-20/study) — no large-n EEG-class exists for rodents.

### Round 4 (agent-reach Exa + Tavily + official repository APIs, 2026-09-11; zebrafish/mouse dynamics)
Queries: `site:openneuro.org/datasets zebrafish Danio rerio calcium imaging neural dynamics`, `site:openneuro.org/datasets mouse neural dynamics calcium electrophysiology fMRI`, `site:datadryad.org/stash/dataset zebrafish Danio rerio neural calcium imaging time series`, `site:datadryad.org/stash/dataset mouse neural dynamics calcium imaging electrophysiology`, and equivalent Figshare searches. Exa reached its free MCP rate limit during this round; Tavily and official public APIs/pages were used for the verified records.

Findings:
- OpenNeuro's public GraphQL exact-keyword query returned **0 confirmed `zebrafish` datasets**. The mouse + fMRI query returned 35 matches; the registry records the strongest verified candidates: `ds004402`, `ds007100`, `ds006663`, `ds001541`, and `ds005496`.
- Dryad added larval-zebrafish habituation and retinal visual-response records, plus mouse multimodal, longitudinal calcium, voltage-imaging, and calcium/behavior records. File formats are heterogeneous and include ZIP, `.mat`, CSV, and IBW.
- Figshare added a 9.90 MB zebrafish tectum dF/F pilot and a 2.37 GB mouse calcium/electrophysiology dataset. Both records are public and list CC BY 4.0 metadata.
- No downloads or acquisitions were performed in this round; status remains **candidate / not acquired**.


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
