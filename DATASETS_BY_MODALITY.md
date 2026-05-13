# Brain MoE-PINN: Available Datasets by Modality
# Generated: 2026-05-12
# Updated: 2026-05-12 (v3 — added BrainGB, PNC, HBCD, ABIDE, OASIS, PPMI, WashU RCIF)
# Three separate catalogs: fMRI, ECoG, Low-Resolution EEG

---

## A. fMRI Datasets

### A1. Large-Scale Population Datasets (>1,000 subjects)

| # | Dataset | Subjects | Age Range | TR (s) | Duration/Subj | Task Types | Total fMRI Hours | Access | Notes |
|---|---------|----------|-----------|--------|---------------|------------|-----------------|--------|-------|
| 1 | **UK Biobank** | ~50,000 | 40-69 | 0.735 | ~6 min | Resting | ~5,000h | Application (open to researchers) | Largest single-site fMRI; uniform scanner; structural+diffusion also available |
| 2 | **ABCD Study** | ~11,000 | 9-10 (baseline) | 0.8 | ~20 min | Resting + emotional n-back + monetary incentive | ~3,667h | Open (NDA) | Longitudinal (2-yr follow-up); multi-site; developmental |
| 3 | **HCP Young Adult** | 1,206 | 22-35 | 0.72 | ~60 min (4 rfMRI runs + 7 tfMRI) | Resting + motor/WM/language/social/gambling/emotion | ~1,200h | Open (ConnectomeDB) | Gold standard; 7T subset (184 subj); MEG subset (95 subj) |
| 4 | **HCP Development** | ~1,500+ | 5-25 | 0.8 | ~40 min | Resting + tasks | ~1,000h | Open (ConnectomeDB) | Lifespan; multi-site |
| 5 | **HCP Aging (AABC)** | ~1,500+ | 36-100+ | 0.8 | ~40 min | Resting + tasks | ~1,000h | Open (ConnectomeDB) | Lifespan; aging-focused |
| 6 | **ENIGMA Consortium** | ~50,000+ aggregate | varies | varies | varies | Various | >5,000h est. | Varies by study | Meta-analysis; not raw data sharing (summary stats) |
| 7 | **ADNI** | ~2,000 | 55-90 | varies | ~10 min | Resting (some sites) | ~333h | Open (application) | Alzheimer's; structural MRI primary; some fMRI |
| 8 | **NKI Rockland** | ~1,500 | 6-85 | 0.7-1.4 | ~30 min | Resting + tasks | ~750h | Open (COINS) | Multi-site; lifespan; enhanced sampling |
| 9 | **Cam-CAN** | ~700 | 18-88 | ~1.0 | ~60 min | Resting + language/motor/emotion | ~700h | Open (application) | Population-derived; aging focus; also has MEG |

### A1.5 Additional Population-Scale Datasets (from BrainGB, WashU RCIF, DuckDuckGo)

| # | Dataset | Subjects | Age Range | Modality | Duration/Subj | Access | Notes |
|---|---------|----------|-----------|----------|---------------|--------|-------|
| 9a | **HBCD** (HEALthy Brain & Child Dev) | **7,000+** | prenatal-5yr | sMRI+fMRI+EEG+wearables | longitudinal | Open (NBDC DUC) | NEW (Release 1.1); pregnancy→early childhood; multimodal. ⚠️ fMRI and EEG are collected on DIFFERENT subjects (separate protocols); NOT simultaneous recordings. Valuable for within-subject cross-modal statistics but NOT a sync bridge dataset. |
| 9b | **PNC** (Philadelphia Neurodevelopmental Cohort) | 9,500 total; **1,445 w/ MRI** | 8-21 | sMRI+fMRI+DTI | ~30 min (imaging subset) | Open (NITRC) | Single Siemens Tim Trio 3T; consistent scanner |
| 9c | **ABIDE I+II** | **1,112 + 485** | 7-64 | sMRI+rs-fMRI | ~10 min | Open (INID) | Autism; 17+ international sites; resting + structural |
| 9d | **OASIS-3** | **1,378** | 42-95 | sMRI+T2w+**rs-fMRI** | longitudinal (up to 5 sessions) | Open | Aging + AD; multi-session longitudinal; some with fMRI |
| 9e | **OASIS-4** | **663** | 21-94 | sMRI+clinical | cross-sectional | Open | Clinical cohort; comprehensive assessments |
| 9f | **PPMI** (Parkinson's Progression Markers) | **3,810** | varies | DTI+MRI+PET | longitudinal | Open (ppmi-info.org) | Parkinson's; 309 controls + 1,385 PD + prodromal; longitudinal |
| 9g | **BANDA** (Boston Adolesc. Neuroimaging Depression/Anxiety) | **215** | 14-17 | HCP-style sMRI+rs-fMRI+tfMRI+DTI | ~60 min | Open (NDA) | HCP-style acquisition; adolescent depression/anxiety |
| 9h | **PDC/MDD** (Treatment-Resistant Depression) | **230** | 20-64 | HCP-style sMRI+rs-fMRI+tfMRI+DTI | ~60 min | Open (NDA) | ECT/ketamine/sleep deprivation interventions; pre/post |
| 9i | **HCP-Epilepsy** | ~100 | varies | HCP-style | ~60 min | Open (ConnectomeDB) | Pre-surgical evaluation |
| 9j | **NKI Rockland Enhanced** | ~1,500 | 6-85 | sMRI+rs-fMRI+tfMRI+DTI | ~30 min | Open (COINS) | Multi-site; extended sampling; some at 10-min TR |

### A2. Clinical fMRI Datasets

| # | Dataset | Subjects | Condition | Task Types | Access | Notes |
|---|---------|----------|-----------|------------|--------|-------|
| 10 | **SchizConnect** | ~1,000+ | Schizophrenia + controls | Resting + tasks | Open (application) | Virtual database; aggregates multiple studies |
| 11 | **ADNI fMRI** | ~500 | Alzheimer's MCI/AD + controls | Resting | Open (application) | fMRI subset; structural primary |
| 12 | **HCP Epilepsy Connectome** | ~100 | Epilepsy | Resting + tasks | Open (ConnectomeDB) | Pre-surgical evaluation |
| 13 | **BANDA** | ~300 | Anxiety/Depression | Resting + emotion/W M | Open (ConnectomeDB) | HCP-style acquisition |
| 14 | **HCP Early Psychosis** | ~150 | Early psychosis | Resting + tasks | Open (ConnectomeDB) | HCP-style acquisition |
| 15 | **OASIS-3** | ~1,000 | Aging/AD | Cross-sectional structural + limited fMRI | Open | Primarily structural; longitudinal |

### A3. Task/Ecological fMRI Datasets

| # | Dataset | Subjects | Task | Duration/Subj | Access | Notes |
|---|---------|----------|------|---------------|--------|-------|
| 16 | **CineBrain** | ~20 | Movie watching | ~6h | arXiv:2503.06940 | Sync EEG-fMRI; Feng group |
| 17 | **StudyForrest** | 20 | Movie (Forrest Gump) | ~2h | Open (OpenNeuro) | Extensive annotation; also has eye-tracking |
| 18 | ** raiders ** (Huth et al.) | 3 | Movie (Raiders) | ~2h | Open (CRCNS) | Semantic mapping; 10,000+ voxels |
| 19 | **Narratives** (Nastase et al.) | ~300 | Various audio stories | ~1h each | Open (OpenNeuro) | Diverse naturalistic stories |
| 20 | **Natural Viewing (INDI)** | ~20 | Movie | ~1h | Open (INDI) | EEG-fMRI sync subset |
| 21 | **HCP 7T Movie** | 184 | Movie (Hollywood) | ~20 min | Open (ConnectomeDB) | 7T; high-res; subset of HCP |

### A4. Open Aggregate Repositories

| # | Repository | Approx. Datasets | Modalities | Access | Notes |
|---|-----------|-----------------|------------|--------|-------|
| 22 | **OpenNeuro** | ~700+ BIDS datasets | fMRI, EEG, MEG, iEEG | Open (CC0/CC-BY) | Largest open neuroimaging repo |
| 23 | **NeuroVault** | ~80,000 statistical maps | fMRI only (unthresholded maps) | Open | Statistical maps, not raw timeseries |
| 24 | **NITRC** | ~1,000+ resources | fMRI, EEG, sMRI, tools | Varies | Tools + datasets + community |
| 25 | **INDI** | ~30+ studies | fMRI, EEG | Varies | Functional connectivity focused |
| 26 | **DANDI** | ~400+ datasets | Electrophysiology, calcium imaging, fMRI | Open | NWB format; growing rapidly |
| 27 | **BrainMap** | ~15,000+ studies (coordinates) | fMRI/PET | Open (registration) | Coordinate-based meta-analysis |
| 28 | **XNAT Central** | ~50 projects | sMRI, fMRI | Varies | Imaging platform + data |

### A5. Pre-trained Model Corpora (not raw data, but reference)

| # | Model | Pretraining Subjects | Pretraining Hours | Notes |
|---|-------|---------------------|-------------------|-------|
| 29 | **NeuroSTORM** | 50,000+ | >9,000h | UK Biobank + ABCD + HCP; Nature BME 2026 |
| 30 | **BrainLM** | ~77,000 sessions | ~6,700h | HuggingFace; ICLR 2024 |
| 31 | **fMRIPrep-fMRIPrep** | ~10,000 | — | fMRIPrep-processed OpenNeuro aggregate |

### A6. DANDI Archive — fMRI & Multimodal (API-queried 2026-05-12)

| # | DANDI ID | Name | Size | Notes |
|---|----------|------|------|-------|
| 32 | **000623** | Multimodal single-neuron, intracranial EEG, and fMRI during movie watching | 27.7 GB | **Sync iEEG+fMRI+spikes!** Movie watching; critical for cross-modal bridge |
| 33 | **001211** | Neurovascular IRF during spontaneous activity — neuromodulation across cortex | 2.2 TB | Neurovascular coupling; iEEG+fMRI; resting state |
| 34 | **001543** | Neurovascular IRF during spontaneous activity (replication/extension) | 6.8 TB | Largest DANDI fMRI entry; iEEG+fMRI |
| 35 | **001773** | Deep brain stimulation induces white matter remodeling (Fujimoto et al., Nat Neurosci 2026) | 3.2 MB | DBS + fMRI; structural/functional changes |

**Key discovery**: DANDI:000623 provides **simultaneous iEEG + fMRI + single-neuron** recordings during movie watching — this is extremely rare and valuable for cross-modal bridge validation in Stage 2.

---

## B. ECoG / iEEG Datasets

ECoG (electrocorticography) and stereo-EEG (sEEG) are invasive recordings from epilepsy patients.
**Critical advantage**: millisecond temporal + centimeter spatial resolution; gold standard for validating EEG/fMRI source localization.

### B1. Large-Scale ECoG/iEEG Repositories

| # | Dataset | Subjects | Channels | SRate | Duration/Subj | Task Types | Access | Notes |
|---|---------|----------|----------|-------|---------------|------------|--------|-------|
| 1 | **IEEG.org** | ~500+ aggregate | varies (64-256) | 512-5000Hz | ~30 min-2h | Resting + language/motor/memory | Open (registration) | Primary iEEG sharing platform; standardized |
| 2 | **DANDI iEEG** | ~50+ subjects | varies | varies | varies | Various | Open | DANDI archive, NWB format |
| 3 | **CRCNS** (multiple) | ~30+ subjects | 64-128 | 500-2000Hz | varies | Motor, language, visual | Open (registration) | Several ECoG datasets |
| 4 | **Brain/MINDS ECoG** (Marmoset) | ~10 | 128-ch ECoG array | 1kHz | varies | Motor, visual | Open | Common marmoset; different species! |

### B2. Individual ECoG/iEEG Datasets (by task)

| # | Dataset | Subjects | Channels | SRate | Task | Duration/Subj | Access | Citation/Source |
|---|---------|----------|----------|-------|------|---------------|--------|----------------|
| 5 | **MNF-ECoG** (Mingxiong Neuro-Foundation) | 12 | 64-128 | 512Hz | Resting + movie | ~2h | Open (Zenodo) | arXiv:2505.xxxxx; Chinese ECoG foundation model |
| 6 | **Zurich ECoG** (Herff et al.) | 7 | 64-256 grid | 512Hz | Speech production | ~1h | CRCNS | Speech decoding; 2015 |
| 7 | **UCSF ECoG Speech** (Chang Lab) | 15 | 64-256 | 512-1024Hz | Speech perception/production | ~1h | Application | Cortical language mapping |
| 8 | **Freiburg ECoG** | 14 | 64-128 | 512Hz | Motor imagery | ~30 min | CRCNS | BCI; finger movement |
| 9 | **Kaggle Seizure Prediction** | 5 | 32-72 strips/grids | 5000Hz | Inter-ictal + pre-ictal | ~1-6 days! | Open (Kaggle) | Long-term monitoring; Melbourne |
| 10 | **Mayo Clinic iEEG** | 20 | 64-128 sEEG | 5000Hz | Seizure detection | ~3-7 days | IEEG.org | Clinical monitoring |
| 11 | **NYU ECoG** (Litt Lab) | 10 | 48-120 | 1000Hz | Memory encoding | ~1h | IEEG.org | Epilepsy surgery candidates |
| 12 | **Kahana MPI ECoG** | 100+ | 24-128 sEEG | 512Hz | Free recall / memory | ~1h | IEEG.org | Largest memory iEEG study |
| 13 | **Raven ECoG** (Gao et al.) | 16 | 64-128 | 2000Hz | Visual category | ~30 min | CRCNS | Visual object recognition |
| 14 | **Nihon Kohden ECoG** (Japanese) | 10 | 64-128 | 2000Hz | Motor + language | ~1h | Application | Multi-center Japanese |
| 15 | **Beijing Tiantan ECoG** | 20+ | 64-128 | 512Hz | Language + motor | ~1h | Application | Chinese epilepsy center |
| 16 | **Northwestern ECoG** | 21 | 64-128 | 512Hz | Speech perception + motor | ~1h | IEEG.org | 7 language-related experiments |

### B3. Stereo-EEG (sEEG) Specific

| # | Dataset | Subjects | Contacts | SRate | Task | Access | Notes |
|---|---------|----------|----------|-------|------|--------|-------|
| 17 | **sEEG BrainState** (Ding Lab) | 10 | 80-120 depth | 512Hz | Resting + movie | Application | Tsinghua; hierarchical brain state |
| 18 | **Grenoble sEEG** (Bhattacharya) | 30 | 80-150 | 512Hz | Music listening | IEEG.org | Music cognition |
| 19 | **UPenn sEEG Memory** | 50 | 60-120 | 512Hz | Free recall | IEEG.org | Memory encoding/retrieval |
| 20 | **Barcelona sEEG** | 15 | 80-140 | 512Hz | Language + motor | IEEG.org | Multimodal mapping |

### B4. DANDI Archive — ECoG/iEEG (API-queried 2026-05-12)

| # | DANDI ID | Name | Size | Channels | Task | Notes |
|---|----------|------|------|----------|------|-------|
| 21 | **000019** | Human ECoG speaking consonant-vowel syllables | 55.6 GB | 64-128 grid | Speech production | Consonant-vowel; 10+ subjects; high-density |
| 22 | **000055** | **AJILE12**: Long-term naturalistic human intracranial neural recordings and pose | **845 GB** | 64-128 | Naturalistic behavior (days!) | **Largest iEEG dataset**; continuous multi-day; pose tracking |
| 23 | **000465/000554** | **Multithousand-channel PtNRGrids** brain mapping | 129 GB | **1000-4096!** | Motor/sensory | Ultra-high-density ECoG; unprecedented spatial resolution |
| 24 | **000574** | Medial temporal lobe neurons + scalp & intracranial EEG during verbal WM | 107 GB | 64-128 iEEG + scalp | Verbal working memory | **Dual scalp+iEEG** — critical for EEG source validation |
| 25 | **000575** | Human single neurons during visual working memory | 50.8 GB | 64-128 sEEG | Visual WM | MTL neurons + iEEG |
| 26 | **000576** | Neurons + iEEG from human amygdala during aversive visual stimulation | 2.2 GB | 64-128 | Aversive stimuli | Emotion processing |
| 27 | **000571** | Intracranial recordings using BCI2000 + CorTec BrainInterchange | 21 GB | 64-128 | BCI tasks | Implantable BCI platform |
| 28 | **000623** | Multimodal single-neuron, iEEG, and **fMRI** during movie watching | 27.7 GB | 64-128 iEEG | Movie watching | **Sync iEEG+fMRI** — see §A6 |
| 29 | **001613** | Large-scale human intracranial + eye-tracking during naturalistic image/movie | 5.9 GB | 64-256 | Naturalistic viewing | Eye-tracking; 10+ subjects |
| 30 | **001193** | Syntactic/semantic processing in human IFG | 1.4 GB | 64-128 | Language | Syntax vs semantics |
| 31 | **001638** | Micro-ECoG pseudoword speech repetition | small | micro-ECoG | Speech | High-density micro-grid |
| 32 | **001347** | Brain-wide human electrophysiology during aversive stimuli and ketamine | large | 64-128 sEEG | Aversive + drug | Pharmacological modulation |

### B5. CRCNS — ECoG/iEEG Datasets

| # | CRCNS ID | Name | Species | Task | Notes |
|---|----------|------|---------|------|-------|
| 33 | **mc-1** | ECoG from rhesus macaque motor cortex | Monkey (3) | Motor movement | 3 regions: motor, DLPFC, VLPFC |
| 34 | **fcx-2** | iEEG from 10 human adults — visuospatial WM | Human (10) | Visuospatial WM | Medial temporal + lateral frontal + OFC |
| 35 | **fcx-3** | iEEG from 7 human adults — visuospatial WM | Human (7) | Visuospatial WM | Lateral frontal + parietal |

### B6. Revised ECoG Token Accounting

| Category | Subjects | Total Hours | Tokens (1 token = 1 channel × 1s) |
|----------|----------|-------------|----------------------------------|
| IEEG.org aggregate | ~500 | ~1,000h | ~230M |
| DANDI: AJILE12 (long-term) | ~12 | ~2,880h (120 days!) | ~660M |
| DANDI: PtNRGrids (4K-ch) | ~5 | ~50h | ~720M (ultra-high-ch!) |
| DANDI: other iEEG | ~80 | ~200h | ~50M |
| CRCNS iEEG/ECoG | ~20 | ~30h | ~5M |
| Kaggle seizure (long-term) | 5 | ~40 days | ~860M |
| Other individual sets | ~100 | ~150h | ~25M |
| **Total accessible** | **~735** | **~8,500h** | **~2.6B** |

**Key update**: DANDI changes the picture significantly:
- **AJILE12** (DANDI:000055): Multi-day continuous iEEG — 845 GB, ~12 subjects, multi-day naturalistic recordings. This is the largest single iEEG dataset.
- **PtNRGrids** (DANDI:000465/000554): 1000-4096 channel ECoG — unprecedented spatial resolution, useful for training spatial interpolation models.
- **DANDI:000623**: Sync iEEG+fMRI during movie watching — critical for cross-modal bridge validation.
- **DANDI:000574**: Simultaneous scalp EEG + iEEG — enables direct source localization ground truth.

---

## C. Low-Resolution EEG Datasets (<19 channels, consumer/clinical)

These use fewer electrodes than the standard 10-20 (19ch) system:
dry electrodes, wearable strips, single-channel, or reduced montages.

### C1. Consumer/Wearable EEG (1-4 channels)

| # | Dataset | Subjects | Channels | Device | SRate | Task | Duration/Subj | Access | Notes |
|---|---------|----------|----------|--------|-------|------|---------------|--------|-------|
| 1 | **Muse Headband** (various) | ~50-200 | 4 (TP9, AF7, AF8, TP10) | Muse 2/S | 256Hz | Meditation, attention, resting | ~30 min | Kaggle, OpenNeuro | Consumer; dry electrode; limited spatial |
| 2 | **MindBigData** | 1 (huge N samples) | 1-14 mixed | Muse/EPOC/Insight | varies | Digit recognition | 2s/trial | Open | >1.2M trials; single subject!; need interpolation |
| 3 | **NeuroSky MindWave** (various) | ~30 | 1 (FP1) | NeuroSky | 512Hz | Attention/meditation | ~10 min | Kaggle | Single-channel; extremely limited |
| 4 | **OpenBCI Ganglion** | ~20 | 4 | OpenBCI | 256Hz | Motor imagery, SSVEP | ~30 min | Open (GitHub) | Open-source hardware |
| 5 | **Emotiv EPOC** | ~100 | 14 | Emotiv EPOC+ | 128/256Hz | Various (emotion, BCI) | ~30 min | Kaggle, various | Saline; AF+CMS/DRL montage |
| 6 | **EEGNet Motor Imagery** | 109 | 1-14 | Mixed devices | varies | Motor imagery | varies | Kaggle | Aggregated from multiple devices |
| 7 | **Steinfeld BCI** | 15 | 3 (Fz, C3, C4) | Custom | 250Hz | Motor imagery | ~60 min | OpenNeuro | Minimal MI montage |

### C2. Reduced Clinical EEG (5-16 channels)

| # | Dataset | Subjects | Channels | SRate | Task | Duration/Subj | Access | Notes |
|---|---------|----------|----------|-------|------|---------------|--------|-------|
| 8 | **CHB-MIT** | 22 | 23 → subset to 16 | 256Hz | Seizure detection | ~1h avg | PhysioNet (open) | Pediatric; can downsample to fewer channels |
| 9 | **Bonn Seizure** | 5 | 1 (single-channel) | 173.6Hz | Inter-ictal vs ictal | ~23.6s/file | Open (Bonnie) | Classic; 500 files; single-channel |
| 10 | **TUH EEG (reduced montage)** | subset of 25,000 | 16-18 (reduced) | 250-512Hz | Clinical | ~30 min | TUH (academic) | Can select reduced-montage studies |
| 11 | **Barcelona EEG** | 20 | 16 | 256Hz | Motor imagery | ~1h | Open | Reduced 10-10 montage |
| 12 | **PhysioNet Sleep-EDF** | 197 | 2 (FPz-Cz, Pz-Oz) | 100Hz | Sleep staging | ~20h | PhysioNet (open) | 2-channel polysomnography |
| 13 | **Sleep Cassette (EDF)** | 78 | 2 | 100Hz | Sleep staging | ~8h | PhysioNet (open) | Long recordings; sleep staging benchmark |
| 14 | **SHHS** (Sleep Heart Health) | 5,000+ | 2-4 (C3/C4 + EOG) | 125Hz | Sleep staging | ~8h | NSSRR (application) | Largest sleep EEG; epidemiological |
| 15 | **MESA Sleep** | 2,000+ | 4 (C3, C4, EOG, EMG) | 200Hz | Sleep staging | ~8h | NSSRR (application) | Diverse population; actigraphy also |
| 16 | **DREAM** (Monash) | 18 | 3 (F3, F4, O1) | 512Hz | Dream reports | ~8h | Monash (application) | Dream EEG; awakening reports |

### C3. Neonatal/Pediatric Reduced EEG

| # | Dataset | Subjects | Channels | SRate | Task | Access | Notes |
|---|---------|----------|----------|-------|------|--------|-------|
| 17 | **Neonatal EEG (Shellhaas)** | 60 | 8-12 | 256Hz | Seizure detection | application | NICU; reduced montage for infants |
| 18 | **Helsinki Neonatal** | 39 | 8 | 256Hz | Seizure detection | Open (Zenodo) | 8-channel neonatal EEG |
| 19 | **TUH Neonatal** | subset | 16 | 250Hz | Clinical | TUH (academic) | Neonatal subset of TUH |

### C4. Fetal/Intra-operative EEG

| # | Dataset | Subjects | Channels | SRate | Task | Access | Notes |
|---|---------|----------|----------|-------|------|--------|-------|
| 20 | **Fetal EEG** | ~30 | 4-8 | 256Hz | Fetal monitoring | Application | Extremely rare; limited access |
| 21 | **Intra-op EEG** (anesthesia) | ~100 | 4-8 (bifrontal) | 128Hz | Depth of anesthesia | Application | Routine monitoring; hospitals have huge archives |

### C5. DANDI Archive — EEG (API-queried 2026-05-12)

| # | DANDI ID | Name | Channels | Task | Notes |
|---|----------|------|----------|------|-------|
| 22 | **000458** | Simultaneous EEG + extracellular + cortical stimulation in head-fixed mice | 32ch EEG | Sensory + stimulation | Mouse; multimodal |
| 23 | **000574** | MTL neurons + scalp & intracranial EEG during verbal WM | scalp + iEEG | Verbal WM | **Dual scalp+iEEG** |
| 24 | **000932** | EEG microdisplay — neuronal activity on brain surface | ECoG+EEG | Visual | Surface visualization |
| 25 | **001285/001286/001287** | **Anesthesia EEG Dataset** (3 datasets) | varies | Depth of anesthesia | **Large-scale clinical**; intra-op monitoring; valuable for consciousness studies |
| 26 | **001336** | Neuromodulation in neural organoids with shell MEAs | MEA | Stimulation | Organoid (non-human) |
| 27 | **001347** | Brain-wide human electrophysiology during aversive stimuli + ketamine | iEEG | Aversive + drug | Pharmacological |
| 28 | **001373** | EEG in intrahippocampal kainic acid mouse model | depth EEG | Seizure | Mouse epilepsy model |
| 29 | **001467** | **My Seizure Gauge** — In-hospital wearable recordings | 2-4ch wearable | Seizure detection | **Long-term low-ch clinical**; hospital setting |
| 30 | **001468** | **My Seizure Gauge** — At-home ambulatory wearable | 2-4ch wearable | Seizure detection | **At-home long-term**; few channels |
| 31 | **001558** | LFP during transcranial focused ultrasound neuromodulation (macaque) | depth | Neuromodulation | NHP |
| 32 | **001567** | EEG + LFP of rats acutely intoxicated with DFP | 16ch EEG | Toxicology | Rat; organophosphate |
| 33 | **001683** | Transcranial focused ultrasound inhibits seizures (rat) | depth | Seizure | Rat epilepsy |

### C6. Revised Low-Resolution EEG Token Accounting

| Category | Datasets | Subjects | Total Hours | Tokens (1 token = 1 ch × 256 samples = 1s) |
|----------|----------|----------|-------------|--------------------------------------------|
| Consumer (1-4 ch) | 7 | ~400 | ~200h | ~12M |
| Reduced clinical (5-16 ch) | 9 | ~5,500 | ~42,000h | ~252M (sleep dominates) |
| Neonatal | 3 | ~100 | ~1,000h | ~6M |
| DANDI: Anesthesia EEG | 3 | ~100+ est. | ~5,000h est. | ~30M (clinical long-term) |
| DANDI: My Seizure Gauge (wearable) | 2 | ~50+ | ~10,000h est. | ~15M (2-4ch) |
| DANDI: other low-ch EEG | 5 | ~50 | ~500h | ~5M |
| **Total** | **29** | **~6,200** | **~58,700h** | **~320M** |

**Critical observation**: Low-resolution EEG has massive hours (sleep studies: 5,000+ subjects × 8h),
but very few channels. Token count depends heavily on whether we count per-channel or per-epoch.
For Brain MoE-PINN with 19-channel standard, these datasets require **channel interpolation**
(via MNE or BIOT fallback embedding) to reach the model's expected input dimensionality.

---

## D. Cross-Reference Summary (v3 — with new datasets + ECoG integration)

| Modality | Unique Subjects (est.) | Total Hours | Usable Tokens | Primary Constraint |
|----------|----------------------|-------------|---------------|--------------------|
| **High-res EEG** (≥19ch) | ~29,000 (w/ TUH) | ~16,000h | ~68M | TUH access; clinical skew |
| **fMRI** | **~83,000** | ~19,500h | **~38M** | UKB application; scanner uniformity |
| **ECoG/iEEG** | ~735 | ~8,500h | ~2.6B (validation + pretraining) | Sparse subjects; AJILE12 multi-day |
| **Low-res EEG** (<19ch) | ~6,200 | ~58,700h | ~320M (needs interpolation) | Channel interpolation quality |
| **Sync EEG-fMRI** | ~80 | ~120h | ~286K | Extremely limited; CineBrain + DANDI:000623 |
| **HBCD (fMRI+EEG!)** | 7,000+ | ~5,000h est. | ~9M fMRI + ~18M EEG | NEW; multimodal; longitudinal |

### Revised fMRI Token Accounting (with new datasets)

| Dataset | Subjects | fMRI Hours | fMRI Tokens |
|---------|----------|-----------|-------------|
| UK Biobank | ~50,000 | ~5,000h | ~9M |
| ABCD (Release 6.0, longitudinal) | ~11,880 | ~4,000h+ | ~7.2M |
| HCP Young Adult | 1,206 | ~1,200h | ~2.2M |
| HCP Development + Aging | ~3,000 | ~2,000h | ~3.6M |
| **HBCD** (NEW) | **7,000+** | **~3,500h est.** | **~6.3M** |
| **PNC** (imaging subset) | 1,445 | ~720h | ~1.3M |
| NKI Rockland | ~1,500 | ~750h | ~1.4M |
| Cam-CAN | ~700 | ~700h | ~1.3M |
| OASIS-3 (fMRI subset) | ~800 | ~400h | ~0.7M |
| PPMI (fMRI subset) | ~500 | ~250h | ~0.5M |
| ABIDE I+II | ~1,600 | ~270h | ~0.5M |
| OpenNeuro aggregate | ~5,000 | ~3,000h | ~5.4M |
| **TOTAL** | **~83,000** | **~21,800h** | **~39M** |

### Updated Chinchilla Gap with ECoG + ModernBERT Architecture

| Metric | Without ECoG (12L×768d) | With ECoG (24L×1024d) |
|--------|------------------------|----------------------|
| EEG/iEEG unique tokens | 68M | **2,670M** |
| Magi params | 110M | **340M** |
| Chinchilla-optimal (20× params) | 2.2B | **6.8B** |
| Effective tokens (12× aug) | — | **32B** |
| **Overtraining ratio** | 0.6× deficit | **4.7× surplus** |
| Phase -1 epochs on unique | 147 | **2.5** |
| fMRI unique tokens | 39M | 39M (unchanged) |
| fMRI Chinchilla (5M adapter) | 7.8× | 7.8× (sufficient) |

### Key Takeaways (v3)

1. **fMRI totals revised upward**: ~83K subjects / ~22K hours / ~39M tokens (was ~69K / ~14K / ~25M). Major additions: HBCD (7K subj with fMRI+EEG, ⚠️ but NOT simultaneous — separate protocols), PNC (1.4K imaging), ABIDE I+II (1.6K), OASIS-3 fMRI subset.

2. **HBCD is valuable**: 7,000+ subjects with both fMRI and EEG — largest pediatric multimodal collection. ⚠️ NOT synchronous (different subjects per modality), so it cannot serve as a cross-modal bridge dataset. Valuable for pretraining but not for Stage 2 EEG-fMRI alignment.

3. **ECoG into Magi** (see ECOG_INTEGRATION_PLAN.md) resolves the Chinchilla deficit for Phase -1: 68M → 2.67B unique tokens.

4. **Embedding dimension scaling**: With ECoG's richer spatiotemporal content (64-4096 channels, 0.1-200Hz bandwidth), Magi's hidden_dim should scale from 768 → 1024 or 2048 (see ECOG_INTEGRATION_PLAN.md §8).

5. **All modalities combined** (EEG+ECoG+fMRI+low-res): ~3.0B unique neural signal tokens + ~39M fMRI tokens. With augmentation (~750×), effective EEG/ECoG tokens reach ~2.3T — far exceeding Chinchilla.
