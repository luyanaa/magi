# Brain MoE-PINN: Comprehensive EEG/ECoG/fMRI Dataset Reference

> **Revised: 2026-05-17** — Merged EEG_DATA_AUDIT_V2.md, EEG_MISSING_DATASETS.md, ECOG_SEEG_DATA_AUDIT.md, ECOG_SEEG_NEW_DATASETS_FOUND.md, EEG_FMRI_DATASET_SEARCH.md, ECOG_INTEGRATION_PLAN.md, plus additional fMRI datasets (clinical, naturalistic, international, developmental), plus MOABB BCI benchmark datasets (60+ datasets across MI/P300/SSVEP/CVEP/speech/workload), plus MEG and fNIRS modality sections.
>
> Requirement: ≥19 channels (can interpolate from ≥4), ≥200 Hz. This is an internal reference for the Brain MoE-PINN project.

---

## A. EEG Datasets (Scalp)

### A.1 Task-Independent Pretraining (Large-Scale)

| # | Dataset | Subjects | Channels | Rate | Duration | Access | Tokens (est) | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **HBN** (Healthy Brain Networks) | ~2,000 | 128 (EGI) | 500 Hz | ~1h/subj | Open (NDA) | ~7M | NIPS 2025 EEG Foundation Challenge |
| 2 | **ABCD Study EEG** | **11,875** | 64 (BrainProducts) | 500 Hz | ~75min × 3+ timepoints | Open (DUC) | **~54M** | Adolescents 9-10, 21 US sites, diverse demographics |
| 3 | **TDBrain** | 1,200 | 33 | 500 Hz | ~1h/subj | Open (reg) | ~4M | Clinical: ADHD, depression, etc. |
| 4 | **TUH EEG Corpus v2.0.2** | **~60,000+** | 16-36 | 250-512 Hz | ~30,000h | Open (acad) | **~108M** | **Largest open clinical EEG**. 2002–present. Correction: was ~25K, official site confirms 60K+ |
| 5 | **MIPDB** (Multimodal) | ~300 | 128 | 500 Hz | ~1h/subj | Open (NDA) | ~1M | |
| 6 | **Cuban Human Brain Mapping** | ~200 | 64 | 200 Hz | ~1h/subj | Open (Synapse) | ~0.7M | |
| 7 | **Latin American Brain Health** | ~200 | 64 | varies | ~1h/subj | Open (Synapse) | ~0.7M | |
| 8 | **LEMON** (Leipzig) | ~200 | 62 | 500 Hz | ~0.5h/subj | Open | ~0.7M | Emotion + resting |
| 9 | **Resting-State across Lifespan** | 608 | 64 | 500 Hz | ~0.5h/subj | Open (OpenNeuro) | ~1.1M | ds005385 |
| 10 | **Large EEG gambling task** | ~100 | 64 | 512 Hz | ~1h/subj | Open (OSF) | ~0.4M | |
| 11 | **NMT Scalp EEG (India)** | **~2,000** | 19-21 (Nihon Kohden) | 200-500 Hz | ~2.5h/subj | Open (figshare) | **~18M** | **South Asian gap** — first major Indian clinical EEG |
| 12 | **VitalDB (Korea)** | **6,388** | 2-4 (BIS, frontal) | 100-500 Hz | ~3h/subj | Open (vitaldb.net) | ~15M (19ch interp) | **Korean + anesthesia** gap |
| 13 | **RAMSES ICU cEEG** | 77 | 19-21 | 200-500 Hz | ~65h/subj | Open | ~18M | ICU continuous monitoring |
| 14 | **NDA EEG Aggregates** | ~5,000+ | 19-128 | 250-1000 Hz | ~5,000h total | DUC | ~18M | Multiple NIMH-funded studies |
| 15 | **SzCORE Benchmark** | ~100+ | 19-23 | 256-512 Hz | ~500h | Open (GitHub) | ~1.8M | Standardized seizure EEG benchmark |

### A.2 Sleep EEG

| # | Dataset | Subjects | Channels | Rate | Duration | Access | Tokens (est) | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **SLEEP Dataset** | ~100 | 2-19+ | 200 Hz | 8h/subj | Open | ~1M | |
| 2 | **AHPHY-SLEEP** | ~50 | varies | varies | ~8h/subj | Open (OSF) | ~0.4M | |
| 3 | **DREAM** (Dream EEG) | ~50 | varies | varies | varies | Open | ~0.2M | |
| 4 | **SHHS** (Sleep Heart Health) | **6,441** | 12–16 (full PSG) | 125–256 Hz | ~8h/subj | NSSRR application | **~184M** | **Largest single sleep cohort**. Population-based; multi-site |
| 5 | **MrOS Sleep** | **3,040** | 18-20 (full PSG) | 200-256 Hz | ~8h/subj | NSSRR application | **~87M** | **Older men (67+)** — high-channel PSG |
| 6 | **Wisconsin Sleep Cohort** | **1,500+** | 16-20 (full PSG) | 200-256 Hz | ~8h/subj × multi-visit | NSSRR application | **~86M** | **Longitudinal** — multiple visits per subject |
| 7 | **Cleveland Family Sleep** | **700+** | 16-20 (full PSG) | 200-256 Hz | ~8h/subj | NSSRR application | **~20M** | **Family-based, African-American** inclusion |
| 8 | **SOF Sleep** (older women) | 460+ | 16-20 | 200-256 Hz | ~8h/subj | NSSRR application | ~13M | Complements MrOS |
| 9 | **CHAT** (Childhood Adenotonsillectomy) | **~1,000+** | 12–16 (full PSG) | 200–256 Hz | ~8h/subj | NSSRR application | ~29M | **Large pediatric sleep EEG**. Ages 5–9 |
| 10 | **BIDSleep** (wearable + EEG) | 47 | 5 (Dreem 2) | 250 Hz | ~8h/night × 253 nights | Open (PhysioNet 2026) | ~2.4M (19ch interp) | Multi-night longitudinal |

### A.3 Motor-Imagery

| # | Dataset | Subjects | Channels | Rate | LaBram/NeuroLM Pretrained | Notes |
|---|---|---|---|---|---|---|
| 1 | **Grasp and Lift EEG Challenge** | 12 | 32 | 500 Hz | ✅ | 6 grasp/lift events |
| 2 | **WAY-EEG-GAL** | 12 | 32 | 500 Hz | — | 3,936 trials, varying weight/friction |
| 3 | **Mental-Imagery Dataset** | 13 | 38 | 1000 Hz | — | 60,000+ examples, 6 mental imageries |
| 4 | **High-Gamma Dataset** | 14 | 128 | 500 Hz | — | 4 motor classes |
| 5 | **Left/Right Hand MI** | 52 | 64 | 1000 Hz | — | GigaDB |
| 6 | **Motor Movement/Imagery** | 109 | 64 | 160 Hz | ✅ | PhysioNet eegmmidb |
| 7 | **BCI Competition IV-1** | 7 | 64 | 1000 Hz | ✅ | 2 classes + idle |

### A.4 Emotion Recognition

| # | Dataset | Subjects | Channels | Rate | LaBram/NeuroLM Pretrained | Notes |
|---|---|---|---|---|---|---|
| 1 | **DEAP** | 32 | 32 | 512 Hz | — | Music-video, arousal/valence |
| 2 | **HCI-Tagging** | 30 | 32 | 512 Hz | — | Movie clips |
| 3 | **Enterface'06** | 16 | 64 | 512 Hz | ✅ | EEG + fNIRS + face video |
| 4 | **SEED** | 15 | 62 | 200 Hz | ✅ | 3 emotion classes |
| 5 | **SEED-IV** | 15 | 62 | 1000 Hz | ✅ | 4 emotion classes, 3 sessions |
| 6 | **SEED-FRA / SEED-GER** | ~30 | 62 | 1000 Hz | ✅ | Cross-cultural |
| 7 | **SEED-V** (5-class) | 16 | 62 | 200/1000 Hz | — | 5 emotion classes via film clips |
| 8 | **SEED-VIG** (vigilance) | 23 | 17 | 200 Hz | — | Driver vigilance, simulated driving |
| 9 | **MPED** (Multi-modal) | 23 | 62 | 1000 Hz | — | 7 emotion categories + ECG/EMG/GSR |
| 10 | **CEEA** (Chinese Emotion Atlas) | ~100+ | 64 | 500-1000 Hz | — | Multi-paradigm emotion induction |
| 11 | **EmoEEG-MC** | ~30 | 64 | 500 Hz | — | OpenNeuro ds005540 |
| 12 | **DENS** | ~50 | 64 | 512 Hz | — | OpenNeuro ds003751 |

### A.5 Event-Related Potentials (ERPs)

| # | Dataset | Subjects | Channels | Rate | LaBram/NeuroLM Pretrained | Notes |
|---|---|---|---|---|---|---|
| 1 | **BCI-NER Challenge** | 26 | 56 | 512 Hz | ✅ | P300 Speller ErrP |
| 2 | **Target vs Non-Target** (bi2014b) | 38 | 32 | 512 Hz | ✅ | P300 BCI, multiplayer |
| 3 | **Target vs Non-Target** (bi2015a) | 50 | 32 | 512 Hz | ✅ | P300 BCI, 3 sessions |
| 4 | **Target vs Non-Target** (bi2015b) | 44 | 32 | 512 Hz | ✅ | P300 BCI, 2-player |
| 5 | **ERP Core** | 40 | 32-64 | 500 Hz | — | 6-7 paradigms (N170, N400, etc.) |
| 6 | **Nencki-Symfonia** | 42 | 64 | 500 Hz | — | MSIT+ + oddball + SRT + resting |
| 7 | **Impedance Data** | 12 | 64 | 512 Hz | — | P300 with varying impedance |

### A.6 Cognitive Tasks

| # | Dataset | Subjects | Channels | Rate | Notes |
|---|---|---|---|---|---|
| 1 | **Face vs House** | 7 | 64 | 1000 Hz | Epilepsy patients, 300 trials |
| 2 | **SADT** (Driving) | 27 | 32 | 500 Hz | Sustained attention, VR |
| 3 | **Working Memory Filtering** | ~30 | 64 | 512 Hz | OSF |
| 4 | **CDA Working Memory** | ~30 | 64 | 512 Hz | OSF |
| 5 | **Gambling Task** | ~100 | 64 | 512 Hz | OSF |

### A.7 Resting State

| # | Dataset | Subjects | Channels | Rate | LaBram/NeuroLM Pretrained | Notes |
|---|---|---|---|---|---|---|
| 1 | **TxState Resting** | 22 | 72 | 500 Hz | — | 8 min: eyes open/closed |
| 2 | **SPIS Resting State** | 10 | 64 | 512 Hz | ✅ | Pre-SART resting |
| 3 | **Resting State EEG** | 22 | 64 | 512 Hz | ✅ | dataverse.tdl.org |
| 4 | **BIDS EEG Meditation** | ~20 | 64 | 512 Hz | — | Zenodo |
| 5 | **Lifespan Resting** (ds005385) | 608 | 64 | 500 Hz | — | 5-year follow-up |

### A.8 Language / EEG-to-Text

| # | Dataset | Subjects | Channels | Rate | Notes |
|---|---|---|---|---|---|
| 1 | **Le Petit Prince HK** | 52 | 64 | 500 Hz | Naturalistic fMRI + EEG, Cantonese |
| 2 | **ZuCo** | 12 | 128 | 500 Hz | Eye-tracking + EEG, 21K words |
| 3 | **ChineseEEG** | 20 | 128 | 1000 Hz | Semantic alignment, ds004952 |
| 4 | **ChineseEEG-2** | 12 | 64 | 500-1000 Hz | Reading aloud + passive listening |
| 5 | **Chisco Dataset** | ~20 | 64 | 512 Hz | ds005170 |
| 6 | **Kara-One** | 14 | 64 | 500 Hz | Imagined + vocalized speech |
| 7 | **Inner Speech** | ~10 | 64 | 500 Hz | ds003626 |

### A.9 Visual Stimulus

| # | Dataset | Subjects | Channels | Rate | Notes |
|---|---|---|---|---|---|
| 1 | **THINGS-EEG / THINGS-EEG2** | ~20 | 64-128 | 1000 Hz | Object recognition, figshare |
| 2 | **THINGS-EEG (OpenNeuro ds003825)** | **50** | **128** | 1,024 Hz | RSVP of 22,248 images / 1,854 concepts | OpenNeuro | **~45M** | Dense visual object representation. Same subjects as THINGS-fMRI |
| 2 | **SEED-DV** | ~30 | 62 | 1000 Hz | Video watching |
| 3 | **MNIST Brain Digits** | 1 | 4-14 | varies | Single subject, need interpolation |
| 4 | **EEG-ImageNet** | 6 | 128 | 500 Hz | Image classification |
| 5 | **MAMEM SSVEP** | 10 | 8-14 | 250 Hz | PhysioNet |

### A.10 Auditory Stimulus

| # | Dataset | Subjects | Channels | Rate | Notes |
|---|---|---|---|---|---|
| 1 | **OpenMIIR** | 10 | 64 | 512 Hz | Music imagery, 12 pieces |
| 2 | **NMED-T/H/M/E/RP** | ~30 | 64 | 500 Hz | Stanford, music/audio |
| 3 | **Auditory EEG** | ~20 | 64 | 512 Hz | PhysioNet |
| 4 | **MUSIN-G** | ~20 | 64 | 512 Hz | Music listening, ds003774 |
| 5 | **Affective Music** | ~20 | 64 | 512 Hz | ds002721 |

### A.11 Clinical EEG

| # | Dataset | Subjects | Channels | Rate | LaBram/NeuroLM Pretrained | Notes |
|---|---|---|---|---|---|---|
| 1 | **TUH EEG** | **~60,000+** | 16-36 | 250-512 Hz | ✅ | **Largest open clinical EEG corpus** — 60K+ recordings |
| 2 | **Siena Scalp EEG** | 14 | 32 | 512 Hz | ✅ | PhysioNet |
| 3 | **CHB-MIT** | 22 | 23 | 256 Hz | — | Pediatric seizures |
| 4 | **ADHD Dataset** | ~300 | 32 | 500 Hz | — | NDA |
| 5 | **Epilepsy pre/post-surgical** | ~50 | 64 | 512 Hz | — | HFO markings |
| 6 | **Cognitive Decline** | ~100 | 64 | 200 Hz | — | INSPECDS |
| 7 | **Parkinson's Disease** | ~50 | 64 | 500 Hz | — | ds002778 |
| 8 | **Alzheimer's / FTD** | ~100 | 64 | 500 Hz | — | ds006036 |
| 9 | **Stroke Rehabilitation** | ~100+ | 19-64 | 256-500 Hz | — | Multiple studies |
| 10 | **UCI EEG Alcoholism** | 122 | 64 | 256 Hz | — | Alcoholic vs control |
| 11 | **EPILEPSIAE** (Europe) | **300+** | 19-128 | 256-1024 Hz | — | **Long-term monitoring**, weeks/subject. Restricted. |
| 12 | **TMS-EEG** (aggregated) | ~50-100 | 60-64 | 500-1000 Hz | — | TMS-evoked potentials |
| 13 | **Chinese Hospital Archives** | ~10,000+ est. | 19-21 | 200-500 Hz | — | **Not public** — largest potential upside if access negotiated |
| 14 | **VA/TBI EEG** | ~500+ est. | 19-21 | 200-512 Hz | — | **Not public** — military/TBI/PTSD |
| 15 | **Japanese AIST Sleep** | ~100-200 | 19+ | 200-500 Hz | — | Limited access |
| 16 | **CNEP** (Chinese Neonatal) | ~200+ | 8-19 | 256-500 Hz | — | Neonatal seizure, partially open |

### A.12 Low-Resolution EEG (<19 channels)

These require channel interpolation to reach the 19-channel model input. Massive hour counts but low spatial resolution.

| # | Dataset | Subjects | Channels | Rate | Task | Duration/Subj | Access | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Muse Headband** (various) | ~50–200 | 4 (TP9, AF7, AF8, TP10) | 256 Hz | Meditation, attention, resting | ~30 min | Kaggle, OpenNeuro | Consumer; dry electrode |
| 2 | **MindBigData** | 1 (>1.2M trials) | 1–14 mixed | varies | Digit recognition | 2s/trial | Open | Single subject; need interpolation |
| 3 | **NeuroSky MindWave** | ~30 | 1 (FP1) | 512 Hz | Attention/meditation | ~10 min | Kaggle | Single-channel; extremely limited |
| 4 | **OpenBCI Ganglion** | ~20 | 4 | 256 Hz | Motor imagery, SSVEP | ~30 min | Open (GitHub) | Open-source hardware |
| 5 | **Emotiv EPOC** | ~100 | 14 | 128–256 Hz | Emotion, BCI | ~30 min | Kaggle | Saline; AF+CMS/DRL montage |
| 6 | **EEGNet Motor Imagery** | 109 | 1–14 | varies | Motor imagery | varies | Kaggle | Aggregated from multiple devices |
| 7 | **Steinfeld BCI** | 15 | 3 (Fz, C3, C4) | 250 Hz | Motor imagery | ~60 min | OpenNeuro | Minimal MI montage |
| 8 | **Bonn Seizure** | 5 | 1 | 173.6 Hz | Inter-ictal vs ictal | ~23.6s/file | Open (Bonn) | Classic; 500 files; single-channel |
| 9 | **PhysioNet Sleep-EDF** | 197 | 2 (FPz-Cz, Pz-Oz) | 100 Hz | Sleep staging | ~20h | PhysioNet (open) | 2-channel polysomnography |
| 10 | **Sleep Cassette (EDF)** | 78 | 2 | 100 Hz | Sleep staging | ~8h | PhysioNet (open) | Sleep staging benchmark |
| 11 | **SHHS** (Sleep Heart Health) | 5,000+ | 2–4 | 125 Hz | Sleep staging | ~8h | NSSRR (application) | Largest sleep EEG; epidemiological |
| 12 | **MESA Sleep** | 2,000+ | 4 | 200 Hz | Sleep staging | ~8h | NSSRR (application) | Diverse population; actigraphy |
| 13 | **DREAM** (Monash) | 18 | 3 (F3, F4, O1) | 512 Hz | Dream reports | ~8h | Monash (application) | Dream EEG; awakening reports |
| 14 | **Neonatal EEG (Shellhaas)** | 60 | 8–12 | 256 Hz | Seizure detection | — | Application | NICU; reduced montage |
| 15 | **Helsinki Neonatal** | 39 | 8 | 256 Hz | Seizure detection | — | Open (Zenodo) | 8-channel neonatal EEG |
| 16 | **Intra-op EEG** (anesthesia) | ~100 | 4–8 bifrontal | 128 Hz | Depth of anesthesia | — | Application | Routine monitoring; huge hospital archives |

> **Low-res token summary**: ~6,200 subjects, ~58,700 hours, ~320M raw tokens (channel-count dependent). Requires interpolation to 19ch.

### A.13 BCI Benchmark Datasets (MOABB)

The Mother of All BCI Benchmarks (MOABB) standardizes 60+ BCI datasets across motor imagery, P300, SSVEP, CVEP, and imagined speech. Below are the datasets **not already listed** in sections A.3–A.7 above, prioritized by size and paradigm diversity.

#### Motor Imagery (MOABB)

| # | Dataset | Subjects | Channels | Rate | Classes | Sessions | Trials | Access | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Stieger2021** | **62** | 64 | 1,000 Hz | 4 (L/R/hand/feet) | 7–11 | **~250,000** | Open (G-Node) | MBSR intervention + BCI training. Largest single MI dataset |
| 2 | **Cho2017** | **52** | 64 | 512 Hz | 2 (L/R hand) | 5–6 | ~9,800 | Open (GigaDB) | Biosemi ActiveTwo; simultaneous EMG |
| 3 | **Lee2019_MI** | **54** | 62 | 1,000 Hz | 2 (L/R hand) | 2 | ~11,000 | Open (GigaDB) | BrainAmp; simultaneous EMG |
| 4 | **Dreyer2023** (A+B+C) | **87** | 27 | 512 Hz | 2 (L/R hand) | 1 | ~20,880 | Open | Multi-center; 60/21/6 subjects |
| 5 | **Yang2025** | **62** | 59 | 1,000 Hz | 3 (L/R/feet) | 3 | ~39,600 | Open | 7.5s trials; 3 sessions |
| 6 | **Zuo2025** | **30** | 30 | 500 Hz | 2 (L/R hand) | 5 | ~15,000 | Open | 5-session longitudinal |
| 7 | **Ma2020** | **25** | 62 | 200 Hz | 2 (L/R hand) | **15** | ~15,000 | Open | 15-session longitudinal; within-subject learning |
| 8 | **Jeong2020** | **25** | 60 | 2,500 Hz | **11** (multi-limb) | 3 | ~41,250 | Open | 11-class MI; highest class diversity |
| 9 | **Forenzo2023** | **25** | 64 | 1,000 Hz | 2 (L/R hand) | 5 | ~1,875 | Open | 60s continuous; 5 sessions |
| 10 | **Chang2025** | **28** | 59 | 1,000 Hz | 3 (L/R/feet) | 4 | ~13,440 | Open | 4 sessions |
| 11 | **Gao2026** | **22** | 32 | 1,000 Hz | **10** | 2 | ~16,800 | Open | 10-class MI |
| 12 | **TrianaGuzman2024** | **32** | 17 | 250 Hz | 4 | 1 | ~7,680 | Open | 15s trials |
| 13 | **Zhou2020** | **20** | 26 | 500 Hz | 4 (L/R/hand/feet) | **7** | ~33,600 | Open | 7 sessions; 6 runs/session |
| 14 | **Rozado2015** | **30** | 32 | 512 Hz | 2 (L/R hand) | 1 | ~1,550 | Open | 6s trials |
| 15 | **Brandl2020** | **16** | 63 | 1,000 Hz | 2 (L/R hand) | 1 | ~8,064 | Open | 7 runs |
| 16 | **Wairagkar2018** | **14** | 19 | 1,024 Hz | 3 (L/R/idle) | 1 | ~1,665 | Open | 19ch; 6s trials |
| 17 | **GrosseWentrup2009** | **10** | 128 | 500 Hz | 2 | 1 | ~3,000 | Open | 128ch; 7s trials |
| 18 | **Ofner2017** | **15** | 61 | 512 Hz | **7** (hand classes) | 1 | ~63,000 | Open | 7-class hand motor; 10 runs |
| 19 | **Weibo2014** | **10** | 60 | 200 Hz | **7** | 1 | ~5,600 | Open | 7-class MI |
| 20 | **BNCI2014_001** | **9** | 22 | 250 Hz | 4 (L/R/feet/tongue) | 2 | ~6,220 | Open | BCI Comp IV dataset IIa; gold standard |
| 21 | **BNCI2015_001** | **12** | 13 | 512 Hz | 2 (L/R hand) | 3 | ~14,400 | Open | 3 sessions |
| 22 | **BNCI2014_004** | **9** | 3 | 250 Hz | 2 (L/R hand) | 5 | ~32,400 | Open | 3-ch; 360 trials/class/session |
| 23 | **Shin2017A** | **29** | 30 | 200 Hz | 2 (L/R hand) | 3 | ~5,220 | Open | 3 sessions; 10s trials |
| 24 | **Shin2017B** | **29** | 30 | 200 Hz | 2 (L/R hand) | 3 | ~5,220 | Open | 3 sessions |
| 25 | **Liu2024** | **50** | 29 | 500 Hz | 2 (L/R hand) | 1 | ~2,000 | Open | 4s trials |
| 26 | **Kumar2024** | **18** | 22 | 512 Hz | 2 (L/R hand) | 6 | ~7,156 | Open | 6 sessions |
| 27 | **Yi2025** | **18** | 62 | 250 Hz | **8** | 1 | ~5,760 | Open | 8-class MI |
| 28 | **Liu2025** | **27** | 64 | 1,000 Hz | 2 (L/R hand) | 3 | ~8,640 | Open | 4 runs/session |
| 29 | **GuttmannFlury2025_MI** | **31** | 62 | 1,000 Hz | 2 (L/R hand) | 3 | ~2,520 | Open | 7.5s trials |
| 30 | **HefmiIch2025** | **37** | 32 | 256 Hz | 2 (L/R hand) | 3 | ~3,330 | Open | 27s trials |
| 31 | **AguileraRodriguez2025** | **15** | 24 | 500 Hz | 4 | 1 | ~1,800 | Open | 4s trials |
| 32 | **BNCI2020_001** | **15** | 11–64 | 256 Hz | 3 | 3 | ~7,200 | Open | 3-class; variable channels |
| 33 | **BNCI2024_001** | **20** | 64 | 512 Hz | **10** | 1 | varies | Open | 10-class |
| 34 | **BNCI2025_001** | **20** | 64 | 500 Hz | **16** | 1 | varies | Open | 16-class |
| 35 | **Tavakolan2017** | **12** | 32 | 1,000 Hz | 3 (L/R/idle) | 4 | ~2,880 | Open | 4 sessions |
| 36 | **Kaya2018** | **7** | 19 | 200 Hz | 3 | 3 | ~16,126 | Open | 1s trials |
| 37 | **BCIComp2020UpperLimb** | **15** | 60 | 250 Hz | 3 (upper limb) | 3 | ~6,750 | Open | Upper limb (not hand); 3 sessions |

#### P300 / ERP (MOABB)

| # | Dataset | Subjects | Channels | Rate | Paradigm | Sessions | Access | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Lee2019_ERP** | **54** | 62 | 1,000 Hz | P300 speller | 2 | Open | Same subjects as Lee2019_MI |
| 2 | **BNCI2015_009** | **21** | 62 | 1,000 Hz | P300 speller | 2 | Open | Large trial count |
| 3 | **Mainsah2025_Q** | **36** | 32 | 256 Hz | P300 | 1 | Open | Largest single P300 |
| 4 | **Mainsah2025_R** | **20** | 32 | 256 Hz | P300 | 2 | Open | 2 sessions |
| 5 | **GuttmannFlury2025_P300** | **31** | 66 | 1,000 Hz | P300 | 1–3 | Open | Multi-session |
| 6 | **Lee2021Mobile_ERP** | **24** | 32 | 500 Hz | Mobile P300 | 5 | Open | Mobile/walking context |
| 7 | **Zheng2020** | **14** | 62 | 1,000 Hz | P300 speller | 2 | Open | 168 targets/session |
| 8 | **EPFLP300** | **8** | 32 | **2,048 Hz** | P300 speller | 4 | Open | Very high sampling rate |
| 9 | **Sosulski2019** | **13** | 31 | 1,000 Hz | P300 matrix | 1 | Open | 7,500 NT / 1,500 T per subject |
| 10 | **Cattan2019_VR** | **21** | 16 | 512 Hz | P300 in VR | 2 | Open | **VR environment** — unique paradigm |
| 11 | **BNCI2015_010** | **12** | 63 | 1,000 Hz | P300 | 1 | Open | Very large trial count |
| 12 | **BNCI2014_009** | **10** | 16 | 256 Hz | P300 speller | 3 | Open | 3 sessions |
| 13 | **BNCI2015_003** | **10** | 8 | 256 Hz | P300 | 1 | Open | 8-channel |
| 14 | **Kojima2024B** | **15** | 64 | 1,000 Hz | P300 | 1 | Open | 2,160 NT / 720 T |
| 15 | **Huebner2017/2018** | **13/12** | 31 | 1,000 Hz | P300 | 3 | Open | 3 sessions each |
| 16 | **Simoes2020** | **15** | 8 | 250 Hz | P300 | **7** | Open | 7 sessions; longitudinal |
| 17 | **Speier2017** | **10** | 32 | 256 Hz | P300 | 2 | Open | 2 sessions |
| 18 | **Chailloux2020** | **19** | 8 | 256 Hz | P300 | 3 | Open | 3 sessions |
| 19 | **RomaniBF2025ERP** | **22** | 8 | 250 Hz | P300 | 3 | Open | 3 sessions |
| 20 | **BNCI2016_002** | **15** | 69 | 200 Hz | Brake/EMG ERP | 1 | Open | Driving brake detection |
| 21 | **BNCI2020_002** | **18** | 31 | 250 Hz | P300 | 1 | Open | 16s trials |
| 22 | **BCIComp2020WalkingERP** | **15** | 46 | 100 Hz | Walking P300 | 1 | Open | **Walking context** — mobility BCI |
| 23 | **Mainsah2025_A–P,S1,S2** | 8–36 each | 16–32 | 256 Hz | Various P300 | 1–8 | Open | 18 sub-datasets; massive aggregate |
| 24 | **Lee2024_TV/DL/EL/BS/AC** | 10–30 | 25–31 | 500 Hz | P300 variants | 1 | Open | 5 variants; TV, dual-task, etc. |
| 25 | **Zhang2025** | **15** | 57 | 1,000 Hz | P300 | 4 | Open | 4 sessions |
| 26 | **Kaneshiro2015** | **10** | **124** | 62.5 Hz | 6-class RSVP | 1 | Open | 124ch; 5,184 trials |

#### SSVEP (MOABB)

| # | Dataset | Subjects | Channels | Rate | Classes | Sessions | Access | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Liu2020BETA** | **70** | 64 | 250 Hz | **40** | 1 | Open | **Largest SSVEP dataset**. 40 frequencies/phases |
| 2 | **Liu2022EldBETA** | **100** | 64 | 1,000 Hz | 9 | **7** | Open | **Largest SSVEP subject count**. Elderly focus; 7 sessions |
| 3 | **Lee2019_SSVEP** | **54** | 62 | 1,000 Hz | 4 | 2 | Open | Same subjects as Lee2019_MI |
| 4 | **Kim2025BetaRange** | **40** | 33 | 1,024 Hz | **40** | **6** | Open | 40-class; beta frequency range; 6 sessions |
| 5 | **Dong2023** | **59** | 8 | 250 Hz | **40** | 1 | Open | 40-class; 8-channel wearable |
| 6 | **Wang2016** | **34** | 64 | 250 Hz | **40** | 1 | Open | 40-class; 6 trials/class |
| 7 | **Han2024Fatigue** | **24** | 64 | 1,000 Hz | **32** | 2 | Open | Fatigue study; 32-class |
| 8 | **GuttmannFlury2025_SSVEP** | **31** | 66 | 1,000 Hz | 4 | 1–3 | Open | Multi-session |
| 9 | **Lee2021Mobile_SSVEP** | **24** | 32 | 500 Hz | 3 | 4–5 | Open | Mobile context |
| 10 | **MAMEM1** | **10** | **256** | 250 Hz | 5 | 1 | Open | **256 channels** (EGI); SSVEP |
| 11 | **MAMEM2** | **10** | **256** | 250 Hz | 5 | 1 | Open | **256 channels** (EGI); SSVEP |
| 12 | **MAMEM3** | **10** | 14 | 128 Hz | 4 | 1 | Open | Dry electrodes; 14ch |
| 13 | **Nakanishi2015** | **9** | 8 | 256 Hz | **12** | 1 | Open | 12-class; 8ch |
| 14 | **Chen2017SingleFlicker** | **12** | 32 | 512/2,048 Hz | 4 | 2 | Open | Single flicker; 2 sessions |
| 15 | **Kalunga2016** | **12** | 8 | 256 Hz | 4 | 1 | Open | 8ch; 2s trials |
| 16 | **Wang2021Combined** | **8** | 32 | 1,000 Hz | 4 | 1 | Open | Combined stimuli |

#### CVEP / Code-Modulated VEP (MOABB)

| # | Dataset | Subjects | Channels | Rate | Classes | Sessions | Access | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Thielen2021** | **30** | 8 | 512 Hz | **20** | 1 | Open | **Largest CVEP**. 20-class Gold codes; 8ch |
| 2 | **MartinezCagigal2023Pary** | **16** | 16 | 256 Hz | **16** | 5 | Open | p-ary m-sequence; 5 sessions |
| 3 | **MartinezCagigal2023Checker** | **16** | 16 | 256 Hz | **16** | 8 | Open | Checkerboard m-sequence; 8 sessions |
| 4 | **Thielen2015** | **12** | 64 | 2,048 Hz | **36** | 1 | Open | 36-class; 64ch; very high rate |
| 5 | **CastillosCVEP40/100** | **12** | 32 | 500 Hz | 4 | 1 | Open | m-sequence; 40Hz/100Hz |
| 6 | **CastillosBurstVEP40/100** | **12** | 32 | 500 Hz | 4 | 1 | Open | Burst-CVEP; novel stimulation |

#### Simultaneous EEG + fNIRS

| # | Dataset | Subjects | EEG Ch | fNIRS Ch | Rate | Task | Sessions | Access | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Shin2017A** | **29** | 30 | 30 | 200 Hz | Motor imagery | 3 | Open (TU Berlin) | **Simultaneous EEG+fNIRS** — cross-modal validation |
| 2 | **Shin2017B** | **29** | 30 | 30 | 200 Hz | Mental arithmetic | 3 | Open (TU Berlin) | **Simultaneous EEG+fNIRS** — cognitive load |

> Shin2017 is the largest open simultaneous EEG+fNIRS dataset. Both motor imagery and mental arithmetic conditions. fNIRS provides hemodynamic correlate at ~6–10s resolution. Useful for EEG→slow-signal bridge experiments (like EEG→fMRI but smaller scale).

#### Imagined Speech / Silent Communication (MOABB)

| # | Dataset | Subjects | Channels | Rate | Classes | Sessions | Access | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Nieto2022** | **10** | **128** | 1,024 Hz | 4 (vowels) | 3 | Open | **128ch**; 3 sessions; vowel imagination |
| 2 | **Nguyen2017_V** | **8** | 64 | 256 Hz | 3 (vowels) | 1 | Open | Vowel imagination |
| 3 | **Nguyen2017_S** | **6** | 64 | 256 Hz | 3 (syllables) | 1 | Open | Syllable imagination |
| 4 | **Nguyen2017_L** | **6** | 64 | 256 Hz | 2 (words) | 1 | Open | Word imagination |
| 5 | **Nguyen2017_SL** | **6** | 64 | 256 Hz | 2 (syllable+word) | 1 | Open | Mixed |
| 6 | **Pressel2016** | **15** | 6 | 1,024 Hz | **11** | 1 | Open | 6ch; 11-class silent speech |

#### Mental Workload / Vigilance / Resting-State Task

| # | Dataset | Subjects | Channels | Rate | Task | Sessions | Access | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Cattan2019_PHMD** | **12** | 16 | 512 Hz | Mental workload / vigilance | 1 | Open | 60s blocks; 2-class workload |
| 2 | **Hinss2021** | **15** | 62 | 250 Hz | 4 resting-state conditions | 1 | Open | Resting with different instructions |
| 3 | **Rodrigues2017** | **20** | 16 | 512 Hz | Eyes open/closed / task | 1 | Open | Resting + task blocks |

> **MOABB aggregate token estimate**: ~2,500+ subjects across all paradigms, ~5M+ trials. Many datasets are small (n<15) but the aggregate is substantial. **Key value**: standardized preprocessing, consistent trial structures, and paradigm diversity (MI/P300/SSVEP/CVEP/speech/workload) that complements our existing clinical/task-heavy catalog.

---

## A.14 MEG (Magnetoencephalography) Datasets

MEG measures magnetic fields from neural currents — same postsynaptic sources as EEG, but with **better spatial localization** and **no volume conduction distortion**. Because EEG and MEG share the same electromagnetic forward model (Maxwell equations), MEG→latent-space alignment is more principled than EEG→fMRI (no hemodynamic coupling required). However, MEG hardware costs limit data scale.

| # | Dataset | Subjects | Channels | Rate | Task | Access | Tokens (est) | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | **Cam-CAN MEG** | **~700** | 306 (Elekta Neuromag) | 1,000 Hz | Resting + sensorimotor + auditory + visual | Open (reg) | ~2.5M | Full lifespan 18–88. Largest open MEG dataset. Same subjects as Cam-CAN fMRI/structural |
| 2 | **HCP MEP (Young Adult)** | **~95** | 248 mag + 23 ref | 2,039 Hz | Resting (4 runs) + motor + WM + language + story/math | Open (ConnectomeDB) | ~0.5M | Gold-standard MEG. Same 95 subjects as HCP-YA 7T subset |
| 3 | **HCP Development MEG** | **~50–100** | 248 mag + 23 ref | 2,039 Hz | Resting + tasks (shorter protocol) | Open (ConnectomeDB) | ~0.2M | Pediatric/adolescent MEG |
| 4 | **ds000117 (OpenNeuro)** | **16** | 306 (Elekta) | 1,100 Hz | Face perception (visual) | Open | ~0.05M | Classic multimodal: MEG + fMRI + EEG on same subjects |
| 5 | **MNE Sample** | **1** | 306 | ~600 Hz | Auditory (left/right tone) | Built into MNE-Python | — | Tutorial dataset; not for training |
| 6 | **OpenNeuro MEG aggregate** | **~200** | Various | 1,000–2,000 Hz | Mixed (auditory, visual, language, resting) | Open | ~0.5M | ~15+ BIDS-MEG datasets across labs |

> **MEG alignment physics**: EEG and MEG are generated by the same primary current sources (postsynaptic potentials in pyramidal cells). The forward operators differ (volume conduction vs. Biot-Savart), but the latent neural state z(t) generates both. A shared latent dynamics model with modality-specific forward/projector heads is the natural architecture. This avoids the HRF convolution ambiguity of EEG→fMRI alignment.
>
> **Total MEG subjects**: ~1,000–1,100 open. Modest vs. EEG/fMRI, but the physics alignment is cleaner.

---

## A.15 fNIRS (functional Near-Infrared Spectroscopy) Datasets

fNIRS measures oxy-/deoxy-hemoglobin concentration changes — a hemodynamic signal like fMRI but with **portable hardware** and **sparse spatial sampling** (typically 10–50 optodes). The temporal resolution is ~1–2 Hz (HRF-limited), and the spatial resolution is ~1–3 cm. fNIRS is theoretically alignable to the same latent space via the same HRF bridge as fMRI, but the extremely sparse spatial coverage and small dataset sizes make it a low-priority modality.

| # | Dataset | Subjects | Optodes | Rate | Task | Access | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Shin2017A** | **29** | 30 | 10 Hz | Motor imagery | Open (TU Berlin) | **Simultaneous EEG+fNIRS** (also in §A.13) |
| 2 | **Shin2017B** | **29** | 30 | 10 Hz | Mental arithmetic | Open (TU Berlin) | **Simultaneous EEG+fNIRS** |
| 3 | **OpenFNIRS aggregate** | **~200** total | 8–52 | 6–10 Hz | Mixed (motor, resting, auditory, visual, clinical) | Open (openfnirs.org) | ~20 small datasets (<30 subjects each). Sparse spatial coverage |
| 4 | **Liu2020BETA-fNIRS** | N/A | — | — | — | — | No large open fNIRS SSVEP dataset found |
| 5 | **fNIRS during walking (various)** | **~50** aggregate | 8–16 | 10 Hz | Motor / gait / mobility | Open | Multiple small studies |

> **fNIRS verdict**: Data volume is insufficient for latent-space alignment training. ~250 subjects total across all open fNIRS datasets, with sparse and inconsistent spatial coverage. **Recommended use**: (a) validation-only for the HRF bridge (Shin2017 provides EEG+fNIRS ground truth at much lower cost than EEG+fMRI), and (b) future real-time BCI deployment where fNIRS portability matters but training uses EEG/fMRI. Do not prioritize fNIRS data acquisition.

---

## B. ECoG / sEEG / iEEG Datasets

### B.1 DANDI Archives

| # | DANDI ID | Name | Subjects | Electrodes | Task | Tokens (est) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **000055** | AJILE12 | 12 | 100-256 ch ECoG | Multi-day motor + speech | ~120M | 845 GB, NWB |
| 2 | **000623** | iEEG+fMRI movie | 10 | ~100 ch sEEG | Movie watching | ~20M | **Simultaneous** iEEG-fMRI |
| 3 | **000574** | Scalp+iEEG verbal WM | 9 | 19 scalp + ~80 sEEG | Verbal WM | ~15M | **Simultaneous** scalp+iEEG |
| 4 | **000465/000554** | PtNRGrids | ~30 est. | ~1024 ch ultra-high-density | Various | ~60M | 129 GB |
| 5 | **000673** | Hippocampal WM PAC | **36** | sEEG depth + single units | Working memory | ~30M | Rutishauser lab, Nature 2024 |
| 6 | **000397** | Human Neuropixels | 3 | 384 ch Neuropixels | Sensory/motor | ~5M | Acute intraoperative |
| 7 | **000950** | FALCON H2 Handwriting | 1 | 96 ch Utah array | BCI handwriting | ~2M | BrainGate2, intracortical |
| 8 | **000954** | FALCON H1 Reach/Grasp | 1 | Utah array | 7-DoF reach+grasp | ~1M | BrainGate2 |
| 9 | **001211** | Mouse iEEG+widefield | 9 (mouse) | — | Neurovascular IRF | — | **Not human** |
| 10 | **001543** | Mouse iEEG+widefield | 19 (mouse) | — | Neurovascular IRF | — | **Not human** |

### B.2 OpenNeuro iEEG Datasets

| # | ID | Name | Subjects | Type | Task | Tokens (est) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **ds004752** | Scalp+iEEG Zurich | **15** | sEEG + scalp EEG | Verbal WM | ~15M | **DOUBLES cross-modal alignment data** |
| 2 | **ds005574** | Podcast ECoG | 9 | High-density ECoG | Natural language comprehension | ~20M | **Only naturalistic language ECoG** |
| 3 | **ds004194** | Visual ECoG | 14 | ECoG grids | Visual stimulation + pRF | ~25M | Multi-site (NYU + Utrecht) |
| 4 | **ds006519** | Romanian iEEG motor | 21 | sEEG | Cortical stimulation (motor) | ~7M | Bucharest — new geographic center |
| 5 | **ds005169** | Romanian iEEG visual | 20 | sEEG | Cortical stimulation (visual) | ~7M | ~17 overlap with ds006519 |
| 6 | **ds004703** | sEEG natural speech | 10 | sEEG depth | Passive speech listening | ~10M | |
| 7 | **ds005670** | Chinese sEEG resting | 2 | sEEG depth | Resting state | ~3M | First Chinese sEEG on OpenNeuro |
| 8 | **ds004944** | Intraop ECoG | ~5 | ECoG strips | Motor during awake craniotomy | ~2M | BCI2000 |
| 9 | **ds004993** | WIRED ICM | 3 | sEEG | Workshop demo | ~3M | Pediatric (Dell Children's, Austin) |

### B.3 Other ECoG/sEEG Sources

| # | Source | Name | Subjects | Type | Access | Tokens (est) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **IEEG.org** | MNI, Grenoble, etc. | ~500+ aggregate | Mixed ECoG/sEEG | Open (portal) | ~1.5B | Major European+US repository |
| 2 | **CRCNS** | Freiburg, etc. | ~50+ | ECoG strip/grid | Open | ~100M | |
| 3 | **EPILEPSIAE** | European DB | 275+ | Mixed | Restricted | ~200M | Multi-center, long-term |
| 4 | **GitHub** | iSEED (Chinese emotion) | >10 | sEEG | Request (junjie.wang@rmlab.cn) | ~5M | sEEG-based emotion |
| 5 | **GitHub** | SingleWordProductionDutch | ~10 | ECoG | Partial | ~10M | Dutch language ECoG |
| 6 | **NEMAR** | CerebroVoice | TBD | sEEG | Open (via NEMAR) | ~5M | Bilingual sEEG speech |
| 7 | **Paper** | SMA iEEG Speech | >50 | sEEG | Unclear | ~25M | "Large-scale" — verify access |

---

## C. fMRI Datasets

### C.1 Major fMRI Repositories

| # | Dataset | Subjects | Sessions | Task | Access | Tokens (ROI-level) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **UK Biobank** | ~43,000 | 86,000 scans | Resting + task + movie | Application | ~20M | **NeuroSTORM training data** |
| 2 | **ABCD Study** | ~11,875 | 3+ timepoints | Resting + tasks | NDA DUC | ~5M | Adolescent, longitudinal |
| 3 | **HCP Young Adult** | ~1,200 | 4 scans each | Resting + 7 tasks | Open (ConnectomeDB) | ~1M | Gold standard |
| 4 | **HCP Development** | ~650 | 2 scans each | Resting + tasks | Open | ~0.5M | Ages 5-21 |
| 5 | **HCP Aging** | ~1,200 | 2 scans each | Resting + tasks | Open | ~0.5M | Ages 36-100+ |
| 6 | **HBCD** (HEALthy Brain) | ~7,500 | planned ongoing | Resting + tasks | Application | ~3M | Pregnant women + infants |
| 7 | **eNKI Rockland** | ~1,000 | 1-2 scans | Resting | Open | ~0.5M | Enhanced Nathan Kline Institute |
| 8 | **NeuroSTORM pretraining** | 50,000+ | 28.65M frames | Resting + task | Model weights open | — | UKB + ABCD + HCP + others |

### C.2 Clinical & Biomarker fMRI

| # | Dataset | Subjects | Sessions | Task | Access | Tokens (ROI-level) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **ADNI** | ~2,500+ | Longitudinal | Resting + task (ADNI-GO/3) + PET | LONI application | ~1M | Alzheimer's biomarker gold standard. 4,000+ publications |
| 2 | **PPMI** | 1,500+ (→4,000) | Longitudinal | fMRI + DATScan + EEG | Open (reg) | ~0.5M | Parkinson's largest biomarker study. LRRK2/GBA cohorts |
| 3 | **OASIS-3** | 1,378 | 2,842 MR sessions | rs-fMRI + Tau PET + ASL | Open (DUA) | ~1.5M | Aging + Alzheimer's. AV1451 Tau in 451 sessions |
| 4 | **OASIS-4** | 663 | — | MR + clinical + CSF | Open (DUA) | — | Memory disorders cohort. Ages 21–94 |
| 5 | **AIBL** | 3,045 | 10,494 person-years | fMRI + PET + structural | Restricted (collab) | ~1M | Australian aging/AD. 15+ year longitudinal |
| 6 | **NUSDAST** | 275 | — | fMRI + structural + GWAS | Open (NITRC) | ~0.1M | Schizophrenia. Manual segmentations + neurocognitive battery |
| 7 | **COBRE** | 146 | — | rs-fMRI + DTI + structural | Open (OpenNeuro ds000030) | ~0.1M | Schizophrenia from Mind Research Network |
| 8 | **REST-meta-MDD** | ~2,300+ | Multi-site | rs-fMRI | Restricted (consortium) | ~1M | Chinese consortium. Largest MDD rs-fMRI meta-analysis |
| 9 | **Brain-CODE** (Ontario Brain Institute) | **~5,000+** | Longitudinal | Multi-modal (clinical, imaging, molecular, genetics) | Restricted (application) | ~2M | Cross-disorder: neurodevelopmental, CP, epilepsy, depression, neurodegeneration, concussion |
| 10 | **PAIN Repository** | **~2,000+** | — | Structural + Diffusion + fMRI | Restricted (application) | ~1M | Chronic somatic and visceral pain. Multi-site |

### C.3 International / Multi-Site fMRI Collections

| # | Dataset | Subjects | Sessions | Task | Access | Tokens (ROI-level) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **IMAGEN** | ~2,000 | Longitudinal | Task fMRI (MID, Stop Signal, Face Go/NoGo) + genetics | Restricted (DSA) | ~1M | European adolescent brain study. Multi-site |
| 2 | **ALSPAC** | ~5,000 scanned | — | rs-fMRI + task + DTI | Restricted (proposal) | ~1.5M | UK birth cohort (14,000+ total). Scanned at ~24y |
| 3 | **Generation R** | ~3,000–4,000 MRI | — | rs-fMRI + structural | Restricted (collab) | ~1M | Dutch population cohort from fetal life. Ages ~10, ~14 |
| 4 | **Cam-CAN** | ~2,000+ | — | Task + rs-fMRI + MEG | Open (reg) | ~1.5M | Full adult lifespan 18–88. Uniform protocols |
| 5 | **PNC** | ~1,600 imaged | — | Task + rs-fMRI + DTI + genetics | dbGaP restricted | ~1M | Philadelphia. Ages 8–21. Diverse clinical presentations |
| 6 | **CHCP** (Chinese HCP) | ~500+ | — | rfMRI + tfMRI + 7T pilot | Mixed | ~0.5M | Chinese counterpart to HCP. Population-specific protocols |
| 7 | **AOMIC** (Amsterdam Open MRI) | ~1,000+ | — | T1w + rfMRI + tfMRI | Open | ~0.8M | Dutch multi-sample initiative |
| 8 | **CoRR** | 1,629 | 5,093 rs-fMRI | Resting-state + DTI + ASL | Open (NITRC) | ~2M | 33 datasets. Benchmark test-retest reliability |
| 9 | **1000 FCP Classic** | ~1,400+ | — | rs-fMRI + structural | Open (NITRC) | ~1.5M | First large-scale rs-fMRI sharing (Biswal 2010). 35+ sites |
| 10 | **ABIDE I & II** | ~2,100 total | — | rs-fMRI + structural (+ DWI II) | Open (NITRC) | ~2M | Largest autism neuroimaging dataset |
| 11 | **ADHD-200** | ~800–1,000 | — | rs-fMRI + structural | Open (NITRC) | ~0.8M | Large-scale ADHD. Used for prediction competitions |
| 12 | **GSP** (Brain Genomics Superstruct) | ~1,570 | — | rs-fMRI + DTI + behavior | Restricted (Harvard) | ~1M | Harvard young adults. Protocol adapted by NKI/Rockland |
| 13 | **NKI/Rockland (Original)** | ~300+ | — | rs-fMRI + DTI + phenotyping | Open (CC BY-NC) | ~0.3M | Distinct from eNKI. Prospective with 2–8 week lag |
| 14 | **CoRR Chinese Sites** | 100s | — | rs-fMRI + DTI + structural | Open (INDI) | ~0.5M | BNU, HNU, IPCAS, SWU, JHNU, XHCUMS. Test-retest designs |
| 15 | **CoRR European Sites** | 100s | — | rs-fMRI + DTI + structural | Open (INDI) | ~0.3M | BMB Berlin, LMU Munich, MPG Leipzig, UM Montreal |
| 16 | **Beijing Enhanced** | 180 | — | rs-fMRI + DTI + MPRAGE | Open (CC BY-NC) | ~0.2M | IQ scores for subset (n=55). Do not combine with Beijing_Zang |
| 17 | **Beijing EOEC** | 48 | — | rs-fMRI (eyes open/closed) + DTI | Open (CC BY-NC) | ~0.05M | Unique EO/EC counterbalanced design |
| 18 | **SRPBS / Brain/MINDS** | 1,000s | — | rs-fMRI + task + DTI | Mixed | ~1M | Japanese national brain project |
| 19 | **fBIRN** | 100s+ | Multi-site | Task + rs-fMRI + phantom QA | Mixed | ~0.3M | Schizophrenia, bipolar, ADHD. Traveling-subject reliability |
| 20 | **PING** (Pediatric Imaging, Neurocognition, Genetics) | **1,493** | — | T1, T2, DTI, rs-fMRI | Open | ~2.5M | Developmental; ages 3–20. UC San Diego |
| 21 | **SALD** (Southwest University Adult Lifespan) | **494** | — | T1, rs-fMRI, task fMRI (subset) | Open | ~0.8M | **Chinese lifespan dataset**. Fills demographic gap |
| 22 | **Cimbi Database** | **~2,000** | — | Serotonin PET + MRI + genetics + neuropsych | Restricted (application) | ~1.5M | Copenhagen. Unique serotonin system focus. Biobank |
| 23 | **PTBP** (Pediatric Template of Brain Perfusion) | **210** | — | T1, dMRI, rs-fMRI, ASL | Open (figshare) | ~0.4M | Pediatric perfusion template |
| 24 | **BRAINS** (Brain Images of Normal Subjects) | **~500+** | — | T1, structural | Open | — | Lifespan normative structural. UK |
| 25 | **Age-ility** | **131** | — | T1, dMRI, rs-fMRI, **EEG** | Open | ~0.3M | **Multi-modal with EEG**. Australia |
| 26 | **Kirby 21** | **21** | — | T1, DTI, FLAIR, ASL, VASO, rs-fMRI | Open | ~0.1M | **Test-retest** (multi-session). Reliability benchmark |
| 27 | **NACC** (National Alzheimer's Coordinating Center) | **~5,000+** | Longitudinal | T1, T2, DTI, FLAIR | Restricted | ~2M | Largest Alzheimer's longitudinal. Multi-site US |
| 28 | **OMEGA** (Open MEG Archive) | **~200+** | — | MEG + anatomical MRI | Open (McGill) | ~0.5M | **MEG + MRI**. Montreal. Growing archive |
| 29 | **B-SNIP** (Bipolar-Schizophrenia Network) | **~1,000+** | — | T1, T2, DTI, rs-fMRI, task fMRI | Open (NiDB) | ~1.5M | Bipolar + schizophrenia + controls. Intermediate phenotypes |
| 30 | **MID** (Monetary Incentive Delay) | **~500+** | — | Task fMRI (reward processing) | Open (NiDB) | ~0.5M | Reward/anticipation task. Part of NiDB aggregate |

### C.4 Naturalistic & Task fMRI

| # | Dataset | Subjects | Sessions | Task | Access | Tokens (ROI-level) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Narratives** | **345** | — | Story listening (naturalistic) | Open (CC0) | ~0.5M | Sam Nastase et al. Large-scale language comprehension |
| 2 | **NNDb** (Naturalistic Neuroimaging DB) | ~180+ | — | Dialogues / conversations | Open | ~0.3M | Nastase lab. Interactive dialogue fMRI |
| 3 | **studyforrest** | ~20 | Multi-session | Movie watching (Forrest Gump) | Open (MIT) | ~0.1M | Rich annotations + eye tracking + physio |
| 4 | **Courtois Neuromod (CNeuroMod)** | 6 | **100+ hrs/subj** | Movies + video games + HCP tasks | Open (reg) | ~0.2M | Dense-sampling. movie10 + hcptrt + gaming |
| 5 | **Budapest Movie** | ~30 | — | Movie watching | Open (OpenNeuro ds001769) | ~0.1M | Hyperalignment / inter-subject correlation benchmark |
| 6 | **Raiders** | ~11 | — | Movie watching (Raiders of the Lost Ark) | Open | ~0.05M | Haxby Lab. Classic naturalistic narrative |
| 7 | **HCP 7T Movie / Retinotopy** | ~184 | — | 7T retinotopy + movie clips | Open (reg) | ~0.5M | Separate from 3T HCP. High-res 7T |
| 8 | **BOLD5000** | 4 | — | Event-related visual object fMRI | Open | ~0.1M | 5,000 ImageNet/COCO/Scene images. Vision benchmark |
| 9 | **NSD** (Natural Scenes Dataset) | 8 | 30–40 sessions | 7T visual perception + memory | Open (DAA) | ~0.3M | 1.8 mm iso, 1.6s TR. Kendrick Kay. NSD-Imagery + Synthetic extensions |
| 10 | **MyConnectome** | 1 | ~100+ sessions | Dense-sampling rs+task+MEG | Open | ~0.05M | Russ Poldrack. 18 months. Landmark single-subject study |
| 11 | **MSC** (Midnight Scan Club) | 10 | 10+ sessions each | Task + rs-fMRI dense-sampling | Open (WashU) | ~0.1M | Precision functional mapping. Individual differences |

### C.5 Developmental / Lifespan fMRI

| # | Dataset | Subjects | Sessions | Task | Access | Tokens (ROI-level) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Baby Connectome Project** | ~500 | Longitudinal | rs-fMRI + naturalistic (infant movies) + DTI | Restricted (HCP/NDA) | ~0.3M | 0–5 years. HCP lifespan companion |
| 2 | **Dallas Lifespan Brain Study** | ~300+ | — | fMRI + DTI + cognitive tests | Open (NITRC) | ~0.3M | Lifespan. Antecedents of cognitive decline |
| 3 | **Neurocognitive Aging** (Spreng) | ~150 | — | Multi-echo rs-fMRI + T1w | Open | ~0.2M | Aging with multi-echo acquisition |
| 4 | **IXI** | ~600 | — | Structural + DTI (limited fMRI) | Open (CC BY-SA) | — | Cross-scanner validation (3 London hospitals) |
| 5 | **Drakenstein Child Health** | **239** | 1 session | T1 + T2 + **rs-fMRI** + DTI + MRS | Open (upon request) | ~0.3M | **SOUTH AFRICA** — only large open pediatric MRI from sub-Saharan Africa. Ages 2–3 years. Natural sleep protocol, 77% scan success rate. Bayley-III cognitive assessments available |

### C.6 Simultaneous EEG-fMRI Datasets (Cross-Modal Bridge — Critical)

**Total synchronized subjects: ~363 (revised from original ~80 estimate)**

| # | Dataset | Subjects | EEG Ch | fMRI TR | Task | Access | Joint Tokens | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **CineBrain** | 6 | 64-ch | ~2.0s | Movie (BBT) | Contact Feng group | ~65K | 6h/subj — highest token density |
| 2 | **Carbon Wire Loop** (ATR) | **39** | 64-ch + CWL | ~2.0s | Oddball + N-back + rest | Registration (doi.org/10.34860/atr-EfP-2025) | ~78K | **Largest single new dataset**. Superior EEG quality (2026) |
| 3 | **ds006040** gradCPT | 28 | 64-ch | ~2.0s | Sustained attention | Open (OpenNeuro, 2026) | ~56K | |
| 4 | **ds007216** experience sampling | 25 | 64-ch | ~2.0s | Rest + movie + sampling | Open (OpenNeuro) | ~50K | Multi-session |
| 5 | **ds003768** sleep | ~20 | 32-64ch | ~2.0s | NREM sleep | Open (OpenNeuro) | ~40K | |
| 6 | **ds005127** NIH sleep | ~20 | 32-64ch | ~2.0s | NREM (thalamic slow waves) | Open (OpenNeuro, 2026) | ~40K | |
| 7 | **EEG-PET-MRI sleep** | 21 | 64-ch | ~2.0s | NREM sleep | Upon request (Geneva) | ~42K | Tri-modal |
| 8 | **NE arousal** | 28 | 64-ch | ~2.0s | Detection + arousal | Partially (ds003768+ds001242) | ~42K | |
| 9 | **NAT_VIEW (NKI)** | **22** | 64-ch BP BrainCapMR | ~2.1s | Movie (Inscapes, Despicable Me EN/HU, The Present, monkeys) + checkerboard + resting | Open (NITRC/INDI, CC BY 4.0) | **~88K** | **Rich stimulus**: 5 movies + checkerboard + resting + eye-tracking + physio. 5,000 Hz EEG |
| 10 | **INDI Naturalistic** | ~20 | 64-ch | ~2.0s | Movie watching | Open (INDI) | ~20K | |
| 11 | **HCP 7T EEG-fMRI** | ~20 | 64-ch | ~0.72s | Resting | Limited (ConnectomeDB) | ~10K | |
| 12 | **ds002338** NF XP2 | 23 | 64-ch | ~2.0s | Motor imagery NF | Open (OpenNeuro) | ~35K | |
| 13 | **ds002158** Pereira | 20 | 64-ch | ~2.0s | Perceptual judgment | Open (OpenNeuro) | ~20K | PNAS |
| 14 | **ds002718** Wakeman & Henson | 13 | 70-ch | ~2.0s | Face processing | Open (OpenNeuro) | ~20K | Classic |
| 15 | **ds002725** music emotion | 20 | 64-ch | ~2.0s | Affective music | Open (OpenNeuro) | ~20K | |
| 16 | **ds004752** Zurich | 15 | sEEG + scalp | ~2.0s | Verbal WM | Open (OpenNeuro) | ~15K | Scalp+iEEG simultaneous |
| 17 | **DANDI:000623** | ~10 | ~100ch sEEG | ~2.0s | Movie watching | Open (DANDI) | ~20K | |
| 18 | **DANDI:000574** | ~10 | 19 + sEEG | ~2.0s | Verbal WM | Open (DANDI) | ~10K | |
| 19 | **Other (ds002336, ds004196, ds006033, Allen et al.)** | ~27 | 64-ch | ~2.0s | Various | Open | ~37K | |

> **Key corrections from the original audit**: CineBrain is 6 subjects (not ~20 — paper says "six participants"). DANDI:001211 and DANDI:001543 are MOUSE data (not human). Cross-modal bridge remains 26,000× below Chinchilla — physics-informed HRF approach is mandatory.

---

## D. Summary Statistics

### D.1 Revised Token Budgets

| Modality | Subjects (approx) | High-res Tokens (19ch equiv) | Notes |
|---|---|---|---|
| **Scalp EEG** (high-res, ≥19ch) | **~90,500+** | **~668M** | **TUH corrected to 60K+** (was 25K) + ABCD (12K) + NSSRR sleep (5.7K→12K w/ SHHS+CHAT) + NMT India (2K) + MOABB BCI (~2,500) + THINGS-EEG (50) |
| **Scalp EEG** (low-res, <19ch) | **~6,200** | **~320M** (raw, needs interp) | Sleep-EDF, SHHS, MESA, Muse, NeuroSky, neonatal |
| **ECoG / sEEG** (human, open) | **~1,128** | **~2.7B** | IEEG.org + DANDI + OpenNeuro + new finds |
| **MEG** (whole-head) | **~1,000–1,100** | **~3.5M** | Cam-CAN (~700) + HCP MEP (~95) + OpenNeuro aggregate (~200). Same ms-timescale as EEG |
| **fMRI** (ROI time series) | **~122,000+** | **~72M** | UKB + ABCD + HCP + ADNI + PPMI + OASIS + ABIDE + CoRR + Cam-CAN + ALSPAC + Generation R + IMAGEN + AIBL + AOMIC + CHCP + REST-meta-MDD + Narratives + PING + NACC + B-SNIP + SALD + Cimbi + others |
| **Sync EEG-fMRI** | **~363** | **~753K** (joint) | 5.1× original estimate; NAT_VIEW adds 22 subjects + 88K tokens; still 26,000× deficit |
| **Sync EEG-fNIRS** | **~29** | **~50K** (joint) | Shin2017A/B — MI + mental arithmetic |
| **Sync MEG-EEG** | **~16** | **~20K** (joint) | ds000117 — face perception. Rare but principled alignment |
| **Sleep PSG** (≥16ch) | **~12,100** | **~390M** | SHHS (6,441) + MrOS (3,040) + Wisconsin (1,500) + Cleveland (700) + SOF (460) + CHAT (1,000) |

### D.2 Geographic/Demographic Diversity

| Region | Before | After (New Datasets Added) | Key Addition |
|---|---|---|---|
| North America | 75% | 60% | ABCD, NSSRR, ADNI, PPMI, OASIS, ABIDE, PNC, GSP |
| Europe | 15% | 15% | Bucharest (Romania), EPILEPSIAE, IMAGEN, ALSPAC, Cam-CAN, AOMIC |
| South Asia | <1% | **5%** | NMT India (~2,000 subj) |
| East Asia (China) | 5% | **8%** | ChineseEEG-2, SEED-V, CNEP, CHCP, REST-meta-MDD, Beijing Enhanced/EOEC |
| East Asia (Korea) | <1% | **5%** | VitalDB (6,388 subj) |
| East Asia (Japan) | <1% | **2%** | Carbon Wire Loop, SRPBS/Brain/MINDS |
| Australia/Oceania | **0%** | **2%** | AIBL (3,045 subj) |
| Africa | **0%** | **<1%** | **Drakenstein Child Health (239 children, South Africa)** — first sub-Saharan African pediatric MRI dataset. Still critically underserved |
| Indigenous / South America | **0%** | **0%** | **Still unfillable** |

### D.3 DIVER-1 Diversity Score

| Dimension | Score (/10) | Rationale |
|---|---|---|
| Subject count | **10** | **90K+ high-res** (TUH corrected 60K+ — was 25K; SHHS + CHAT + Brain-CODE + PAIN add ~15K) |
| Age diversity | 8 | ABCD adolescents, MrOS aging, CNEP neonates |
| Clinical diversity | **8** | ADNI, PPMI, OASIS, ABIDE, REST-meta-MDD, COBRE, NUSDAST, Brain-CODE (cross-disorder), PAIN (chronic pain), Temple EEG (60K+ clinical) added |
| Geographic diversity | 8 | India, Korea, Romania, China, Japan, Australia, **South Africa (Drakenstein)** added — Africa seat now <1% |
| Task diversity | **10** | **MOABB BCI**: MI (37 datasets), P300 (26 datasets), SSVEP (16 datasets), CVEP (6 datasets), imagined speech (6 datasets), mental workload (3 datasets). Plus sleep, anesthesia, TMS, language, movies, visual objects, dialogues |
| Equipment diversity | 7 | Consumer, clinical, research, high-density, 7T, multi-echo, dry electrodes, MEG (Elekta Neuromag) |
| **Overall** | **8.2** | **Sufficient for ~3.89B total / ~3.49B active MoE params**。DIVER-1 数据多样性 is the quality ceiling, not parameter count |

### D.4 Cross-Modal Alignment Potential

| Modality Pair | Shared Physics | Alignment Quality | Data Volume | Priority |
|---|---|---|---|---|
| **EEG ↔ MEG** | Same source currents (Maxwell equations) | **Excellent** — different forward operators, same latent state | ~1,100 MEG subjects; simultaneous in ds000117 (16 subj) | **HIGH** for physics validation |
| **EEG ↔ ECoG** | Same electrical potential domain | **Excellent** — spatial scale bridge | ~1,128 ECoG subjects; DANDI:000574 has scalp+iEEG | **HIGH** |
| **fMRI ↔ fNIRS** | Same hemodynamic signal (HRF) | **Good** — both BOLD/oxy-deoxy | ~29 sync subjects (Shin2017); very sparse fNIRS spatial coverage | **LOW** — insufficient data |
| **EEG ↔ fMRI** | Neurovascular coupling (HRF) | **Mediocre** — indirect, nonlinear, slow | **~363** sync subjects | **CRITICAL** — only viable macro-scale bridge |
| **MEG ↔ fMRI** | Neurovascular coupling | **Mediocre** — same HRF problem as EEG→fMRI | Very rare simultaneous | **LOW** |

---

## E. Data Access Priority Queue

### P0: Download Immediately (Open, No Barriers)
1. **NMT Scalp EEG (India)** — ~2,000 subjects, figshare, fills South Asian gap
2. **VitalDB (Korea)** — 6,388 subjects, vitaldb.net, fills Korean + anesthesia gap
3. **ds006040** gradCPT — 28 subjects, largest single new open sync EEG-fMRI
4. **ds007216** experience sampling — 25 subjects, multi-session sync EEG-fMRI
5. **ds002338** NF XP2 — 23 subjects, motor imagery sync EEG-fMRI
6. **ds004752** Zurich scalp+iEEG — 15 subjects, DOUBLES cross-modal alignment

### P1: Apply for Access (Minimal Barriers)
1. **ABCD Study** — NDA DUC (11,875 subjects, transforms adolescent coverage)
2. **NSSRR Sleep** (MrOS + Wisconsin + Cleveland + SOF) — ~5,700 subjects, 206M tokens
3. **Carbon Wire Loop** (ATR Japan) — register at doi.org/10.34860/atr-EfP-2025
4. **NDA EEG Aggregates** — NDA DUC (~5,000+ subjects across studies)
5. **CoRR** — Open, 1,629 subjects, 5,093 rs-fMRI scans. Benchmark reliability
6. **Narratives** — Open (CC0), 345 subjects, largest naturalistic language fMRI
7. **Cam-CAN** — Open (reg), ~2,000+ subjects fMRI + **~700 subjects MEG**, full lifespan 18–88. **Only large lifespan MEG dataset**
8. **Cam-CAN MEG** — Open (reg), ~700 subjects, 306ch Elekta, resting + tasks. **Best MEG→latent-space alignment data due to same subjects as fMRI/structural**
9. **OASIS-3** — Open (DUA), 1,378 subjects, aging + Tau PET
9. **ABIDE I & II** — Open, ~2,100 subjects, largest autism neuroimaging
10. **ADHD-200** — Open, ~800–1,000 subjects, ADHD competition benchmark
11. **MOABB BCI aggregate** — Open, ~2,500+ subjects across MI/P300/SSVEP/CVEP/speech. Standardized preprocessing via `pip install moabb`
12. **Stieger2021** — Open, 62 subjects, ~250K trials, 7–11 sessions MI. Largest single MI dataset
13. **Liu2020BETA** — Open, 70 subjects, 40-class SSVEP. Largest SSVEP
14. **Liu2022EldBETA** — Open, 100 subjects, 7 sessions. Largest SSVEP subject count
15. **Lee2019** — Open, 54 subjects, MI + SSVEP + ERP (same subjects, multi-paradigm)
16. **Cho2017** — Open, 52 subjects, 64ch MI. Large Korean dataset
17. **Dreyer2023** — Open, 87 subjects, multi-center MI. Largest MI subject count
18. **Shin2017A/B** — Open (TU Berlin), 29 subjects, **simultaneous EEG+fNIRS**. MI + mental arithmetic

### P2: Negotiate Access (Restricted / Requires Collaboration)
1. **Chinese Hospital Archives** — ~10,000+ subjects, largest potential upside
2. **VA/TBI EEG** — requires VA data use agreement, ~500+ subjects
3. **EPILEPSIAE** — European consortium, 275+ patients
4. **CineBrain** — contact Feng group at Fudan (6 subjects, 6h/subj movie)
5. **ADNI** — LONI application, ~2,500+ subjects, Alzheimer's gold standard
6. **PPMI** — Open (reg), 1,500+ subjects, Parkinson's largest biomarker study
7. **AIBL** — Collaboration application, 3,045 subjects, 15+ year longitudinal
8. **IMAGEN** — DSA required, ~2,000 adolescents, European multi-site
9. **ALSPAC** — Research proposal, ~5,000 scanned, UK birth cohort
10. **Generation R** — Collaboration agreement, ~3,000–4,000 MRI, Dutch cohort
11. **REST-meta-MDD** — Consortium access, ~2,300+ subjects, Chinese MDD
12. **PNC** — dbGaP, ~1,600 imaged, adolescent development
13. **GSP** — Harvard application, ~1,570 young adults
14. **CHCP** — Mixed access, ~500+ subjects, Chinese counterpart to HCP

### P3: Verification Needed
1. **TUH v3.0 status** — already confirmed 60K+ on official site. Monitor for further expansion
2. **SMA iEEG Speech** — >50 subjects claimed, access path unclear
3. **iSEED** — available on request, Chinese sEEG emotion
4. **SRPBS / Brain/MINDS** — Japanese national project, 1,000s subjects, access paths unclear
5. **Brain-CODE** (Ontario Brain Institute) — ~5,000+ subjects, cross-disorder multi-modal. Application required
6. **PAIN Repository** — ~2,000+ subjects, chronic pain fMRI. Application required
