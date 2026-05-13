# EEG Datasets for Project MAGI
> The overall requirement is that data can't be less than 32 channels. This won't be a complete list for all EEG datasets. Instead, we only want it to be an internal reference.

## Artifact Removing / Denoising (Important)
1. [EEGdenoisenet](https://github.com/ncclabsustech/EEGdenoiseNet)
2. [A semi-simulated EEG/EOG dataset for the comparison of EOG artifact rejection techniques](https://data.mendeley.com/datasets/wb6yvr725d)
3. [Dataset on Emotion with Naturalistic Stimuli (DENS)](https://openneuro.org/datasets/ds003751/versions/1.0.6), EEG, ECG, and EMG 

## Task-Independent Pretraining Large-Scale Dataset
1. [HBN - Healthy Brain Networks](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_EEG.html) [NIPS 2025 EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding](https://arxiv.org/abs/2506.19141)
2. [Multimodal Resource for Studying Information Processing in the Developing Brain (MIPDB)](https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/index.html)
3. [Cuban Human Brain Mapping](https://www.synapse.org/Synapse:syn22324937)
4. [Latin American Brain Health Institute](https://www.synapse.org/Synapse:syn51549340/wiki/624187)
5. [Resting-state EEG data before and after cognitive activity across the adult lifespan and a 5-year follow-up](https://openneuro.org/datasets/ds005385/versions/1.0.3) (n=608)
6. [A large EEG dataset with a simple gambling task](https://osf.io/65x4v/)
7. [TDBrain Dataset](https://brainclinics.com/resources/) The TDBrain dataset is EEG data for 1200 subjects. (need interpolation)

## 32-64 Channels 
### Motor-Imagery
1. **LaBram, NeuroLM Pretrained** [Grasp and Lift EEG Challenge](https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data): 12 subjects, 32channels@500Hz, for 6 grasp and lift  events, namely a). HandStart b). FirstDigitTouch c). Both StartLoadPhase d). LiftOff e). Replace  f). BothReleased
2. [Multi-channel EEG Recordings During 3,936 Grasp and Lift Trials with Varying Weight and Friction](https://figshare.com/collections/WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376)
3. [Mental-Imagery Dataset](https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698): 13 participants with over 60,000 examples of motor imageries in 4 interaction paradigms recorded with 38 38-channel medical-grade EEG system. It contains data for upto 6 mental imageries primarily for the motor movements. [[Article]](https://www.nature.com/articles/sdata2018211#ref-CR57)
4. [High-Gamma Dataset](https://github.com/robintibor/high-gamma-dataset): 128-electrode dataset obtained from 14 healthy subjects with roughly 1000 four-second trials of executed movements divided into 13 runs per subject.  The four classes of movements were movements of either the left hand, the right hand, both feet, or rest.

### Emotion Recognition
1. [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/): Includes 32 subjects, each watching 1-min long excerpts of music-videos, rated by users in terms of arousal/valence/like-dislike/dominance/familiarity, and frontal face recording of 22/32 subjects.
2. [HCI-Tagging](https://mahnob-db.eu/hci-tagging/): Subjects were shown video clips (fragments of movies) and they were asked to annotate the emotional state on the scale of valence and arousal. During the whole experiment, audio, video, gaze data, and physiological data were recorded simultaneously with accurate synchronisation between sensors.
3. [Leipzig Mind-Brain-Body Dataset - LEMON](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html)

### Error-Related Potentials (ErrP)
1. **LaBram, NeuroLM Pretrained** [BCI-NER Challenge](https://www.kaggle.com/c/inria-bci-challenge): 26 subjects, 56 EEG Channels for a P300 Speller task, and labeled dataset for the response elicited when P300 decodes a correct or incorrect letter.
2. [Error Related Potentials from Gaze-Based Typesetting](https://www.mamem.eu/results/datasets/)

### Event Related Potentials [ERPs]
1. **LaBram, NeuroLM Pretrained** [Target Versus Non-Target](https://zenodo.org/record/3267302): 38 subjects playing a multiplayer and collaborative version of Brain Invaders, a visual P300 Brain-Computer Interface using oddball paradigm with adaptive Riemannian Geometry (no-calibration). 32 electrodes per subject, wet, 2 subjects during each session. [publication](https://hal.archives-ouvertes.fr/hal-02173958), [code](https://github.com/plcrodrigues/py.BI.EEG.2014b-GIPSA). Dataset id: bi2014b.
2. **LaBram, NeuroLM Pretrained** [Target Versus Non-Target](https://zenodo.org/record/3266930): 50 subjects playing Brain Invaders, a visual P300 Brain-Computer Interface using oddball paradigm with adaptive Riemannian Geometry (no-calibration). 32-electrodes, wet. 3 sessions per subject with modulation of flash duration. [publication](https://hal.archives-ouvertes.fr/hal-02172347), [code](https://github.com/plcrodrigues/py.BI.EEG.2015a-GIPSA). 
3. **LaBram, NeuroLM Pretrained** [Target Versus Non-Target](https://zenodo.org/record/3268762): 44 subjects playing a multiplayer (cooperation and competition) version of Brain Invaders, a visual P300 Brain-Computer Interface using oddball paradigm with adaptive Riemannian Geometry (no-calibration). 32 electrodes per subject, wet, 2 subjects for each session. [publication](https://hal.archives-ouvertes.fr/hal-02173913), [code](https://github.com/plcrodrigues/py.BI.EEG.2015b-GIPSA). Dataset id: bi2015b.

## 64+ Channels
### Motor-Imagery
1. [Left/Right Hand MI](http://gigadb.org/dataset/100295):  Includes 52 subjects (38 validated subjects with discriminative features), results of physiological and psychological questionnaires, EMG Datasets, location of 3D EEG electrodes, and EEGs for non-task related states
2. **LaBram, NeuroLM Pretrained** [Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/): Includes 109 volunteers, 64 electrodes, 2 baseline tasks (eye-open and eye-closed), motor movement, and motor imagery (both fists or both feet)
3. **LaBram, NeuroLM Pretrained**, **Benchmark Dataset?** [BCI Competition IV-1](http://www.bbci.de/competition/iv/#dataset1): 64 EEG channels at 1000Hz sampling rate for 2 classes of left hand, right hand, foot (+ idle state) for 7 subjects. Evaluation data is continuous EEG, which also contains periods of idle state. 
  
### Emotion-Recognition
1. **LaBram, NeuroLM Pretrained** [Enterface'06](http://www.enterface.net/results/): Enterface'06 Project 07: EEG(64 Channels) + fNIRS + face video, Includes 16 subjects, where emotions were elicited through a selected subset of the IAPS dataset.
2. **LaBram, NeuroLM Pretrained**, **Benchmark Dataset** [SEED](http://bcmi.sjtu.edu.cn/~seed/seed.html): 15 subjects were shown video clips eliciting positive/negative/neutral emotion and EEG was recorded over 62 channels.
3. **LaBram, NeuroLM Pretrained** [SEED-IV](http://bcmi.sjtu.edu.cn/~seed/seed-iv.html): 15 subjects were shown video clips eliciting happy/sad/neutral/fear emotions, and EEG was recorded over 62 channels (with eye-tracking) for 3 sessions per subject (24 trials per session).
4. **LaBram, NeuroLM Pretrained** SEED-FRA, SEED-GER
5. [EmoEEG-MC: A Multi-Context Emotional EEG Dataset for Cross-Context Emotion Decoding](https://openneuro.org/datasets/ds005540/versions/1.0.7)
6. [Imagined Emotion Study](https://openneuro.org/datasets/ds003004/versions/1.1.1)
7. [EEG4EMO](https://www.interdigital.com/data_sets/hr-eeg4emo-dataset)
    
### Cognitive Tasks
1. [Face vs. House Discrimination](https://purl.stanford.edu/xd109qh3109): 7 Epileptic subjects were presented with 50  grayscale stimulations each for Face and House pictures. For each subject, a total of 3 experimental runs were conducted, resulting in 300 stimulations. 
2. [Impedance Data](https://erpinfo.org/impedance): 12 subjects for P300 task (Oddball paradigm) with 20% of rare stimuli. In total, there were 128 target stimuli and 512 standard stimuli. The dataset was collected in a way such that one recording contains different impedances in the electrodes. [[Article]](https://static1.squarespace.com/static/5abefa62d274cb16de90e935/t/5ac6962a8a922d0b8b8be6a1/1522964012664/Kappenman+2010+Psychophys+Impedance.pdf) [[Data]](https://erpinfo.org/impedance)
3. [Sustained-Attention Driving Task (SADT)](https://figshare.com/articles/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task/6427334/5): 27 subjects for sustained-attention driving in a VR setting for monitoring event-related potentials. Each subject participated in two 90-minute sessions (with and without kinesthetic feedback) and was recorded with 32 channels and 500Hz. [[Article]](https://www.nature.com/articles/s41597-019-0027-4#Sec12) [[Pre-processed dataset]](https://figshare.com/articles/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task_preprocessed_dataset_/7666055/3)
4. [ERP Core](https://github.com/andrewxstewart/ERP_CORE) 6-7 ERP paradigms including N170, N400, LRP/ERN, etc., from 40 participants, includes analysis scripts, experiments, results, and data. [[Article]](https://psyarxiv.com/4azqm/) [[Website]](https://erpinfo.org/erp-core)
5. [The Nencki-Symfonia EEG/ERP dataset](https://doi.org/10.1093/gigascience/giac015): high-density electroencephalography (EEG) dataset obtained at the Nencki Institute of Experimental Biology from a sample of 42 healthy young adults with three cognitive tasks: (1) an extended Multi-Source Interference Task (MSIT+) with control, Simon, Flanker, and multi-source interference trials; (2) a 3-stimuli oddball task with frequent standard, rare target, and rare distractor stimuli; (3) a control, simple reaction task (SRT); and additionally (4) a resting-state protocol (REST). [[Data]](http://doi.org/10.5524/100990)
6. [Neural Evidence for the Contribution of Active Suppression During Working Memory Filtering](https://osf.io/a65xz/)
7. [Open Data 2018: Contralateral delay activity tracks fluctuations in working memory performance](https://osf.io/8xuk3/)

### Resting State & Large-Scale Dataset
1. [Resting State EEG Data](https://dataverse.tdl.org/dataverse/txstatecogelectro): 22 subjects, 72 EEG Channels for a resting task of 8 mins with 4 mins of eyes closed and 4 mins of eyes open. [[Article]](https://www.frontiersin.org/articles/10.3389/fnins.2017.00425)
2. **LaBram, NeuroLM Pretrained** [SPIS Resting State Dataset](https://github.com/mastaneht/SPIS-Resting-State-Dataset): 10 subjects, 64 channels, 2.5 minutes recording in each state (eyes-closed and eyes-open) prior to a 105-minute session of Sustained Attention to Response Task with fixed-sequence and varying ISIs. [[Article]](https://www.ncbi.nlm.nih.gov/pubmed/32167917)
3. [BIDS formatted EEG meditation experiment data](https://zenodo.org/records/2536267)
4. **LaBram, NeuroLM pretrained** [Resting State EEG Data](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/EG0LJI)
  
### Auditory Stimulus
1. [OpenMIIR](https://github.com/sstober/openmiir): 10 subjects, 64 EEG Channels for a music imagery task of 12 different pieces w/ different meter, length and tempo. [[Article]](https://pdfs.semanticscholar.org/cde4/b1ec89f2c05a41f1143792a890a00e89541a.pdf)
2. Dataset from Stanford [NMED-T](https://exhibits.stanford.edu/data/catalog/jn859kj8079), [NMED-H](https://exhibits.stanford.edu/data/catalog/sd922db3535), [NMED-M](https://exhibits.stanford.edu/data/catalog/kt396gb0630), [NMED-E](https://exhibits.stanford.edu/data/catalog/pp371jh5722), [NMED-RP](https://exhibits.stanford.edu/data/catalog/rz763kn3821), [EEG-Recorded Responses to Short Chord Progressions](https://purl.stanford.edu/js383fs8244)
3. [Auditory EEG](https://www.physionet.org/content/auditory-eeg/1.0.0/)
4. [Music Listening- Genre EEG dataset (MUSIN-G)](https://openneuro.org/datasets/ds003774)
5. [An EEG dataset recorded during affective music listening](https://openneuro.org/datasets/ds002721/versions/1.0.3)

### Visual Stimulus
  1. [MNIST Brain Digits](http://mindbigdata.com/opendb/index.html): EEG data when a digit(0-9) is shown to the subject, recorded 2s for a single subject using Minwave, EPOC, Muse, Insight. Includes over 1.2M samples. (Need interpolation)
  2. [Imagenet Brain](http://www.mindbigdata.com/opendb/imagenet.html): A random image is shown (out of 14k images from the Imagenet ILSVRC2013 train dataset) and EEG signals are recorded for 3s for one subject. Includes over 70k samples. (Need interpolation)
  3. [EEG-ImageNet](https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset)
  4. [ThoughtViz](https://github.com/ptirupat/ThoughtViz): consists of 10 different object classes, which is a subset of the ImageNet
  5. [A Representational Similarity Analysis of the Dynamics of Object Processing Using Single-Trial EEG Classification](https://purl.stanford.edu/bq914sc3730)
  6. [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://github.com/ncclab-sustech/EEG_Image_decode)
  7. [SEED-DV](https://bcmi.sjtu.edu.cn/home/eeg2video/)
  8. [THINGS-EEG](https://figshare.com/articles/dataset/THINGS-EEG/14721282), [THINGS-EEG2](https://osf.io/3jk45/), https://openneuro.org/datasets/ds003825/versions/1.1.0
  9. [EEG human categorization data](https://www.nitrc.org/projects/eegdataanimal). Subjects are performing a go-nogo categorization task and a go-no recognition task on natural photographs presented very briefly
  10. [EEG/FMRI Naturalistic Viewing Dataset](https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html)
  11. [A large and rich EEG dataset for modeling human visual object recognition](https://osf.io/3jk45/)
  12. [MAMEM SSVEP](https://www.physionet.org/content/mssvepdb/1.0.0/)
  13. **LaBram, NeuroLM pretrained** [Mental Effort and Information-Processing Costs Are Inversely Related to Global Brain Free Energy During Visual Categorization](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/SS2NHB)

  
### Clinical EEG
1. **LaBram, NeuroLM Pretrained**, **Benchmark Dataset** [TUH EEG Resources](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml): Massive amount of data for (i) Abnormal EEG and (ii) EEG Seizures, needs interpolation
2. [Predict-UNM](http://predict.cs.unm.edu/): A large repository of clinical EEG datasets
3. [ADHD Dataset](https://nda.nih.gov/study.html?id=1938)
4. [pre- and postsurgical EEG recorded from epilepsy patients and HFO markings](https://gin.g-node.org/USZ_NCH/Scalp_EEG_HFO)
5. [Resting State EEG Biomarkers of Cognitive Decline (PLOS ONE - INSPECDS)](https://data.mendeley.com/datasets/wttpypkctg/2)
6. [CHB-MIT Scalp EEG Database](https://www.physionet.org/content/chbmit/1.0.0/) a collection of EEG recordings of 22 pediatric subjects with intractable seizures
7. [DREAM: A Dream EEG and Mentation Database](https://bridges.monash.edu/projects/The_Dream_EEG_and_Mentation_DREAM_database/158987)
8. [SLEEP Dataset](https://sleeptight.isr.uc.pt/)
9. [AHPHY-SLEEP](https://osf.io/r26fh/)
10. [Simultaneous EEG and fMRI signals during sleep from humans](https://openneuro.org/datasets/ds003768/versions/1.0.3)
11. **LaBram, NeuroLM pretrained** [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
12. [UC San Diego Resting State EEG Data from Patients with Parkinson's Disease](https://openneuro.org/datasets/ds002778/versions/1.0.5)]
13. [A complementary dataset of open-eyes EEG recordings in a photo-stimulation setting from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects](https://openneuro.org/datasets/ds006036/versions/1.0.5)

### Language - EEG-To-Text
1. [Le Petit Prince Hong Kong: Naturalistic fMRI and EEG dataset from older Cantonese speakers](https://openneuro.org/datasets/ds004718)
2. [Dryad-Speech](https://datadryad.org/stash/dataset/doi:10.5061/dryad.070jc): 5 different experiments for studying natural speech comprehension through a variety of tasks including audio, visual stimulus and imagined speech. [[Main Article]](https://www.sciencedirect.com/science/article/pii/S0960982218301465) [[Supplemntary Article]](https://www.ncbi.nlm.nih.gov/pubmed/26412129)
3. [ZuCo](https://www.nature.com/articles/sdata2018291): EEG and eye-tracking recording for 12 healthy adult native English speakers, each reading natural English text for 4–6 hours. 21,629 words in 1107 sentences and 154,173 fixations.  [[Paper]](https://www.nature.com/articles/sdata2018291) [[Data]](https://osf.io/2urht/)
4. [ChineseEEG: A Chinese Linguistic Corpora EEG Dataset for Semantic Alignment and Neural Decoding](https://openneuro.org/datasets/ds004952)
5. [Inner Speech](https://openneuro.org/datasets/ds003626/versions/1.0.1)
6. [Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech](https://datadryad.org/dataset/doi:10.5061/dryad.070jc)
7. [Chisco Dataset](https://openneuro.org/datasets/ds005170/versions/1.1.2)
8. [Kara-One](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html): Imagined and vocalized phonemic and single-word prompts to access the language and speech production. 14 subjects recorded using a 64-channel Neuroscan Quick-Cap, face tracking, and audio. [[Article]](http://www.cs.toronto.edu/~complingweb/data/karaOne/ZhaoRudzicz15.pdf)


### Misc
1. **LaBram, NeuroLM pretrained** Self-collected EEG corpus (Jiang et al., 2023; 2021; Luo et al., 2022; Li et al., 2021; Tao & Lu, 2020), seems that not publicly available
