# Thesis: Document Forgery Detection

This repository contains the code developed for a thesis project on **document forgery detection and localization**.

The project focuses on pixel-level localization of manipulated regions in document images using deep learning models and forensic image representations. The main experimental direction uses **RGB**, **ELA**, auxiliary channels, and combinations of these inputs, especially with SegFormer-based architectures and the RTM dataset.

The repository also contains exploratory code for **PEP** experiments. PEP support was implemented during development, but it was not selected as a main thesis direction because it is less established in the literature for this specific setting, unlike ELA, which is a well-known forensic representation in image forensics.

---

## Repository Structure

```text
thesis-document-forgery-detection/
│
├── datasets/
│   └── Dataset-related code and preprocessing utilities
│
├── models/
│   ├── HRNet-based models
│   ├── SegFormer-based models
│   ├── MIML-related models
│   ├── Dataset classes
│   ├── Training utilities
│   ├── Testing utilities
│   ├── Metrics
│   ├── Losses
│   └── Custom collate functions
│
├── scripts/
│   └── Training, testing and experiment scripts
│
├── inspection_pngs_doctamper_rtm/
│   └── Visual inspection outputs
│
├── test_doctamper_pep_outputs/
│   └── DocTamper test outputs for PEP-based exploratory experiments
│
├── test_doctamper_rgb_outputs/
│   └── DocTamper test outputs for RGB-based experiments
│
├── test_rtm_dual_branch_model_rgb_outputs/
│   └── RTM test outputs for dual-branch RGB models
│
├── test_rtm_dual_branch_model_with_conv_rgb_outputs/
│   └── RTM test outputs for dual-branch RGB models with convolutional fusion
│
├── test_rtm_mmseg_segformer_rgb_outputs/
│   └── RTM test outputs for SegFormer RGB models
│
├── test_rtm_mmseg_segformer_rgb_with_class_head_outputs/
│   └── RTM test outputs for SegFormer RGB models with classification head
│
├── test_rtm_several_inputs_outputs/
│   └── RTM test outputs for several input combinations
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Datasets

This project uses two main datasets.

### DocTamper

DocTamper is used for document tampering localization experiments.

Dataset link:

```text
https://www.kaggle.com/datasets/dinmkeljiame/doctamper
```

Expected structure:

```text
DocTamper/
├── DocTamperV1-FCD/
│   ├── tampered/
│   └── mask/
│
├── DocTamperV1-SCD/
│   ├── tampered/
│   └── mask/
│
└── DocTamperV1-TestingSet/
    ├── tampered/
    └── mask/
```

### Real Text Manipulation Dataset

The Real Text Manipulation dataset, referred to in this repository as **RTM**, is used for experiments involving real text manipulation cases.

Dataset link:

```text
https://drive.google.com/file/d/11AHZ8ih_kDCFilGceevppcGkKR4vDJD2/view
```

After downloading and extracting the dataset, update the dataset paths inside the training and testing scripts according to your local machine.

---

## Input Modalities

This repository contains code for several input representations used in document forgery localization experiments.

### RGB

RGB is the standard image representation and is used as a baseline input for document forgery localization.

It is used in both DocTamper and RTM experiments.

### ELA

ELA stands for **Error Level Analysis**.

It is a forensic image representation commonly used to highlight compression inconsistencies and possible manipulated regions. Since ELA is a known and established technique in image forensics, it is one of the main representations considered in the final experimental workflow.

### Auxiliary Channels

Some SegFormer experiments also include two auxiliary channels computed from the same input representation used by the model.

These auxiliary channels are:

- **High-pass channel**: highlights local high-frequency details and residual patterns.
- **Edge channel**: highlights contours, boundaries, and abrupt spatial transitions.

These channels provide complementary spatial information to the model. They are not independent forensic representations like ELA; instead, they are derived from the selected base input, such as RGB or ELA.

In this repository, variants marked with `2aux` use these two auxiliary channels together with the main input representation.

### PEP

PEP stands for **Probabilistic Error Potential**.

PEP was implemented as an experimental input representation based on JPEG compression information. The repository contains dataset classes, collate functions, model runners, and training/testing scripts that support PEP inputs.

However, PEP was not selected as one of the main experimental directions in the final thesis workflow. The main reason is that PEP is a more exploratory representation in this context and has less direct literature support for the selected datasets and experimental setup. In contrast, ELA is a better-known forensic representation and is easier to justify with existing work.

Therefore, PEP-related code is kept in the repository for completeness and future experimentation, but the main experimental focus is on RGB, ELA, auxiliary channels, and their combinations.

### Multiple Input Combinations

The experiments with SegFormer on the RTM dataset, explore different input configurations, such as:

- RGB only
- ELA only
- RGB + auxiliary channels
- ELA + auxiliary channels
- RGB + ELA
- RGB + ELA + auxiliary channels

PEP-based combinations may exist in the codebase, but they should be considered exploratory rather than part of the main reported experimental setup.

---

## Models

### HRNet

HRNet-based models are included for document forgery localization experiments in the DocTamper dataset.

Implemented variants include:

- HRNet with RGB input
- HRNet with PEP input support

The PEP variant was implemented during the exploratory phase of the project, but the final focus moved away from PEP due to the lack of direct literature support for this representation in the selected experimental setting.

### SegFormer

SegFormer-based models are used for the main segmentation experiments, especially with the RTM dataset.

Implemented variants include:

- SegFormer with RGB input (DocTamper and RTM datasets)
- SegFormer with PEP input support (DocTamper and RTM datasets)
- SegFormer with ELA input (RTM dataset)
- SegFormer with auxiliary channels (RTM dataset)
- SegFormer with several input combinations (RTM dataset)
- SegFormer with classification head (RTM dataset)
- SegFormer in a dual-branch model (with and without convolutional layers) (RTM dataset)

### MIML

HRNet-based models are included for document forgery localization experiments in the DocTamper dataset.

Implemented variants include:

- MIML with RGB input
- MIML with PEP input support

## Important Note

Some paths in the scripts are machine-specific and must be changed before running the code.
