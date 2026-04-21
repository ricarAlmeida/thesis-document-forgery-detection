# Datasets (DocTamper)

This repository does **not** include the DocTamper dataset files (LMDB or extracted images/masks).

## 1) Download / location

Download DocTamper following the official source/instructions from the dataset authors (https://www.kaggle.com/datasets/dinmkeljiame/doctamper).

## 2) Expected directory structure (example)

```text
/path/to/datasets/doc-tamper/
├── DocTamperV1-FCD/
│   ├── data.mdb
│   ├── lock.mdb
│   ├── tampered/
│   ├── mask/
│   └── test/
│       ├── tampered/
│       └── mask/
├── DocTamperV1-SCD/
│   ├── data.mdb
│   ├── lock.mdb
│   ├── tampered/
│   ├── mask/
│   └── test/
│       ├── tampered/
│       └── mask/
└── DocTamperV1-TestingSet/
    ├── data.mdb
    ├── lock.mdb
    ├── tampered/
    ├── mask/
    └── test/
        ├── tampered/
        └── mask/
```
# Datasets (RTM / RealTextManipulation)

This repository does **not** include the RTM (RealTextManipulation) dataset files (images, masks, or generated splits).

## 1) Download / location

Download the RTM / RealTextManipulation dataset following the official source/instructions from the dataset authors (https://drive.google.com/file/d/11AHZ8ih_kDCFilGceevppcGkKR4vDJD2/view).

## 2) Expected directory structure (example)

```text
/path/to/datasets/RealTextManipulation/
├── JPEGImages/
│   ├── cover_0001.jpg
│   ├── cpmv_0001.jpg
│   ├── good_0001.jpg
│   ├── edit_0001.jpg
│   ├── inpaint_0001.jpg
│   ├── insert_0001.jpg
│   └── splice_0001.jpg
├── SegmentationClass/
│   ├── cover_0001.png
│   ├── cpmv_0001.png
│   ├── good_0001.png
│   ├── edit_0001.png
│   ├── inpaint_0001.png
│   ├── insert_0001.png
│   └── splice_0001.png
├── train_v2/
│   ├── JPEGImages/
│   └── SegmentationClass/
├── val_v2/
│   ├── JPEGImages/
│   └── SegmentationClass/
├── test_v2/
│   ├── JPEGImages/
│   └── SegmentationClass/
├── train_v2.txt
├── val_v2.txt
└── test_v2.txt
