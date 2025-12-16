# Datasets (DocTamper)

This repository does **not** include the DocTamper dataset files (LMDB or extracted images/masks).

## 1) Download / location

Download DocTamper following the official source/instructions from the dataset authors.

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
