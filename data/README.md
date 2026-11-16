# MRNet data

Source: Stanford ML Group MRNet (Bien et al., 2018), ~5.7 GB. Request access via the Stanford AIMI
shared-datasets page (registration + terms); check the dataset licence before re-uploading it
anywhere (e.g. a private Kaggle Dataset for the team only). Do not commit it.

Expected layout under `data/raw/MRNet-v1.0/`:
```
train/{axial,coronal,sagittal}/0000.npy ... 1129.npy   each (slices, 256, 256) uint8
valid/{axial,coronal,sagittal}/1130.npy ... 1249.npy
train-abnormal.csv  train-acl.csv  train-meniscus.csv   (no header: exam_id,label)
valid-abnormal.csv  valid-acl.csv  valid-meniscus.csv
```
Official sizes: 1130 train, 120 valid. The 120-exam test set is hidden by Stanford, so this
project uses an exam-level val split from `train` for tuning and the official `valid` as the
held-out test set. State this in the report.

Verify: `python scripts/verify_data.py`
