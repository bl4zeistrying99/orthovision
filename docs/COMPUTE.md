# Training compute (no local GPU needed)

**Kaggle (recommended):** free T4/P100, ~30 GPU hrs/week per account, persistent datasets.
Upload MRNet as a private Kaggle Dataset once; every teammate attaches it. Clone the repo in a
notebook cell: `!git clone <repo-url> && cd orthovision && pip install -q -e . grad-cam timm`.

**Colab:** mount Drive, put data on Drive, same clone/install cell. Sessions disconnect, so
checkpoint every epoch to Drive and make training resumable.

Tips: 4 people = 4 free accounts. Split model training (one encoder each) so the four
encoders train in parallel. Use `configs/local.yaml` to point `paths.data_root` at the
Kaggle input path (e.g. `/kaggle/input/mrnet/MRNet-v1.0`).
