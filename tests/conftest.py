import numpy as np
import pytest


@pytest.fixture
def fake_mrnet(tmp_path):
    """Tiny synthetic MRNet-shaped dataset: 6 train exams, 2 valid, random slice counts."""
    rng = np.random.RandomState(0)
    for folder, ids in [("train", range(6)), ("valid", range(1130, 1132))]:
        for p in ["axial", "coronal", "sagittal"]:
            (tmp_path / folder / p).mkdir(parents=True)
            for i in ids:
                n = rng.randint(4, 9)
                np.save(tmp_path / folder / p / f"{i:04d}.npy",
                        rng.randint(0, 255, (n, 256, 256)).astype(np.uint8))
        for lab in ["abnormal", "acl", "meniscus"]:
            rows = "\n".join(f"{i},{rng.randint(0, 2)}" for i in ids)
            (tmp_path / f"{folder}-{lab}.csv").write_text(rows + "\n")
    return tmp_path
