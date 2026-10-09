"""
src/download_dataset.py
Automated script to download the real Kaggle MIT-BIH dataset:
"ECG Heartbeat Categorization Dataset" (shayanfazeli/heartbeat).
Places mitbih_train.csv and mitbih_test.csv directly into data/.
"""

import os
import shutil

def download():
    import kagglehub
    print("Downloading shayanfazeli/heartbeat via kagglehub...")
    path = kagglehub.dataset_download("shayanfazeli/heartbeat")
    print(f"Downloaded to cache: {path}")

    os.makedirs("data", exist_ok=True)
    train_src = os.path.join(path, "mitbih_train.csv")
    test_src = os.path.join(path, "mitbih_test.csv")

    shutil.copyfile(train_src, "data/mitbih_train.csv")
    shutil.copyfile(test_src, "data/mitbih_test.csv")
    print("Successfully copied mitbih_train.csv and mitbih_test.csv into data/!")

if __name__ == "__main__":
    download()
