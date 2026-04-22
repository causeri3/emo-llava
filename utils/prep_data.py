import numpy as np
import random
import os
import pandas as pd
from PIL import Image
from datasets import Dataset
import kagglehub
import shutil
from pathlib import Path
from utils.clean_data import CSV_LABEL_PATH

DATASET_DIR = Path("./dataset")

try:
    kagglehub.dataset_download("magdawjcicka/emotic",
                               output_dir=DATASET_DIR)
except FileExistsError:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        kagglehub.dataset_download("magdawjcicka/emotic",
                                   output_dir=DATASET_DIR)

IMG_DIR = DATASET_DIR / "img_arrs"
CSV_FILES = [
    DATASET_DIR / "annots_arrs" / "annot_arrs_train.csv",
    DATASET_DIR / "annots_arrs" / "annot_arrs_val.csv",
    DATASET_DIR / "annots_arrs" / "annot_arrs_test.csv"
]

def get_labelled_emotions(row_idx:int, df:pd.DataFrame) -> str:
    emotions = []
    for column_name, value in df.loc[row_idx].iloc[8:-6].items():
        if value == 1.0:
            emotions.append(column_name)
    return ", ".join(emotions)

def data_one_file(file:str, df:pd.DataFrame) -> (Image, str):
    arr = np.load(IMG_DIR / file)
    img = Image.fromarray(arr)
    row_mask = (df["Crop_name"] == file)
    if not row_mask.any():
        print(f"Not found in df: {file}")
        return
    row_idx = df.index[row_mask][0]
    emo_string = get_labelled_emotions(row_idx, df)
    return img, emo_string

def split_dataset(dataset: Dataset) -> (Dataset, Dataset, Dataset):
    first_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = first_split["train"]
    temp_data = first_split["test"]
    second_split = temp_data.train_test_split(test_size=0.5, seed=42)
    eval_data = second_split["train"]
    test_data = second_split["test"]
    return train_data, eval_data, test_data


def get_sample_data_set(sample_perc:float = 0.005) -> (Dataset, Dataset, Dataset):
    df = pd.read_csv(CSV_FILES[0])
    files = [f for f in os.listdir(IMG_DIR) if f.endswith('.npy') and 'crop' and 'train' in f]
    sample_size = int(sample_perc * len(files))
    files_sample = random.sample(files, sample_size)
    df_small = df[df["Crop_name"].isin(files_sample)]
    results = [data_one_file(file, df_small) for file in files_sample]

    data_dict = {
        'image': [result[0] for result in results], # size(224,224)
        'label': [result[1] for result in results]
    }

    dataset = Dataset.from_dict(data_dict)
    train_data, eval_data, test_data = split_dataset(dataset)
    return train_data, eval_data, test_data


def get_clean_data():
    df_clean = pd.read_csv(CSV_LABEL_PATH)
    df_clean_clean = df_clean[df_clean['emotions_fit'] + df_clean['has_face'] > 0]
    dfs = {}
    for csv_path in CSV_FILES:
        name = csv_path.stem
        df = pd.read_csv(csv_path)
        df_filtered = df[df["Crop_name"].isin(df_clean_clean["file"])].copy()
        results = [data_one_file(file, df_filtered) for file in df_filtered["Crop_name"]]

        data_dict = {
            'image': [result[0] for result in results],
            'label': [result[1] for result in results]
        }

        dataset = Dataset.from_dict(data_dict)
        dfs[name] = dataset


    return list(dfs.values())