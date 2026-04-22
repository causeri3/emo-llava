import numpy as np
import os
import pandas as pd
from PIL import Image
import kagglehub
import shutil
from pathlib import Path
from transformers import pipeline
import time

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
    DATASET_DIR / "annots_arrs" / "annot_arrs_test.csv",
    DATASET_DIR / "annots_arrs" / "annot_arrs_val.csv"
]

CSV_LABEL_PATH = DATASET_DIR / "cleaned_labels.csv"

pipe = pipeline("image-text-to-text", model="llava-1.5-7b-hf",  device="mps" )

def prompt_to_binary(prompt, img):
    binary = ''
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",
                 "text": prompt},
            ],
        },
    ]

    start = time.perf_counter()
    out = pipe(text=messages, max_new_tokens=5)
    end = time.perf_counter()
    print(f"Time taken: {end - start:.3f} seconds")
    #print(out)
    answer = out[0]['generated_text'][1]['content'].lower()
    if "no" in answer:
        binary = 0
    if "yes" in answer:
        binary = 1
    return binary


def create_csv(files):
    if os.path.exists(CSV_LABEL_PATH):
        df = pd.read_csv(CSV_LABEL_PATH)
    else:
        df = pd.DataFrame({"file": files})

    for col in binaries:
        if col not in df.columns:
            df[col] = pd.NA

    done_mask = df[binaries].notna().all(axis=1)
    done_files = set(df.loc[done_mask, "file"])
    return done_mask, done_files, df



def combine_dfs(csv_file_paths):
    df = pd.DataFrame()
    for file_path in csv_file_paths:
        df_add = pd.read_csv(file_path)
        df = pd.concat([df, df_add], ignore_index=True)
    return df

df_annotations = combine_dfs(CSV_FILES)

def row_of_one_file(file):
    #print(file)
    row_mask = (df_annotations["Crop_name"] == file)
    if not row_mask.any():
        print(f"Not found in df: {file}")
        return
    row_idx = df_annotations.index[row_mask][0]
    return row_idx

def get_labelled_emotions(row_idx):
    emotions = []
    for column_name, value in df.loc[row_idx].iloc[8:-6].items():
        if value == 1.0:
            emotions.append(column_name)
    return ", ".join(emotions)


binaries = [
    "has_face",
    # "multiple_faces",
    # "emotions_fit"
]
prompt_face = "Can you see a whole face in this picture? More than just the back of a head. Answer with yes or no."
prompt_mult_faces = "Can you see multiple faces in this picture, more than one? Answer with yes or no."

def get_emotions_prompt(file):
    row_idx = row_of_one_file(file)
    return "does this face display all the following emotions: {}? Answer with yes or no.".format(
        get_labelled_emotions(row_idx))


files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.npy') and 'crop' in f])

done_mask, done_files, df = create_csv(files)

print('Number of files done: ', len(done_files))

def main():
    for i, file in enumerate(files, 1):
        if file in done_files:
            print(f"Already Done: {i}/{len(files)}: {file}")
            continue

        print(f"{i}/{len(files)}: {file}")
        prompts = [
            prompt_face,
            # prompt_mult_faces,
            # get_emotions_prompt(file)
                ]

        arr = np.load(os.path.join(IMG_DIR, file))
        img = Image.fromarray(arr)
        row_idx = df.index[df["file"] == file][0]

        for prompt, col in zip(prompts, binaries):
            df.loc[row_idx, col] = prompt_to_binary(prompt, img)

        df.to_csv(CSV_LABEL_PATH, index=False)

if __name__ == "__main__":
    main()
