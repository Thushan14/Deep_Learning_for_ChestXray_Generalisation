import os
import pandas as pd


DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    print("Total rows in original CSV:", len(df))
    return df


def get_available_images(image_dir):
    images = {
        filename
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(".png")
    }

    print("PNG images available locally:", len(images))
    return images


def filter_available_images(df, available_images):
    filtered_df = df[
        df["Image Index"].isin(available_images)
    ].copy()

    print("Matching CSV records:", len(filtered_df))
    return filtered_df


def create_binary_labels(df):
    df["binary_label"] = df["Finding Labels"].apply(
        lambda label: 0 if label == "No Finding" else 1
    )

    return df


def create_multilabel_columns(df):
    for disease in DISEASES:
        df[disease] = df["Finding Labels"].apply(
            lambda labels: 1
            if disease in str(labels).split("|")
            else 0
        )

    return df