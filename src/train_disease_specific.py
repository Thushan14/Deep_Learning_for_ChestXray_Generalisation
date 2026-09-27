import os
import time
import argparse

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models

from multilabel_dataset import MultiLabelChestXrayDataset
from transforms import train_transform, validation_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 3
LEARNING_RATE = 0.0001
BATCH_SIZE = 8

IMAGE_DIR = "data/images"

ALL_DISEASES = [
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

DISEASE_NAME_MAP = {
    "atelectasis": "Atelectasis",
    "effusion": "Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax"
}


# --------------------------------------------------
# Argument
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--disease",
    required=True,
    choices=list(DISEASE_NAME_MAP.keys())
)

args = parser.parse_args()

held_out_name = args.disease
held_out_column = DISEASE_NAME_MAP[held_out_name]


# --------------------------------------------------
# Remaining known diseases
# --------------------------------------------------

disease_columns = [
    disease
    for disease in ALL_DISEASES
    if disease != held_out_column
]

NUM_CLASSES = len(disease_columns)


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = os.path.join(
    "data",
    "unseen",
    held_out_name
)

TRAIN_CSV = os.path.join(
    DATA_DIR,
    "train.csv"
)

VALIDATION_CSV = os.path.join(
    DATA_DIR,
    "validation.csv"
)

MODEL_DIR = os.path.join(
    "models",
    "disease_specific",
    held_out_name
)

RESULTS_DIR = os.path.join(
    "results",
    "disease_specific_training",
    held_out_name
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print(" STEP 10 - DISEASE-SPECIFIC MODEL TRAINING")
print("============================================")

print(f"\nHeld-out disease: {held_out_column}")
print(f"Known disease outputs: {NUM_CLASSES}")
print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("\nKnown diseases:")

for disease in disease_columns:

    print(f" - {disease}")


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = MultiLabelChestXrayDataset(
    TRAIN_CSV,
    IMAGE_DIR,
    disease_columns,
    transform=train_transform
)

validation_dataset = MultiLabelChestXrayDataset(
    VALIDATION_CSV,
    IMAGE_DIR,
    disease_columns,
    transform=validation_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    f"\nTraining images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(validation_dataset)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)


# --------------------------------------------------
# Calculate class weights
# --------------------------------------------------

train_df = pd.read_csv(
    TRAIN_CSV
)

positive_counts = (
    train_df[disease_columns]
    .sum(axis=0)
)

negative_counts = (
    len(train_df)
    - positive_counts
)

pos_weight_values = (
    negative_counts /
    positive_counts
)

pos_weight = torch.tensor(
    pos_weight_values.values,
    dtype=torch.float32
).to(device)


print("\nPositive class weights:")

for disease, weight in zip(
    disease_columns,
    pos_weight_values
):

    print(
        f"{disease}: {weight:.2f}"
    )


# --------------------------------------------------
# Model creation
# --------------------------------------------------

def create_densenet():

    print("\nCreating DenseNet121...")

    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    input_features = (
        model.classifier.in_features
    )

    model.classifier = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model.to(device)


def create_resnet():

    print("\nCreating ResNet50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model.to(device)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(
    model,
    model_name
):

    print("\n============================================")
    print(
        f" TRAINING {model_name}"
    )
    print(
        f" WITHOUT {held_out_column.upper()}"
    )
    print("============================================")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": []
    }

    best_validation_loss = float(
        "inf"
    )

    start_time = time.time()


    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        # ==================================================
        # Training
        # ==================================================

        model.train()

        train_loss_total = 0.0
        train_samples = 0


        for batch_number, (
            images,
            labels
        ) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()


            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()


            train_loss_total += (
                loss.item()
                * images.size(0)
            )

            train_samples += (
                images.size(0)
            )


            if batch_number % 200 == 0:

                print(
                    f"  Training batch "
                    f"{batch_number}/"
                    f"{len(train_loader)}"
                )


        train_loss = (
            train_loss_total /
            train_samples
        )


        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        validation_loss_total = 0.0
        validation_samples = 0


        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)
                labels = labels.to(device)


                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )


                validation_loss_total += (
                    loss.item()
                    * images.size(0)
                )

                validation_samples += (
                    images.size(0)
                )


        validation_loss = (
            validation_loss_total /
            validation_samples
        )


        history["epoch"].append(
            epoch + 1
        )

        history["train_loss"].append(
            train_loss
        )

        history[
            "validation_loss"
        ].append(
            validation_loss
        )


        print(
            f"\nTrain Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )


        # --------------------------------------------------
        # Best model
        # --------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            model_path = os.path.join(
                MODEL_DIR,
                f"{model_name.lower()}_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"Best model saved: "
                f"{model_path}"
            )


    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print(
        f"\n{model_name} training time: "
        f"{total_time / 60:.2f} minutes"
    )


    # --------------------------------------------------
    # Save history
    # --------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )


    # --------------------------------------------------
    # Save label configuration
    # --------------------------------------------------

    labels_path = os.path.join(
        RESULTS_DIR,
        "known_disease_labels.txt"
    )

    with open(
        labels_path,
        "w"
    ) as file:

        for disease in disease_columns:

            file.write(
                disease + "\n"
            )


    # --------------------------------------------------
    # Loss graph
    # --------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{model_name} Disease-Specific Model\n"
        f"Held-out: {held_out_column}"
    )

    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_loss.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Training history saved: "
        f"{history_path}"
    )

    print(
        f"Loss graph saved: "
        f"{graph_path}"
    )


# --------------------------------------------------
# DenseNet121
# --------------------------------------------------

densenet = create_densenet()

train_model(
    densenet,
    "DenseNet121"
)

del densenet

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# --------------------------------------------------
# ResNet50
# --------------------------------------------------

resnet = create_resnet()

train_model(
    resnet,
    "ResNet50"
)

del resnet

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# --------------------------------------------------
# Complete
# --------------------------------------------------

print("\n============================================")
print(" DISEASE-SPECIFIC TRAINING COMPLETE")
print("============================================")

print(
    f"\nHeld-out disease: "
    f"{held_out_column}"
)

print(
    f"Models saved in: "
    f"{MODEL_DIR}"
)

print(
    f"Results saved in: "
    f"{RESULTS_DIR}"
)