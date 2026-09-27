import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from transforms import train_transform, validation_transform
from model_setup import create_densenet121, create_resnet50


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 3
LEARNING_RATE = 0.0001
BATCH_SIZE = 8

IMAGE_DIR = "data/images"

VALID_DISEASES = [
    "atelectasis",
    "effusion",
    "pneumonia",
    "pneumothorax"
]


# --------------------------------------------------
# Command line argument
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--disease",
    required=True,
    choices=VALID_DISEASES,
    help="Disease to exclude from training"
)

args = parser.parse_args()

disease = args.disease


# --------------------------------------------------
# File locations
# --------------------------------------------------

DATA_DIR = os.path.join(
    "data",
    "unseen",
    disease
)

TRAIN_CSV = os.path.join(
    DATA_DIR,
    "train.csv"
)

VALIDATION_CSV = os.path.join(
    DATA_DIR,
    "validation.csv"
)

RESULTS_DIR = os.path.join(
    "results",
    "unseen_training",
    disease
)

MODEL_DIR = os.path.join(
    "models",
    "unseen",
    disease
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print(" STEP 8 - UNSEEN DISEASE TRAINING")
print("============================================")

print(f"\nHeld-out disease: {disease}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = ChestXrayDataset(
    TRAIN_CSV,
    IMAGE_DIR,
    transform=train_transform
)

validation_dataset = ChestXrayDataset(
    VALIDATION_CSV,
    IMAGE_DIR,
    transform=validation_transform
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

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
# Training function
# --------------------------------------------------

def train_model(model, model_name):

    print("\n============================================")
    print(
        f" TRAINING {model_name} WITHOUT "
        f"{disease.upper()}"
    )
    print("============================================")

    criterion = nn.CrossEntropyLoss()

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
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    best_validation_loss = float("inf")

    start_time = time.time()


    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )


        # ==================================================
        # Training
        # ==================================================

        model.train()

        train_loss_total = 0.0
        train_correct = 0
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

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss_total += (
                loss.item()
                * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_samples += labels.size(0)


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

        train_accuracy = (
            train_correct /
            train_samples
        )


        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        validation_loss_total = 0.0
        validation_correct = 0
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

                _, predictions = torch.max(
                    outputs,
                    1
                )

                validation_correct += (
                    predictions == labels
                ).sum().item()

                validation_samples += (
                    labels.size(0)
                )


        validation_loss = (
            validation_loss_total /
            validation_samples
        )

        validation_accuracy = (
            validation_correct /
            validation_samples
        )


        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        history["epoch"].append(
            epoch + 1
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )


        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        print(
            f"\nTrain Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )


        # --------------------------------------------------
        # Save best model
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
    # Training time
    # --------------------------------------------------

    total_time = (
        time.time() - start_time
    )

    print(
        f"\n{model_name} training time: "
        f"{total_time / 60:.2f} minutes"
    )


    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_training_history.csv"
    )

    history_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Training history saved: "
        f"{csv_path}"
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
        f"{model_name} - "
        f"Without {disease.capitalize()}"
    )

    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        history["epoch"],
        history["validation_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"{model_name} - "
        f"Without {disease.capitalize()}"
    )

    plt.legend()
    plt.grid(True)

    accuracy_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Graphs saved in: "
        f"{RESULTS_DIR}"
    )


# --------------------------------------------------
# DenseNet121
# --------------------------------------------------

print("\nCreating DenseNet121...")

densenet = create_densenet121()

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

print("\nCreating ResNet50...")

resnet = create_resnet50()

train_model(
    resnet,
    "ResNet50"
)

del resnet

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n============================================")
print(" STEP 8 TRAINING COMPLETE")
print("============================================")

print(
    f"\nHeld-out disease: "
    f"{disease}"
)

print(
    f"Models saved in: "
    f"{MODEL_DIR}"
)

print(
    f"Results saved in: "
    f"{RESULTS_DIR}"
)