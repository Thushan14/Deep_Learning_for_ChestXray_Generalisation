import os
import time
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model_setup import create_densenet121, create_resnet50
from dataloaders import create_dataloaders


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EPOCHS = 3
LEARNING_RATE = 0.0001

RESULTS_DIR = os.path.join("results", "baseline_training")
MODEL_DIR = os.path.join("models", "baseline")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print(" STEP 7 - BASELINE MODEL TRAINING")
print("============================================")

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\nLoading training and validation data...")

train_loader, validation_loader, _ = create_dataloaders()

print(f"Training images: {len(train_loader.dataset)}")
print(f"Validation images: {len(validation_loader.dataset)}")
print(f"Batch size: {train_loader.batch_size}")


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_model(model, model_name):

    print("\n============================================")
    print(f" TRAINING {model_name}")
    print("============================================")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Mixed precision helps reduce GPU memory use
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

    model_start_time = time.time()

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        # ==================================================
        # TRAINING
        # ==================================================

        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_number, (images, labels) in enumerate(
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

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            # Show progress every 200 batches
            if batch_number % 200 == 0:

                print(
                    f"  Training batch "
                    f"{batch_number}/{len(train_loader)}"
                )

        train_loss = (
            running_loss /
            total_samples
        )

        train_accuracy = (
            correct_predictions /
            total_samples
        )

        # ==================================================
        # VALIDATION
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
                    loss.item() * images.size(0)
                )

                _, predictions = torch.max(
                    outputs,
                    1
                )

                validation_correct += (
                    predictions == labels
                ).sum().item()

                validation_samples += labels.size(0)

        validation_loss = (
            validation_loss_total /
            validation_samples
        )

        validation_accuracy = (
            validation_correct /
            validation_samples
        )

        # --------------------------------------------------
        # Save epoch results
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

        print(
            f"\nTrain Loss: {train_loss:.4f}"
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

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            model_path = os.path.join(
                MODEL_DIR,
                f"{model_name.lower()}_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"Best model saved: {model_path}"
            )

    # --------------------------------------------------
    # Training time
    # --------------------------------------------------

    total_time = time.time() - model_start_time

    print(
        f"\n{model_name} training time: "
        f"{total_time / 60:.2f} minutes"
    )

    # --------------------------------------------------
    # Save history CSV
    # --------------------------------------------------

    history_df = pd.DataFrame(history)

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_training_history.csv"
    )

    history_df.to_csv(
        csv_path,
        index=False
    )

    print(f"Training history saved: {csv_path}")

    # --------------------------------------------------
    # Loss graph
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

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
        f"{model_name} - Training and Validation Loss"
    )
    plt.legend()
    plt.grid(True)

    loss_graph = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_loss.png"
    )

    plt.savefig(
        loss_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

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
        f"{model_name} - Training and Validation Accuracy"
    )
    plt.legend()
    plt.grid(True)

    accuracy_graph = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_accuracy.png"
    )

    plt.savefig(
        accuracy_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Loss graph saved: {loss_graph}")
    print(f"Accuracy graph saved: {accuracy_graph}")

    return history


# --------------------------------------------------
# DenseNet121
# --------------------------------------------------

print("\nCreating DenseNet121...")

densenet = create_densenet121()

densenet_history = train_model(
    densenet,
    "DenseNet121"
)

# Free GPU memory before ResNet
del densenet

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# --------------------------------------------------
# ResNet50
# --------------------------------------------------

print("\nCreating ResNet50...")

resnet = create_resnet50()

resnet_history = train_model(
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
print(" STEP 7 BASELINE TRAINING COMPLETE")
print("============================================")

print("\nModels saved in:")
print(MODEL_DIR)

print("\nTraining results saved in:")
print(RESULTS_DIR)