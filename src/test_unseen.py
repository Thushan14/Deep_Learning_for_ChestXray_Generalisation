import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from transforms import test_transform
from model_setup import create_densenet121, create_resnet50


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 8
IMAGE_DIR = "data/images"

DISEASES = [
    "atelectasis",
    "effusion",
    "pneumonia",
    "pneumothorax"
]

RESULTS_DIR = os.path.join(
    "results",
    "unseen_testing"
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
print(" STEP 9 - UNSEEN DISEASE TESTING")
print("============================================")

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# --------------------------------------------------
# Testing function
# --------------------------------------------------

def test_model(
    model,
    model_name,
    disease,
    test_loader
):

    print("\n============================================")
    print(
        f" TESTING {model_name} ON "
        f"{disease.upper()}"
    )
    print("============================================")

    model.eval()

    total_images = 0
    abnormal_predictions = 0
    normal_predictions = 0

    probabilities = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = softmax(outputs)

            predictions = torch.argmax(
                probs,
                dim=1
            )

            abnormal_predictions += (
                predictions == 1
            ).sum().item()

            normal_predictions += (
                predictions == 0
            ).sum().item()

            total_images += images.size(0)

            abnormal_probs = (
                probs[:, 1]
                .detach()
                .cpu()
                .numpy()
            )

            probabilities.extend(
                abnormal_probs
            )


    # --------------------------------------------------
    # Unseen disease detection rate
    # --------------------------------------------------

    detection_rate = (
        abnormal_predictions /
        total_images
    )

    average_abnormal_probability = (
        sum(probabilities) /
        len(probabilities)
    )


    print(
        f"\nTotal unseen disease images: "
        f"{total_images}"
    )

    print(
        f"Predicted Abnormal: "
        f"{abnormal_predictions}"
    )

    print(
        f"Predicted Normal: "
        f"{normal_predictions}"
    )

    print(
        f"Unseen Disease Detection Rate: "
        f"{detection_rate * 100:.2f}%"
    )

    print(
        f"Average Abnormal Probability: "
        f"{average_abnormal_probability:.4f}"
    )


    return {
        "Disease": disease.capitalize(),
        "Model": model_name,
        "Total_Images": total_images,
        "Predicted_Abnormal": abnormal_predictions,
        "Predicted_Normal": normal_predictions,
        "Detection_Rate": detection_rate,
        "Detection_Rate_Percentage":
            detection_rate * 100,
        "Average_Abnormal_Probability":
            average_abnormal_probability
    }


# --------------------------------------------------
# Run tests
# --------------------------------------------------

all_results = []


for disease in DISEASES:

    print("\n############################################")
    print(
        f" HELD-OUT DISEASE: {disease.upper()}"
    )
    print("############################################")


    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_csv = os.path.join(
        "data",
        "unseen",
        disease,
        "unseen_test.csv"
    )

    test_dataset = ChestXrayDataset(
        test_csv,
        IMAGE_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    print(
        f"\nUnseen test images: "
        f"{len(test_dataset)}"
    )


    # ==================================================
    # DenseNet121
    # ==================================================

    densenet = create_densenet121()

    densenet_path = os.path.join(
        "models",
        "unseen",
        disease,
        "densenet121_best.pth"
    )

    densenet.load_state_dict(
        torch.load(
            densenet_path,
            map_location=device
        )
    )

    densenet = densenet.to(device)

    densenet_result = test_model(
        densenet,
        "DenseNet121",
        disease,
        test_loader
    )

    all_results.append(
        densenet_result
    )

    del densenet

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ==================================================
    # ResNet50
    # ==================================================

    resnet = create_resnet50()

    resnet_path = os.path.join(
        "models",
        "unseen",
        disease,
        "resnet50_best.pth"
    )

    resnet.load_state_dict(
        torch.load(
            resnet_path,
            map_location=device
        )
    )

    resnet = resnet.to(device)

    resnet_result = test_model(
        resnet,
        "ResNet50",
        disease,
        test_loader
    )

    all_results.append(
        resnet_result
    )

    del resnet

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------
# Save summary CSV
# --------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

output_file = os.path.join(
    RESULTS_DIR,
    "unseen_disease_testing_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Display summary
# --------------------------------------------------

print("\n============================================")
print(" UNSEEN DISEASE TESTING SUMMARY")
print("============================================\n")

summary_columns = [
    "Disease",
    "Model",
    "Total_Images",
    "Predicted_Abnormal",
    "Predicted_Normal",
    "Detection_Rate_Percentage",
    "Average_Abnormal_Probability"
]

print(
    results_df[
        summary_columns
    ].to_string(
        index=False
    )
)


print("\n============================================")
print(" STEP 9 TESTING COMPLETE")
print("============================================")

print("\nResults saved to:")
print(output_file)