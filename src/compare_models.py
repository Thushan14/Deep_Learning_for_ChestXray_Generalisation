import os
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models

from multilabel_dataset import MultiLabelChestXrayDataset
from transforms import test_transform


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 8
THRESHOLD = 0.5
IMAGE_DIR = "data/images"

DISEASES = [
    "atelectasis",
    "effusion",
    "pneumonia",
    "pneumothorax"
]

DISEASE_NAME_MAP = {
    "atelectasis": "Atelectasis",
    "effusion": "Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax"
}

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

RESULTS_DIR = os.path.join(
    "results",
    "model_comparison"
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
print(" STEP 10B - MODEL COMPARISON")
print("============================================")

print(f"\nDevice: {device}")
print(f"Multi-label threshold: {THRESHOLD}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# --------------------------------------------------
# Create multi-label model
# --------------------------------------------------

def create_model(model_name, num_classes):

    if model_name == "DenseNet121":

        model = models.densenet121(
            weights=None
        )

        input_features = (
            model.classifier.in_features
        )

        model.classifier = nn.Linear(
            input_features,
            num_classes
        )

    elif model_name == "ResNet50":

        model = models.resnet50(
            weights=None
        )

        input_features = model.fc.in_features

        model.fc = nn.Linear(
            input_features,
            num_classes
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(device)


# --------------------------------------------------
# Test disease-specific model
# --------------------------------------------------

def test_disease_specific_model(
    model,
    model_name,
    disease,
    test_loader
):

    model.eval()

    total_images = 0
    abnormal_predictions = 0
    normal_predictions = 0

    max_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            )

            # Each disease output becomes 0 or 1
            disease_predictions = (
                probabilities >= THRESHOLD
            )

            # If ANY known disease is detected,
            # classify the X-ray as abnormal
            abnormal = disease_predictions.any(
                dim=1
            )

            abnormal_predictions += (
                abnormal.sum().item()
            )

            normal_predictions += (
                (~abnormal).sum().item()
            )

            total_images += images.size(0)

            # Highest known-disease probability
            # for each image
            batch_max_probabilities = (
                probabilities.max(dim=1)
                .values
                .cpu()
                .numpy()
            )

            max_probabilities.extend(
                batch_max_probabilities
            )


    detection_rate = (
        abnormal_predictions /
        total_images
    )

    average_max_probability = (
        sum(max_probabilities) /
        len(max_probabilities)
    )


    print(
        f"\n{model_name}"
    )

    print(
        f"Total unseen images: "
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
        f"Disease-Specific Detection Rate: "
        f"{detection_rate * 100:.2f}%"
    )

    print(
        f"Average Maximum Known-Disease "
        f"Probability: "
        f"{average_max_probability:.4f}"
    )


    return {
        "Disease":
            DISEASE_NAME_MAP[disease],

        "Model":
            model_name,

        "Disease_Specific_Total":
            total_images,

        "Disease_Specific_Abnormal":
            abnormal_predictions,

        "Disease_Specific_Normal":
            normal_predictions,

        "Disease_Specific_Detection_Rate":
            detection_rate * 100,

        "Average_Max_Known_Disease_Probability":
            average_max_probability
    }


# --------------------------------------------------
# Load Step 9 binary results
# --------------------------------------------------

binary_results_path = os.path.join(
    "results",
    "unseen_testing",
    "unseen_disease_testing_results.csv"
)

binary_results = pd.read_csv(
    binary_results_path
)


# --------------------------------------------------
# Run comparison
# --------------------------------------------------

disease_specific_results = []


for disease in DISEASES:

    held_out_column = (
        DISEASE_NAME_MAP[disease]
    )

    known_diseases = [
        label
        for label in ALL_DISEASES
        if label != held_out_column
    ]

    num_classes = len(
        known_diseases
    )


    print("\n############################################")
    print(
        f" HELD-OUT DISEASE: "
        f"{held_out_column.upper()}"
    )
    print("############################################")

    print(
        f"Known disease outputs: "
        f"{num_classes}"
    )


    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_csv = os.path.join(
        "data",
        "unseen",
        disease,
        "unseen_test.csv"
    )

    test_dataset = (
        MultiLabelChestXrayDataset(
            test_csv,
            IMAGE_DIR,
            known_diseases,
            transform=test_transform
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------
    # DenseNet121
    # --------------------------------------------------

    densenet = create_model(
        "DenseNet121",
        num_classes
    )

    densenet_path = os.path.join(
        "models",
        "disease_specific",
        disease,
        "densenet121_best.pth"
    )

    densenet.load_state_dict(
        torch.load(
            densenet_path,
            map_location=device
        )
    )

    result = test_disease_specific_model(
        densenet,
        "DenseNet121",
        disease,
        test_loader
    )

    disease_specific_results.append(
        result
    )

    del densenet

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # --------------------------------------------------
    # ResNet50
    # --------------------------------------------------

    resnet = create_model(
        "ResNet50",
        num_classes
    )

    resnet_path = os.path.join(
        "models",
        "disease_specific",
        disease,
        "resnet50_best.pth"
    )

    resnet.load_state_dict(
        torch.load(
            resnet_path,
            map_location=device
        )
    )

    result = test_disease_specific_model(
        resnet,
        "ResNet50",
        disease,
        test_loader
    )

    disease_specific_results.append(
        result
    )

    del resnet

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

disease_specific_df = pd.DataFrame(
    disease_specific_results
)


# --------------------------------------------------
# Prepare binary results
# --------------------------------------------------

binary_comparison = binary_results[
    [
        "Disease",
        "Model",
        "Total_Images",
        "Predicted_Abnormal",
        "Predicted_Normal",
        "Detection_Rate_Percentage"
    ]
].copy()

binary_comparison = (
    binary_comparison.rename(
        columns={
            "Total_Images":
                "Binary_Total",

            "Predicted_Abnormal":
                "Binary_Abnormal",

            "Predicted_Normal":
                "Binary_Normal",

            "Detection_Rate_Percentage":
                "Binary_Detection_Rate"
        }
    )
)


# --------------------------------------------------
# Merge
# --------------------------------------------------

comparison = pd.merge(
    binary_comparison,
    disease_specific_df,
    on=[
        "Disease",
        "Model"
    ]
)


# --------------------------------------------------
# Difference
# --------------------------------------------------

comparison[
    "Detection_Rate_Difference"
] = (
    comparison[
        "Binary_Detection_Rate"
    ]
    -
    comparison[
        "Disease_Specific_Detection_Rate"
    ]
)


comparison[
    "Better_Approach"
] = comparison.apply(
    lambda row:
        "Binary"
        if row[
            "Detection_Rate_Difference"
        ] > 0
        else (
            "Disease-Specific"
            if row[
                "Detection_Rate_Difference"
            ] < 0
            else "Equal"
        ),
    axis=1
)


# --------------------------------------------------
# Save comparison
# --------------------------------------------------

output_path = os.path.join(
    RESULTS_DIR,
    "binary_vs_disease_specific.csv"
)

comparison.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# Display final table
# --------------------------------------------------

print("\n============================================")
print(" BINARY VS DISEASE-SPECIFIC COMPARISON")
print("============================================\n")

display_columns = [
    "Disease",
    "Model",
    "Binary_Detection_Rate",
    "Disease_Specific_Detection_Rate",
    "Detection_Rate_Difference",
    "Better_Approach"
]

print(
    comparison[
        display_columns
    ].to_string(
        index=False
    )
)


# --------------------------------------------------
# Average comparison
# --------------------------------------------------

print("\n============================================")
print(" AVERAGE DETECTION RATES")
print("============================================")

for model_name in [
    "DenseNet121",
    "ResNet50"
]:

    model_results = comparison[
        comparison["Model"]
        == model_name
    ]

    binary_average = (
        model_results[
            "Binary_Detection_Rate"
        ].mean()
    )

    disease_specific_average = (
        model_results[
            "Disease_Specific_Detection_Rate"
        ].mean()
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Binary average: "
        f"{binary_average:.2f}%"
    )

    print(
        f"Disease-specific average: "
        f"{disease_specific_average:.2f}%"
    )


print("\n============================================")
print(" STEP 10 MODEL COMPARISON COMPLETE")
print("============================================")

print(
    f"\nResults saved to: "
    f"{output_path}"
)