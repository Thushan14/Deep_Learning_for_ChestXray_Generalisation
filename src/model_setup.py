import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print(" MODEL SETUP")
print("============================================")

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# DenseNet121
# --------------------------------------------------

def create_densenet121():

    print("\nCreating DenseNet121...")

    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    # Number of input features
    input_features = model.classifier.in_features

    # Replace final classifier with 2-class output
    model.classifier = nn.Linear(
        input_features,
        2
    )

    model = model.to(device)

    return model


# --------------------------------------------------
# ResNet50
# --------------------------------------------------

def create_resnet50():

    print("\nCreating ResNet50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Number of input features
    input_features = model.fc.in_features

    # Replace final fully connected layer
    model.fc = nn.Linear(
        input_features,
        2
    )

    model = model.to(device)

    return model


# --------------------------------------------------
# Test model creation
# --------------------------------------------------

if __name__ == "__main__":

    densenet_model = create_densenet121()

    print("\nDenseNet121 final classifier:")
    print(densenet_model.classifier)

    resnet_model = create_resnet50()

    print("\nResNet50 final classifier:")
    print(resnet_model.fc)

    # --------------------------------------------------
    # Test forward pass
    # --------------------------------------------------

    print("\n============================================")
    print(" TESTING MODEL OUTPUT")
    print("============================================")

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)

    densenet_output = densenet_model(dummy_input)

    resnet_output = resnet_model(dummy_input)

    print(
        "\nDenseNet121 output shape:",
        densenet_output.shape
    )

    print(
        "ResNet50 output shape:",
        resnet_output.shape
    )

    if (
        densenet_output.shape == (1, 2)
        and resnet_output.shape == (1, 2)
    ):
        print(
            "\nPASS: Both models are correctly "
            "configured for binary classification."
        )
    else:
        print(
            "\nERROR: Model output shape is incorrect."
        )

    print("\n============================================")
    print(" STEP 5 MODEL SETUP COMPLETE")
    print("============================================")