import matplotlib.pyplot as plt
import torch

from dataloaders import create_dataloaders


# --------------------------------------------------
# Image denormalisation
# --------------------------------------------------

def denormalize(image):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image * std + mean

    return torch.clamp(image, 0, 1)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("============================================")
print(" DATA VISUALISATION")
print("============================================")

train_loader, _, _ = create_dataloaders()

images, labels = next(iter(train_loader))


# --------------------------------------------------
# Display first 8 images
# --------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

for i, ax in enumerate(axes.flat):

    image = denormalize(images[i])

    # Convert tensor:
    # [C, H, W] -> [H, W, C]
    image = image.permute(1, 2, 0).numpy()

    label = labels[i].item()

    if label == 0:
        label_name = "Normal"
    else:
        label_name = "Abnormal"

    ax.imshow(image)
    ax.set_title(
        f"Label: {label} ({label_name})"
    )
    ax.axis("off")


plt.tight_layout()

output_path = (
    "results/data_processing/"
    "training_sample_visualisation.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nSample image saved to:")
print(output_path)

print("\nPASS: Images and labels visualised successfully.")

print("\n============================================")
print(" STEP 6 VISUALISATION COMPLETE")
print("============================================")

plt.show()