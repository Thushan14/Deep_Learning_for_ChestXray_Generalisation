from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from transforms import (
    train_transform,
    validation_transform,
    test_transform
)


IMAGE_DIR = "data/images"

TRAIN_CSV = "data/train.csv"
VALIDATION_CSV = "data/validation.csv"
TEST_CSV = "data/test.csv"

BATCH_SIZE = 8


def create_dataloaders():

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

    test_dataset = ChestXrayDataset(
        TEST_CSV,
        IMAGE_DIR,
        transform=test_transform
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


if __name__ == "__main__":

    print("============================================")
    print(" DATALOADER TEST")
    print("============================================")

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Testing batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Labels:", labels)

    if images.shape[1:] == (3, 224, 224):
        print("\nPASS: Image dimensions are correct.")
    else:
        print("\nERROR: Image dimensions are incorrect.")

    print("\n============================================")
    print(" DATALOADER TEST COMPLETE")
    print("============================================")