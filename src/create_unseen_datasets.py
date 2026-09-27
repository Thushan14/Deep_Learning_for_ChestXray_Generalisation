import os
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "unseen")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

UNSEEN_DISEASES = [
    "Atelectasis",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
]


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("============================================")
print(" UNSEEN DISEASE DATASET CREATION")
print("============================================")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nOriginal dataset sizes:")
print(f"Training:   {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Testing:    {len(test_df)}")


# --------------------------------------------------
# Create unseen disease datasets
# --------------------------------------------------

for disease in UNSEEN_DISEASES:

    print("\n============================================")
    print(f" HOLDING OUT: {disease}")
    print("============================================")

    # Create output directory for disease
    disease_folder_name = disease.lower().replace(" ", "_")
    disease_output_dir = os.path.join(
        OUTPUT_DIR,
        disease_folder_name
    )

    os.makedirs(disease_output_dir, exist_ok=True)

    # --------------------------------------------------
    # Remove held-out disease from training
    # --------------------------------------------------

    unseen_train_rows = train_df[train_df[disease] == 1]

    filtered_train_df = train_df[
        train_df[disease] == 0
    ].copy()

    # --------------------------------------------------
    # Remove held-out disease from validation
    # --------------------------------------------------

    unseen_validation_rows = validation_df[
        validation_df[disease] == 1
    ]

    filtered_validation_df = validation_df[
        validation_df[disease] == 0
    ].copy()

    # --------------------------------------------------
    # Create unseen disease test set
    # --------------------------------------------------

    unseen_test_df = test_df[
        test_df[disease] == 1
    ].copy()

    # --------------------------------------------------
    # Check that held-out disease is absent
    # --------------------------------------------------

    train_remaining = int(filtered_train_df[disease].sum())
    validation_remaining = int(
        filtered_validation_df[disease].sum()
    )

    if train_remaining != 0:
        raise ValueError(
            f"{disease} is still present in training set."
        )

    if validation_remaining != 0:
        raise ValueError(
            f"{disease} is still present in validation set."
        )

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------

    train_output = os.path.join(
        disease_output_dir,
        "train.csv"
    )

    validation_output = os.path.join(
        disease_output_dir,
        "validation.csv"
    )

    unseen_test_output = os.path.join(
        disease_output_dir,
        "unseen_test.csv"
    )

    filtered_train_df.to_csv(
        train_output,
        index=False
    )

    filtered_validation_df.to_csv(
        validation_output,
        index=False
    )

    unseen_test_df.to_csv(
        unseen_test_output,
        index=False
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print(f"\nRemoved from training:   {len(unseen_train_rows)}")
    print(
        f"Removed from validation: {len(unseen_validation_rows)}"
    )

    print(f"\nNew training size:       {len(filtered_train_df)}")
    print(
        f"New validation size:     {len(filtered_validation_df)}"
    )
    print(f"Unseen test images:      {len(unseen_test_df)}")

    print(
        f"\n{disease} remaining in training: "
        f"{train_remaining}"
    )

    print(
        f"{disease} remaining in validation: "
        f"{validation_remaining}"
    )

    print("\nPASS: Held-out disease successfully removed.")


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n============================================")
print(" STEP 4 COMPLETE")
print("============================================")

print("\nUnseen disease datasets saved in:")
print(OUTPUT_DIR)