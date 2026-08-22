import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

IMAGE_DIR = "data/images"

TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validation.csv"
TEST_PATH = "data/test.csv"

RESULTS_DIR = "results/data_processing"


# ---------------------------------------------------------
# Disease labels used in NIH ChestX-ray14
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

def load_datasets():

    print("\n--- Loading Processed Datasets ---\n")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Training images:", len(train_df))
    print("Validation images:", len(validation_df))
    print("Testing images:", len(test_df))

    return train_df, validation_df, test_df


# ---------------------------------------------------------
# Check patient overlap
# ---------------------------------------------------------

def check_patient_overlap(train_df, validation_df, test_df):

    print("\n--- Patient Leakage Check ---")

    train_patients = set(train_df["Patient ID"])
    validation_patients = set(validation_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    train_validation = train_patients & validation_patients
    train_test = train_patients & test_patients
    validation_test = validation_patients & test_patients

    print("Train / Validation overlap:", len(train_validation))
    print("Train / Test overlap:", len(train_test))
    print("Validation / Test overlap:", len(validation_test))

    if (
        len(train_validation) == 0
        and len(train_test) == 0
        and len(validation_test) == 0
    ):
        print("PASS: No patient-level data leakage.")
        return True

    print("FAIL: Patient overlap detected.")
    return False


# ---------------------------------------------------------
# Check whether image files actually exist
# ---------------------------------------------------------

def check_missing_images(name, df):

    missing_images = []

    for image_name in df["Image Index"]:

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        if not os.path.isfile(image_path):
            missing_images.append(image_name)

    print(f"{name} missing images:", len(missing_images))

    return missing_images


# ---------------------------------------------------------
# Check duplicate image records
# ---------------------------------------------------------

def check_duplicates(name, df):

    duplicates = df["Image Index"].duplicated().sum()

    print(f"{name} duplicate image records:", duplicates)

    return duplicates


# ---------------------------------------------------------
# Check binary labels
# ---------------------------------------------------------

def check_binary_labels(name, df):

    valid_labels = {0, 1}

    actual_labels = set(
        df["binary_label"].dropna().unique()
    )

    invalid_labels = actual_labels - valid_labels

    if len(invalid_labels) == 0:
        print(f"{name} binary labels: PASS")
        return True

    print(
        f"{name} binary labels: FAIL",
        invalid_labels
    )

    return False


# ---------------------------------------------------------
# Check disease columns
# ---------------------------------------------------------

def check_disease_columns(df):

    print("\n--- Disease Column Check ---")

    missing_columns = [
        disease
        for disease in DISEASES
        if disease not in df.columns
    ]

    if len(missing_columns) == 0:

        print("PASS: All 14 disease columns are available.")

        return True

    print("Missing disease columns:")
    print(missing_columns)

    return False


# ---------------------------------------------------------
# Check disease representation
# ---------------------------------------------------------

def check_disease_representation(
    train_df,
    validation_df,
    test_df
):

    print("\n--- Disease Representation Check ---")

    all_present = True

    for disease in DISEASES:

        train_count = int(train_df[disease].sum())
        validation_count = int(validation_df[disease].sum())
        test_count = int(test_df[disease].sum())

        print(
            f"{disease}: "
            f"Train={train_count}, "
            f"Validation={validation_count}, "
            f"Test={test_count}"
        )

        if (
            train_count == 0
            or validation_count == 0
            or test_count == 0
        ):
            all_present = False

    if all_present:
        print(
            "\nPASS: Every disease appears "
            "in all three datasets."
        )
    else:
        print(
            "\nWARNING: At least one disease "
            "is missing from a dataset split."
        )

    return all_present


# ---------------------------------------------------------
# Create dataset summary
# ---------------------------------------------------------

def create_dataset_summary(
    train_df,
    validation_df,
    test_df
):

    print("\n--- Creating Dataset Summary ---")

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df
    }

    rows = []

    for name, df in datasets.items():

        total = len(df)

        normal = int(
            (df["binary_label"] == 0).sum()
        )

        abnormal = int(
            (df["binary_label"] == 1).sum()
        )

        patients = df["Patient ID"].nunique()

        rows.append({
            "Dataset": name,
            "Images": total,
            "Patients": patients,
            "Normal": normal,
            "Abnormal": abnormal,
            "Normal_Percentage":
                round(normal / total * 100, 2),
            "Abnormal_Percentage":
                round(abnormal / total * 100, 2)
        })

    summary_df = pd.DataFrame(rows)

    output_path = os.path.join(
        RESULTS_DIR,
        "dataset_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False
    )

    print(summary_df)

    print("\nSaved:")
    print(output_path)

    return summary_df


# ---------------------------------------------------------
# Create disease distribution table
# ---------------------------------------------------------

def create_disease_distribution(
    train_df,
    validation_df,
    test_df
):

    print("\n--- Creating Disease Distribution ---")

    rows = []

    for disease in DISEASES:

        rows.append({
            "Disease": disease,
            "Training": int(
                train_df[disease].sum()
            ),
            "Validation": int(
                validation_df[disease].sum()
            ),
            "Testing": int(
                test_df[disease].sum()
            )
        })

    disease_df = pd.DataFrame(rows)

    output_path = os.path.join(
        RESULTS_DIR,
        "disease_distribution.csv"
    )

    disease_df.to_csv(
        output_path,
        index=False
    )

    print(disease_df)

    print("\nSaved:")
    print(output_path)

    return disease_df


# ---------------------------------------------------------
# Plot binary class distribution
# ---------------------------------------------------------

def plot_binary_distribution(
    train_df,
    validation_df,
    test_df
):

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df
    }

    rows = []

    for name, df in datasets.items():

        rows.append({
            "Dataset": name,
            "Normal": int(
                (df["binary_label"] == 0).sum()
            ),
            "Abnormal": int(
                (df["binary_label"] == 1).sum()
            )
        })

    plot_df = pd.DataFrame(rows)

    plot_df.set_index(
        "Dataset"
    )[["Normal", "Abnormal"]].plot(
        kind="bar"
    )

    plt.title(
        "Normal and Abnormal Image Distribution"
    )

    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Images")

    plt.xticks(rotation=0)

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "binary_class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("\nSaved:")
    print(output_path)


# ---------------------------------------------------------
# Plot disease distribution
# ---------------------------------------------------------

def plot_disease_distribution(disease_df):

    plot_df = disease_df.set_index(
        "Disease"
    )

    plot_df.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title(
        "Disease Distribution Across Dataset Splits"
    )

    plt.xlabel("Disease")
    plt.ylabel("Number of Images")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "disease_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(output_path)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print(
        "\n===================================="
    )
    print(
        " FINAL DATA PROCESSING VALIDATION"
    )
    print(
        "===================================="
    )

    train_df, validation_df, test_df = load_datasets()

    # Patient leakage
    patient_check = check_patient_overlap(
        train_df,
        validation_df,
        test_df
    )

    # Missing images
    print("\n--- Missing Image Check ---")

    train_missing = check_missing_images(
        "Training",
        train_df
    )

    validation_missing = check_missing_images(
        "Validation",
        validation_df
    )

    test_missing = check_missing_images(
        "Testing",
        test_df
    )

    # Duplicates
    print("\n--- Duplicate Check ---")

    train_duplicates = check_duplicates(
        "Training",
        train_df
    )

    validation_duplicates = check_duplicates(
        "Validation",
        validation_df
    )

    test_duplicates = check_duplicates(
        "Testing",
        test_df
    )

    # Binary labels
    print("\n--- Binary Label Check ---")

    binary_train = check_binary_labels(
        "Training",
        train_df
    )

    binary_validation = check_binary_labels(
        "Validation",
        validation_df
    )

    binary_test = check_binary_labels(
        "Testing",
        test_df
    )

    # Disease columns
    disease_column_check = check_disease_columns(
        train_df
    )

    # Disease representation
    disease_representation = (
        check_disease_representation(
            train_df,
            validation_df,
            test_df
        )
    )

    # Summary files
    create_dataset_summary(
        train_df,
        validation_df,
        test_df
    )

    disease_df = create_disease_distribution(
        train_df,
        validation_df,
        test_df
    )

    # Figures
    plot_binary_distribution(
        train_df,
        validation_df,
        test_df
    )

    plot_disease_distribution(
        disease_df
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    all_checks_passed = (
        patient_check
        and len(train_missing) == 0
        and len(validation_missing) == 0
        and len(test_missing) == 0
        and train_duplicates == 0
        and validation_duplicates == 0
        and test_duplicates == 0
        and binary_train
        and binary_validation
        and binary_test
        and disease_column_check
        and disease_representation
    )

    print(
        "\n===================================="
    )

    if all_checks_passed:

        print(
            "PASS: DATA PROCESSING COMPLETE"
        )

    else:

        print(
            "WARNING: CHECK RESULTS ABOVE"
        )

    print(
        "====================================\n"
    )


if __name__ == "__main__":
    main()