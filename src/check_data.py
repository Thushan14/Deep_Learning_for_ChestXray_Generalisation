import pandas as pd


TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validation.csv"
TEST_PATH = "data/test.csv"


def check_patient_overlap(train_df, validation_df, test_df):

    train_patients = set(train_df["Patient ID"])
    validation_patients = set(validation_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    train_validation_overlap = train_patients.intersection(
        validation_patients
    )

    train_test_overlap = train_patients.intersection(
        test_patients
    )

    validation_test_overlap = validation_patients.intersection(
        test_patients
    )

    print("\n--- Patient Overlap Check ---")

    print(
        "Train / Validation overlap:",
        len(train_validation_overlap)
    )

    print(
        "Train / Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation / Test overlap:",
        len(validation_test_overlap)
    )

    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    ):
        print("\nPASS: No patient-level data leakage detected.")
    else:
        print("\nWARNING: Patient overlap detected.")


def check_binary_distribution(name, df):

    print(f"\n--- {name} Set ---")

    total = len(df)

    normal = (df["binary_label"] == 0).sum()
    abnormal = (df["binary_label"] == 1).sum()

    print("Total images:", total)
    print("Normal images:", normal)
    print("Abnormal images:", abnormal)

    print(
        "Normal percentage:",
        round((normal / total) * 100, 2),
        "%"
    )

    print(
        "Abnormal percentage:",
        round((abnormal / total) * 100, 2),
        "%"
    )


def main():

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    check_patient_overlap(
        train_df,
        validation_df,
        test_df
    )

    check_binary_distribution(
        "Training",
        train_df
    )

    check_binary_distribution(
        "Validation",
        validation_df
    )

    check_binary_distribution(
        "Testing",
        test_df
    )


if __name__ == "__main__":
    main()
    