import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/available_data.csv"

TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validation.csv"
TEST_PATH = "data/test.csv"


def main():

    print("\n--- Dataset Splitting ---\n")

    df = pd.read_csv(INPUT_PATH)

    print("Total images:", len(df))
    print("Total patients:", df["Patient ID"].nunique())

    # Get unique patients
    patients = df["Patient ID"].unique()

    # 70% training, 30% temporary
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.30,
        random_state=42
    )

    # Split remaining 30% equally:
    # 15% validation, 15% testing
    validation_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=42
    )

    train_df = df[
        df["Patient ID"].isin(train_patients)
    ].copy()

    validation_df = df[
        df["Patient ID"].isin(validation_patients)
    ].copy()

    test_df = df[
        df["Patient ID"].isin(test_patients)
    ].copy()

    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nTraining images:", len(train_df))
    print("Validation images:", len(validation_df))
    print("Testing images:", len(test_df))

    print("\nTraining patients:", train_df["Patient ID"].nunique())
    print("Validation patients:", validation_df["Patient ID"].nunique())
    print("Testing patients:", test_df["Patient ID"].nunique())

    print("\nFiles saved:")
    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)

    print("\nDataset splitting completed successfully.")


if __name__ == "__main__":
    main()