import pandas as pd


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


def print_dataset_summary(name, df):
    print(f"\n--- {name} Summary ---")

    print("Images:", len(df))
    print("Patients:", df["Patient ID"].nunique())

    normal = (df["binary_label"] == 0).sum()
    abnormal = (df["binary_label"] == 1).sum()

    print("Normal:", normal)
    print("Abnormal:", abnormal)

    print("\nDisease counts:")

    for disease in DISEASES:
        print(f"{disease}: {int(df[disease].sum())}")


def main():
    train_df = pd.read_csv("data/train.csv")
    validation_df = pd.read_csv("data/validation.csv")
    test_df = pd.read_csv("data/test.csv")

    print_dataset_summary("Training", train_df)
    print_dataset_summary("Validation", validation_df)
    print_dataset_summary("Testing", test_df)


if __name__ == "__main__":
    main()