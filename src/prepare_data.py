from data_utils import (
    DISEASES,
    load_csv,
    get_available_images,
    filter_available_images,
    create_binary_labels,
    create_multilabel_columns
)


CSV_PATH = "data/Data_Entry_2017_v2020.csv"
IMAGE_DIR = "data/images"
OUTPUT_PATH = "data/available_data.csv"


def main():

    print("\n--- NIH Chest X-ray Data Preparation ---\n")

    df = load_csv(CSV_PATH)

    available_images = get_available_images(IMAGE_DIR)

    df = filter_available_images(
        df,
        available_images
    )

    df = create_binary_labels(df)

    print("\nBinary label distribution:")
    print(df["binary_label"].value_counts())

    df = create_multilabel_columns(df)

    print("\nDisease distribution:")

    for disease in DISEASES:
        print(f"{disease}: {df[disease].sum()}")

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nProcessed dataset saved to:")
    print(OUTPUT_PATH)

    print("\nFinal number of images:", len(df))

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()