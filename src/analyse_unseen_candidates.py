import os
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data"
RESULTS_DIR = os.path.join("results", "unseen_disease_analysis")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

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
    "Hernia",
]


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("============================================")
print(" UNSEEN DISEASE CANDIDATE ANALYSIS")
print("============================================")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDatasets loaded successfully.")

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(validation_df)}")
print(f"Testing images: {len(test_df)}")


# --------------------------------------------------
# Check disease columns
# --------------------------------------------------

missing_columns = [
    disease
    for disease in DISEASES
    if disease not in train_df.columns
]

if missing_columns:
    print("\nERROR: Some disease columns are missing:")
    print(missing_columns)
    raise SystemExit

print("\nPASS: All 14 disease columns are available.")


# --------------------------------------------------
# Function to analyse each disease
# --------------------------------------------------

def analyse_disease(disease, train_df, validation_df, test_df):

    # Number of images containing this disease
    train_count = int(train_df[disease].sum())
    validation_count = int(validation_df[disease].sum())
    test_count = int(test_df[disease].sum())

    # Count how many diseases are present in each image
    train_disease_count = train_df[DISEASES].sum(axis=1)
    validation_disease_count = validation_df[DISEASES].sum(axis=1)
    test_disease_count = test_df[DISEASES].sum(axis=1)

    # Single-disease images:
    # selected disease = 1 AND total disease count = 1
    train_single = int(
        ((train_df[disease] == 1) & (train_disease_count == 1)).sum()
    )

    validation_single = int(
        ((validation_df[disease] == 1) &
         (validation_disease_count == 1)).sum()
    )

    test_single = int(
        ((test_df[disease] == 1) & (test_disease_count == 1)).sum()
    )

    # Multi-disease images:
    # selected disease = 1 AND more than one disease is present
    train_multi = int(
        ((train_df[disease] == 1) & (train_disease_count > 1)).sum()
    )

    validation_multi = int(
        ((validation_df[disease] == 1) &
         (validation_disease_count > 1)).sum()
    )

    test_multi = int(
        ((test_df[disease] == 1) & (test_disease_count > 1)).sum()
    )

    return {
        "Disease": disease,

        "Train_Total": train_count,
        "Train_Single": train_single,
        "Train_Multi": train_multi,

        "Validation_Total": validation_count,
        "Validation_Single": validation_single,
        "Validation_Multi": validation_multi,

        "Test_Total": test_count,
        "Test_Single": test_single,
        "Test_Multi": test_multi,
    }


# --------------------------------------------------
# Analyse all diseases
# --------------------------------------------------

results = []

for disease in DISEASES:
    result = analyse_disease(
        disease,
        train_df,
        validation_df,
        test_df
    )

    results.append(result)


results_df = pd.DataFrame(results)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n============================================")
print(" DISEASE DISTRIBUTION ANALYSIS")
print("============================================\n")

print(results_df.to_string(index=False))


# --------------------------------------------------
# Candidate disease summary
# --------------------------------------------------

candidate_diseases = [
    "Atelectasis",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
]

print("\n============================================")
print(" CURRENT CANDIDATE UNSEEN DISEASES")
print("============================================")

for disease in candidate_diseases:

    row = results_df[
        results_df["Disease"] == disease
    ].iloc[0]

    print(f"\n{disease}")
    print(f"  Training total:      {row['Train_Total']}")
    print(f"  Training single:     {row['Train_Single']}")
    print(f"  Training multi:      {row['Train_Multi']}")
    print(f"  Validation total:    {row['Validation_Total']}")
    print(f"  Testing total:       {row['Test_Total']}")
    print(f"  Testing single:      {row['Test_Single']}")
    print(f"  Testing multi:       {row['Test_Multi']}")


# --------------------------------------------------
# Save results
# --------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

output_file = os.path.join(
    RESULTS_DIR,
    "unseen_disease_candidate_analysis.csv"
)

results_df.to_csv(output_file, index=False)

print("\n============================================")
print(" ANALYSIS COMPLETE")
print("============================================")

print("\nResults saved to:")
print(output_file)