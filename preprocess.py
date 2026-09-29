import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "customers.parquet"
OUTPUT_DIR = Path("data/processed")

BENCHMARK_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]

FEATURES = [
    "lifetime_orders",
    "lifetime_spend",
    "average_order_value"
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df):,} customers")
print(f"Columns: {list(df.columns)}")


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = [
    "customer_id",
    "lifetime_orders",
    "lifetime_spend"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------

print("\nChecking missing values...")

print(
    df[
        ["lifetime_orders", "lifetime_spend"]
    ].isnull().sum()
)

df = df.dropna(
    subset=["lifetime_orders", "lifetime_spend"]
).copy()


# ---------------------------------------------------------
# Create Average Order Value
# ---------------------------------------------------------

# For customers with zero orders, AOV is defined as 0.
df["average_order_value"] = np.where(
    df["lifetime_orders"] > 0,
    df["lifetime_spend"] / df["lifetime_orders"],
    0
)


# ---------------------------------------------------------
# Active customer dataset
# ---------------------------------------------------------

active_df = df[df["lifetime_orders"] > 0].copy()

print(
    f"\nActive customers: {len(active_df):,}"
)

print(
    f"Inactive customers: "
    f"{len(df) - len(active_df):,}"
)


# ---------------------------------------------------------
# Feature transformation
# ---------------------------------------------------------

print("\nApplying log1p transformation...")

# log1p(x) = log(1 + x)
# It reduces the effect of extreme values while
# preserving zero values.

for feature in FEATURES:
    df[f"{feature}_log"] = np.log1p(df[feature])
    active_df[f"{feature}_log"] = np.log1p(
        active_df[feature]
    )


LOG_FEATURES = [
    f"{feature}_log"
    for feature in FEATURES
]


# ---------------------------------------------------------
# Standardization
# ---------------------------------------------------------

print("Standardizing features...")

scaler = StandardScaler()

df_scaled = scaler.fit_transform(
    df[LOG_FEATURES]
)

active_scaled = scaler.transform(
    active_df[LOG_FEATURES]
)


# ---------------------------------------------------------
# Create processed feature matrices
# ---------------------------------------------------------

processed_df = pd.DataFrame(
    df_scaled,
    columns=FEATURES
)

processed_active_df = pd.DataFrame(
    active_scaled,
    columns=FEATURES
)


# ---------------------------------------------------------
# Output directories
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"
BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Save complete processed dataset
# ---------------------------------------------------------

all_output = OUTPUT_DIR / "customers_all_scaled.csv"

processed_df.to_csv(
    all_output,
    index=False
)

print(
    f"\nSaved full scaled dataset: "
    f"{all_output}"
)


# ---------------------------------------------------------
# Save active customer dataset
# ---------------------------------------------------------

active_output = (
    OUTPUT_DIR / "customers_active_scaled.csv"
)

processed_active_df.to_csv(
    active_output,
    index=False
)

print(
    f"Saved active customer dataset: "
    f"{active_output}"
)


# ---------------------------------------------------------
# Generate benchmark datasets
# ---------------------------------------------------------

print("\nGenerating benchmark datasets...")

for size in BENCHMARK_SIZES:

    if size > len(processed_df):
        print(
            f"Skipping {size:,}: "
            f"larger than dataset"
        )
        continue

    subset = processed_df.iloc[:size]

    output_file = (
        BENCHMARK_DIR /
        f"customers_{size}.csv"
    )

    subset.to_csv(
        output_file,
        index=False
    )

    print(
        f"  {size:,} rows -> {output_file}"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(
    f"Full dataset: "
    f"{len(processed_df):,} customers"
)

print(
    f"Active dataset: "
    f"{len(processed_active_df):,} customers"
)

print(
    "\nFeatures used for K-Means:"
)

for feature in FEATURES:
    print(f"  - {feature}")

print(
    "\nTransformations:"
)

print("  1. log1p transformation")
print("  2. StandardScaler normalization")

print("\nBenchmark sizes:")

for size in BENCHMARK_SIZES:
    if size <= len(processed_df):
        print(f"  - {size:,}")