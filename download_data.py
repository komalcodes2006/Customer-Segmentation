import pandas as pd

URL = "https://huggingface.co/datasets/ziadatalabs/FreeEcommerceTwin14M/resolve/main/customers.parquet"

print("Downloading customer dataset...")

df = pd.read_parquet(URL)

print(f"Loaded {len(df):,} customers")

df.to_parquet("customers.parquet", index=False)

print("Saved as customers.parquet")