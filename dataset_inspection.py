import pandas as pd

# Load dataset
df = pd.read_excel("Online Retail.xlsx")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== UNIQUE CUSTOMERS =====")
print(df["CustomerID"].nunique())

print("\n===== UNIQUE COUNTRIES =====")
print(df["Country"].nunique())

print("\n===== QUANTITY STATISTICS =====")
print(df["Quantity"].describe())

print("\n===== UNIT PRICE STATISTICS =====")
print(df["UnitPrice"].describe())

print("\n===== DATE RANGE =====")
print(df["InvoiceDate"].min())
print(df["InvoiceDate"].max())

print("\n===== NEGATIVE QUANTITIES =====")
print((df["Quantity"] < 0).sum())

print("\n===== ZERO PRICES =====")
print((df["UnitPrice"] == 0).sum())

print("\n===== MISSING CUSTOMER IDs =====")
print(df["CustomerID"].isna().sum())