import pandas as pd
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
# ==========================================

df = pd.read_csv("AIMLT5\\data\\heart.csv")
####################################################


feature_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

print(
    "Duplicate feature rows:",
    df.duplicated(subset=feature_columns).sum()
)
feature_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

conflicting = (
    df.groupby(feature_columns)['target']
      .nunique()
      .gt(1)
)

print("Conflicting duplicate feature patterns:",
      conflicting.sum())
##############################################
import pandas as pd

df = pd.read_csv("AIMLT5\\data\\heart.csv")

print("Original rows:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())

# Remove exact duplicates
df= df.drop_duplicates()

print("Rows after removing duplicates:", len(df))
print(
    "Rows removed:",
    len(df) - len(df)
)
##############################################


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

print("\nTarget distribution:")
print(y.value_counts())
print("Total rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate percentage:",
      round(df.duplicated().mean() * 100, 2), "%")
duplicates = df[df.duplicated(keep=False)]

print("Duplicate records:", len(duplicates))
SAVE_PATH = "AIMLT5\\outputs\\duplicate_heart.csv"
duplicates.to_csv(SAVE_PATH, index=False)