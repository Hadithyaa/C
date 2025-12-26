import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("iris.csv")

# Preview data
print(df.head())

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode species column
le = LabelEncoder()
df["species_encoded"] = le.fit_transform(df["species"])

print("\nCleaned Iris dataset head:")
print(df.head())

# Split features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Print shapes
print("Training features shape :", X_train.shape)
print("Testing features shape :", X_test.shape)
print("Training labels shape :", y_train.shape)
print("Testing labels shape :", y_test.shape)
