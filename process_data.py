import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_excel("irrigation_prediction.csv.xlsx")

print("Original dataset shape:", df.shape)

# Check missing values and duplicates
print("Missing values:", df.isnull().sum().sum())
print("Duplicate records:", df.duplicated().sum())

# Remove duplicate records and rows without a target
df = df.drop_duplicates()
df = df.dropna(subset=["Irrigation_Need"])

# Separate input features and target
X = df.drop(columns=["Irrigation_Need"])
y = df["Irrigation_Need"]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Prepare data processing steps
num_process = Pipeline([
    ("fill", SimpleImputer(strategy="median"))
])

cat_process = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_process, num_cols),
    ("cat", cat_process, cat_cols)
])

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Process the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
print("Data preprocessing completed successfully!")
