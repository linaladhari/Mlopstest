import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    df = df.dropna()

    # Separate features and target
    y = df["species"]
    X = df.drop("species", axis=1)

    # Identify types of columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols)
        ]
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Convert back to DataFrame
    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )

    # Add back species column
    X_df["species"] = y.values

    # Save processed data
    X_df.to_csv(output_path, index=False)


def main():
    preprocess_data("data/raw/iris.csv", "data/processed/iris.csv")


if __name__ == "__main__":
    main()
