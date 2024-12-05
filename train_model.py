import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Step 1: Load synthetic dataset
def create_synthetic_data(num_samples=500):
    print("Creating synthetic dataset...")
    data = {
        "sales_amount": np.random.uniform(10000, 100000, num_samples),
        "purchase_amount": np.random.uniform(5000, 80000, num_samples),
        "tax_slab": np.random.choice([5, 12, 18, 28], num_samples),
        "inflation_rate": np.random.uniform(1.5, 7.0, num_samples),
        "profit_margin": np.random.uniform(0.05, 0.2, num_samples),
        "capital_expenditure": np.random.uniform(50000, 500000, num_samples),
        "revenue_growth": np.random.uniform(0.01, 0.2, num_samples),
        "interest_rate": np.random.uniform(3.0, 10.0, num_samples),
        "gdp_growth_rate": np.random.uniform(1.5, 5.0, num_samples),
        "industry_type": np.random.choice(
            ["Retail", "Manufacturing", "IT Services", "Pharmaceuticals", "Education"], num_samples
        ),
    }

    # Generate target variable (GST liability)
    data["gst_liability"] = (
        data["sales_amount"] * 0.1 +
        data["purchase_amount"] * 0.05 +
        data["tax_slab"] * 100 +
        data["profit_margin"] * 5000 - 
        data["capital_expenditure"] * 0.01 + 
        data["revenue_growth"] * 3000 + 
        data["interest_rate"] * 50 + 
        data["gdp_growth_rate"] * 100
    ) + np.random.normal(0, 100, num_samples)  # Add some noise

    df = pd.DataFrame(data)
    print("Synthetic dataset created successfully.")
    return df

# Step 2: Preprocess data (one-hot encode the industry_type)
def preprocess_data(df):
    print("Preprocessing data...")
    # Convert categorical column 'industry_type' into one-hot encoding
    df = pd.get_dummies(df, columns=["industry_type"], drop_first=True)

    # Split into features (X) and target (y)
    X = df.drop(columns=["gst_liability"])
    y = df["gst_liability"]
    print("Data preprocessing completed.")
    return X, y

# Step 3: Train and save the model
def train_and_save_model(X, y, model_file_path="gst_forecast_model.pkl"):
    print("Training the model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Save the model to a file
    print(f"Saving the model to {model_file_path}...")
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

    # Save the column names used during training
    columns = X.columns
    with open('columns.pkl', 'wb') as f:
        pickle.dump(columns, f)

# Main function
def main():
    print("Starting GST model training...")
    # Create synthetic dataset
    df = create_synthetic_data()

    # Preprocess dataset
    X, y = preprocess_data(df)

    # Train and save the model
    train_and_save_model(X, y)
    print("GST forecasting model training completed!")

if __name__ == "__main__":
    main()
