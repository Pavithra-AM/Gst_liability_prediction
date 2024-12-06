from flask import Flask, request, render_template, redirect, url_for
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the GST forecasting model
model_file_path = "gst_forecast_model.pkl"
columns_file_path = "columns.pkl"
try:
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)
    with open(columns_file_path, "rb") as f:
        feature_columns = pickle.load(f)
except Exception as e:
    print("Error loading model or columns:", e)

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle forecasting
@app.route("/forecasting", methods=["GET", "POST"])
def forecasting():
    if request.method == "POST":
        try:
            # Parse form inputs
            sales_amount = float(request.form.get("sales_amount", 0))
            purchase_amount = float(request.form.get("purchase_amount", 0))
            tax_slab = float(request.form.get("tax_slab", 0))
            inflation_rate = float(request.form.get("inflation_rate", 0))
            profit_margin = float(request.form.get("profit_margin", 0))
            capital_expenditure = float(request.form.get("capital_expenditure", 0))
            revenue_growth = float(request.form.get("revenue_growth", 0))
            interest_rate = float(request.form.get("interest_rate", 0))
            gdp_growth_rate = float(request.form.get("gdp_growth_rate", 0))
            industry_type = request.form.get("industry_type", "Manufacturing")

            # Prepare input for the model
            input_data = pd.DataFrame([{
                "SalesAmount": sales_amount,
                "PurchaseAmount": purchase_amount,
                "TaxSlab": tax_slab,
                "InflationRate": inflation_rate,
                "ProfitMargin": profit_margin,
                "CapitalExpenditure": capital_expenditure,
                "RevenueGrowth": revenue_growth,
                "InterestRate": interest_rate,
                "GDPGrowthRate": gdp_growth_rate,
                "IndustryType": industry_type
            }])

            # Ensure feature columns are aligned
            input_data = input_data.reindex(columns=feature_columns, fill_value=0)

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Render result
            return render_template("forecasting.html", prediction=prediction, error=None)
        except Exception as e:
            return render_template("forecasting.html", prediction=None, error=f"Error: {e}")
    return render_template("forecasting.html", prediction=None, error=None)

# Route to handle fraudulent transaction detection
@app.route("/fraudulent", methods=["GET", "POST"])
def fraudulent():
    if request.method == "POST":
        file = request.files.get("file")

        # Check if file is uploaded
        if not file or file.filename == "":
            return render_template("fraudulent.html", result=None, error="No file chosen. Please upload a valid file.", fraud_count=None)

        try:
            # Save the uploaded file to the server
            uploads_folder = "uploads"
            os.makedirs(uploads_folder, exist_ok=True)
            file_path = os.path.join(uploads_folder, file.filename)
            file.save(file_path)

            # Log the file path for debugging
            print(f"File saved to: {file_path}")

            # Read the file based on its extension
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                return render_template("fraudulent.html", result=None, error="Unsupported file format. Please upload a CSV or Excel file.", fraud_count=None)

            # Log the dataframe for debugging
            print(f"Dataframe head: {df.head()}")

            # Validate if the file contains data
            if df.empty:
                return render_template("fraudulent.html", result=None, error="The uploaded file is empty. Please upload a valid file.", fraud_count=None)

            # Analyze fraud and count fraudulent transactions
            fraud_result = analyze_fraud(df)
            fraud_count = len(fraud_result)

            # Log the fraud result for debugging
            print(f"Fraud result: {fraud_result}")

            # Return the analysis result and count to the template
            return render_template("fraudulent.html", result=fraud_result.to_dict(orient="records"), error=None, fraud_count=fraud_count)

        except ValueError as ve:
            # Handle specific validation errors
            return render_template("fraudulent.html", result=None, error=str(ve), fraud_count=None)
        except Exception as e:
            # Handle general errors
            return render_template("fraudulent.html", result=None, error=f"Error processing file: {e}", fraud_count=None)
    
    return render_template("fraudulent.html", result=None, error=None, fraud_count=None)


# Function to analyze fraudulent transactions
def analyze_fraud(df):
    # Check if 'Fraudulent' column exists in the dataset
    fraud_column_name = 'Fraudulent'  # The column name for fraud indicator
    if fraud_column_name not in df.columns:
        raise ValueError(f"The uploaded file is missing the '{fraud_column_name}' column.")

    # Filter fraudulent transactions (where Fraudulent == 1)
    fraud_transactions = df[df[fraud_column_name] == 1]

    # If no fraudulent transactions, return an empty dataframe
    if fraud_transactions.empty:
        return pd.DataFrame()

    # Return the fraudulent transactions (only the relevant columns)
    return fraud_transactions[["InvoiceID", "InvoiceAmount", "InvoiceText"]]


if __name__ == "__main__":
    app.run(debug=True)
