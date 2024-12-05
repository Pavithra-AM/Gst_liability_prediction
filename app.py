from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model_file_path = "gst_forecast_model.pkl"
columns_file_path = "columns.pkl"

try:
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    # Load columns used during training
    with open(columns_file_path, "rb") as f:
        feature_columns = pickle.load(f)

except EOFError:
    raise Exception("Error: The model file or columns file is empty or corrupted. Please train the model again.")

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle form submission and prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        sales_amount = float(request.form["sales_amount"])
        purchase_amount = float(request.form["purchase_amount"])
        tax_slab = float(request.form["tax_slab"])
        inflation_rate = float(request.form["inflation_rate"])
        profit_margin = float(request.form["profit_margin"])
        capital_expenditure = float(request.form["capital_expenditure"])
        revenue_growth = float(request.form["revenue_growth"])
        interest_rate = float(request.form["interest_rate"])
        gdp_growth_rate = float(request.form["gdp_growth_rate"])
        industry_type = request.form["industry_type"]

        # Prepare input data (one-hot encoding for industry type)
        industries = ["Manufacturing", "IT Services", "Pharmaceuticals", "Education", "Retail"]
        industry_data = {f"industry_type_{ind}": 0 for ind in industries}
        if f"industry_type_{industry_type}" in industry_data:
            industry_data[f"industry_type_{industry_type}"] = 1

        # Prepare the user data
        user_data = {
            "sales_amount": [sales_amount],
            "purchase_amount": [purchase_amount],
            "tax_slab": [tax_slab],
            "inflation_rate": [inflation_rate],
            "profit_margin": [profit_margin],
            "capital_expenditure": [capital_expenditure],
            "revenue_growth": [revenue_growth],
            "interest_rate": [interest_rate],
            "gdp_growth_rate": [gdp_growth_rate],
            **industry_data,
        }

        # Create DataFrame and ensure columns match the feature columns
        user_df = pd.DataFrame(user_data).reindex(columns=feature_columns, fill_value=0)

        # Predict the GST liability
        prediction = model.predict(user_df)[0]
        prediction = round(prediction, 2)

        # Render result with formatted prediction
        return render_template("index.html", prediction=f"₹{prediction:,.2f}")

    except Exception as e:
        # If there's any error, show the error message
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
