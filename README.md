
# Insurance Pricing Model

A simple predictive model for calculating insurance premiums based on driver and vehicle characteristics. This project demonstrates core actuarial skills: data analysis, statistical modeling, and Python programming.

## Project Structure
```
├── data/ Synthetic data used for training and testing.
├── models/ Python script containing the pricing model.
├── results/ Output graphs and results (gitignored).
├── requirements.txt List of Python dependencies.
└── README.md This file.
```

## How to Run
1.  Clone this repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the model: `python models/pricing_model.py`

## Model Overview
- **Algorithm:** Linear Regression (from scikit-learn)
- **Target Variable:** `total_claims_cost` (The total claim amount a customer is predicted to have)
- **Features:** Age, Gender, Driving Experience, Vehicle Age, Annual Mileage.
- **Output:** The model predicts claims cost and suggests a premium (predicted cost + 20% load).

## Results
The model's performance is measured by:
- **Mean Absolute Error (MAE):** ~$[EL VALOR QUE TE SALIÓ] (on average, how wrong the prediction is in dollars).
- **R² Score:** ~[EL VALOR QUE TE SALIÓ] (how well the model explains the variance in the data).

A plot comparing actual vs. predicted values is saved in the `results` folder.

## Author
Fabián Lizcano - Mathematician and Actuarial Analyst (SOA Exam P Candidate)