from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained decision tree model
model_path = os.path.join(os.path.dirname(__file__), "../decision_tree_model.pkl")
with open(model_path, "rb") as file:
    decision_tree_model = pickle.load(file)


@app.route("/")
def home():
    # Render the home page
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        user_input = request.form

        print(user_input)

        age = float(user_input["age"])
        annualIncome = float(user_input["annualIncome"])
        delayFromDueDate = float(user_input["delayFromDueDate"])
        numDelayedPayment = float(user_input["numDelayedPayment"])
        numCreditInquiries = float(user_input["numCreditInquiries"])
        creditMix = user_input["creditMix"]
        outstandingDebt = float(user_input["outstandingDebt"])
        creditUtilizationRatio = float(user_input["creditUtilizationRatio"])
        totalEMI = float(user_input["totalEMI"])
        creditAgeYears = float(user_input["creditAgeYears"])
        paymentMinAmount = user_input["paymentMinAmount"]

        credit_mix_val = 0
        paymentMinAmount_yes = 0
        paymentMinAmount_No = 0
        paymentMinAmount_NM = 0

        # set paymentMinAmount
        if paymentMinAmount == "no":
            paymentMinAmount_No = 1
        elif paymentMinAmount == "yes":
            paymentMinAmount_yes = 1
        elif paymentMinAmount == "not_mention":
            paymentMinAmount_NM = 1

        # set credit mix
        if creditMix == "Good":
            credit_mix_val = 1
        elif creditMix == "Standard":
            credit_mix_val = 2
        else:
            credit_mix_val = 0

        featured_values = [
            age,
            annualIncome,
            delayFromDueDate,
            numDelayedPayment,
            numCreditInquiries,
            credit_mix_val,
            outstandingDebt,
            creditUtilizationRatio,
            totalEMI,
            creditAgeYears,
            paymentMinAmount_yes,
            paymentMinAmount_No,
            paymentMinAmount_NM,
        ]

        print(featured_values)

        # Prepare data for prediction
        features = np.array([featured_values])
        prediction = decision_tree_model.predict(features)
        print(prediction[0])

        # Render the output page with the prediction result
        return render_template("output.html", score=int(prediction[0]))

    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
