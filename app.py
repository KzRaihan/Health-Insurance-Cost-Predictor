# ================================
# Health Insurance Cost Predictor (Gradio App)
# ================================

import gradio as gr
import pandas as pd
import pickle

# -------------------------------
# 1. Load Trained Pipeline
# -------------------------------
with open("insurance_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# 2. Prediction Logic with Validation
# -------------------------------
def cost_predictor(age, sex, bmi, children, smoker, region):
    
    # 🔐 Input Validation
    if age is None or age <= 0 or age > 100:
        return "❌ Age must be between 1 and 100."

    if bmi is None or bmi < 10 or bmi > 60:
        return "❌ BMI must be between 10 and 60."

    if sex not in ["male", "female"]:
        return "❌ Invalid gender selection."

    if smoker not in ["yes", "no"]:
        return "❌ Invalid smoker status."

    if region not in ["southwest", "southeast", "northeast", "northwest"]:
        return "❌ Invalid region selection."

    # Create input DataFrame
    input_df = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]],
        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    )

    # Prediction
    prediction = model.predict(input_df)[0]

    return f"✅ Predicted Health Insurance Cost: ${prediction:,.2f}"

# -------------------------------
# 3. Gradio Interface
# -------------------------------
inputs = [
    gr.Number(label="Age", value=18),
    gr.Radio(["male", "female"], label="Gender"),
    gr.Number(label="BMI", value=30.66),
    gr.Slider(0, 5, step=1, label="Number of Children"),
    gr.Radio(["yes", "no"], label="Smoker Status"),
    gr.Dropdown(
        ["southwest", "southeast", "northeast", "northwest"],
        label="Residential Area (US)"
    )
]

app = gr.Interface(
    fn=cost_predictor,
    inputs=inputs,
    outputs="text",
    title="🏥 Health Insurance Cost Predictor",
    description="Predict annual medical insurance cost using a Gradient Boosting model"
)

# -------------------------------
# 4. Launch App
# -------------------------------
if __name__ == "__main__":
    app.launch(share=True)
