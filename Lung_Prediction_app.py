import streamlit as st
import numpy as np
import tensorflow as tf

# ----------------- Load TFLite Model -----------------
interpreter = tf.lite.Interpreter(model_path="Lung_prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------- Setup Streamlit -----------------
st.set_page_config(page_title="Lung Cancer Risk Prediction", layout="centered")
st.markdown("## 🫁 Lung Cancer Risk Prediction")
st.markdown("Selamat datang! Masukkan data Anda di bawah ini untuk memprediksi level risiko kanker paru-paru.")

# ----------------- Fitur & Rentang Asli Dataset -----------------
features = ['Air Pollution', 'Alcohol use', 'Dust Allergy',
   'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
   'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
   'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
   'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
   'Frequent Cold', 'Dry Cough', 'Snoring']

feature_ranges = {
    'Air Pollution' : (1,8), 'Alcohol use' : (1,8), 'Dust Allergy' : (1,8),
   'OccuPational Hazards' : (1,8), 'Genetic Risk' : (1,7), 'chronic Lung Disease' : (1,7),
   'Balanced Diet' : (1,7), 'Obesity' : (1,7), 'Smoking' : (1,8), 'Passive Smoker' : (1,8), 'Chest Pain' : (1,9),
   'Coughing of Blood' : (1,9), 'Fatigue' : (1,9), 'Weight Loss' : (1,8), 'Shortness of Breath' : (1,9),
   'Wheezing' : (1,8), 'Swallowing Difficulty' : (1,8), 'Clubbing of Finger Nails' : (1,9),
   'Frequent Cold' : (1,7), 'Dry Cough' : (1,7), 'Snoring' : (1,7)
}

# ----------------- Input Form -----------------
with st.form("prediction_form"):
    st.markdown("### 📋 Informasi Dasar")
    age = st.number_input("Age", min_value=14, max_value=100, value=30)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_encoded = 1 if gender == "Male" else 2

    st.markdown("### ⚙️ Kondisi & Riwayat Kesehatan")
    col1, col2 = st.columns(2)
    user_input = [age]

    for i, feature in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            input_val_0_10 = st.slider(f"{feature}", min_value=0, max_value=10, value=5)
            min_old, max_old = feature_ranges[feature]
            # Mapping dari skala 0–10 ke rentang asli
            mapped_val = min_old + (input_val_0_10 / 10) * (max_old - min_old)
            user_input.append(mapped_val)

    user_input.append(gender_encoded)
    submitted = st.form_submit_button("🔍 Prediksi Risiko")

# ----------------- Prediction -----------------
if submitted:
    input_data = np.array(user_input, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    pred_label = np.argmax(prediction)
    label_names = ['Low', 'Medium', 'High']

    # Tampilan hasil
    risk_level = label_names[pred_label]
    color_map = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
    desc_map = {
        "Low": "Risiko rendah. Tetap jaga gaya hidup sehat! 💪",
        "Medium": "Risiko sedang. Perhatikan gejala dan konsultasi rutin. 🩺",
        "High": "Risiko tinggi. Disarankan untuk periksa lebih lanjut! ⚠️"
    }

    st.markdown("---")
    st.markdown(f"### 🎯 Hasil Prediksi: **{color_map[risk_level]} {risk_level} Risk**")
    st.info(desc_map[risk_level])
