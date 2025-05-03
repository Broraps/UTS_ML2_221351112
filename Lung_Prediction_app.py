import streamlit as st
import numpy as np
import tensorflow as tf

# ----------------- Load TFLite Model -----------------
interpreter = tf.lite.Interpreter(model_path="Lung_prediction.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------- Title -----------------
st.set_page_config(page_title="Lung Cancer Risk Prediction", layout="centered")
st.markdown("## 🫁 Lung Cancer Risk Prediction")
st.markdown("Selamat datang! Masukkan data Anda di bawah ini untuk memprediksi level risiko kanker paru-paru.")

# ----------------- User Form -----------------
with st.form("prediction_form"):
    st.markdown("### 📋 Informasi Dasar")
    age = st.number_input("Age", min_value=14, max_value=73, value=30)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_encoded = 1 if gender == "Male" else 2

    features = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards',
                'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity',
                'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood',
                'Fatigue', 'Weight Loss', 'Wheezing']

    feature_ranges = {
        'Air Pollution': (1, 8), 'Alcohol use': (1, 8), 'Dust Allergy': (1, 8),
        'OccuPational Hazards': (1, 8), 'Genetic Risk': (1, 7), 'chronic Lung Disease': (1, 7),
        'Balanced Diet': (1, 7), 'Obesity': (1, 7), 'Smoking': (1, 8),
        'Passive Smoker': (1, 8), 'Chest Pain': (1, 9), 'Coughing of Blood': (1, 9),
        'Fatigue': (1, 9), 'Weight Loss': (1, 8), 'Wheezing': (1, 8)
    }

    col1, col2 = st.columns(2)
    user_input = [age]

    st.markdown("### ⚙️ Kondisi & Riwayat Kesehatan")
    for i, feature in enumerate(features):
        min_val, max_val = feature_ranges[feature]
        default_val = (min_val + max_val) // 2
        if i % 2 == 0:
            with col1:
                value = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default_val)
        else:
            with col2:
                value = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default_val)
        user_input.append(value)

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

    # Styling hasil prediksi
    risk_level = label_names[pred_label]
    color_map = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
    desc_map = {
        "Low": "Risiko rendah. Tetap jaga gaya hidup sehat!",
        "Medium": "Risiko sedang. Perhatikan gejala dan konsultasi rutin.",
        "High": "Risiko tinggi. Disarankan untuk periksa lebih lanjut!"
    }

    st.markdown("---")
    st.markdown(f"### 🎯 Hasil Prediksi: **{color_map[risk_level]} {risk_level} Risk**")
    st.info(desc_map[risk_level])
