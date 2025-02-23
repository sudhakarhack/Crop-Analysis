import streamlit as st
import pandas as pd
import joblib
import time
from deep_translator import GoogleTranslator

# Load trained regression model
model = joblib.load("regression_model.pkl")

# Load dataset to get dynamic options
file_path = "crp (1).csv"
df = pd.read_csv(file_path)

# Supported languages
languages = {
    "English": "en",
    "తెలుగు": "te",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
   
}

# Sidebar: Language Selection
selected_language = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()))

def translate(text, target_lang):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        time.sleep(2)  # Small delay before retrying
        return text  # Return original text if translation fails

def t(text):
    return translate(text, languages[selected_language])

st.title(t("🌾 Crop Analysis & Prediction"))
st.write(t("Enter values below to predict Annual Rainfall, Fertilizer Use, and Pesticide Use."))

# Translate crop names and season names dynamically
unique_crops = sorted([t(crop) for crop in df["Crop"].unique().tolist()])
unique_seasons = sorted([t(season.strip()) for season in df["Season"].unique().tolist()])

# User Inputs
crop = st.selectbox(t("🌱 Select Crop"), unique_crops)
season = st.selectbox(t("📅 Select Season"), unique_seasons)
area = st.number_input(t("🌍 Enter Area (in hectares)"), min_value=0.1, format="%.2f")

# Prediction
if st.button(t("🔮 Predict")):
    try:
        # Encode categorical inputs
        crop_encoded = unique_crops.index(crop)
        season_encoded = unique_seasons.index(season)

        # Prepare input DataFrame
        input_data = pd.DataFrame([[crop_encoded, season_encoded, area]], columns=["Crop", "Season", "Area"])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display results
        st.success(f"🌧️ {t('Predicted Rainfall')}: {prediction[0]:.2f} mm")
        st.success(f"🧪 {t('Predicted Pesticide Use')}: {prediction[1]:.2f} kg/ha")
        st.success(f"🌿 {t('Predicted Fertilizer Use')}: {prediction[2]:.2f} kg/ha")

    except Exception as e:
        st.error(f"❌ {t('Error')}: {str(e)}")
