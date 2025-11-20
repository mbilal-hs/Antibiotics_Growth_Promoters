
import streamlit as st
import pandas as pd
import joblib # Changed from pickle
import numpy as np

# Load the full pipeline
@st.cache_resource
def load_pipeline():
    with open('model.joblib', 'rb') as file: # Changed filename to model.joblib
        pipeline = joblib.load(file) # Changed to joblib.load
    return pipeline

pipeline = load_pipeline()

st.title('Antibiotic Outcome Prediction')
st.write('Predict the outcome based on antibiotic and animal farming parameters.')

# Dynamically get categorical options from the OneHotEncoder in the pipeline
def get_categorical_options_from_pipeline(pipeline):
    preprocessor = pipeline.named_steps['preprocessor']
    onehot_encoder = preprocessor.named_transformers_['cat']

    original_categorical_cols = None
    for name, transformer, cols in preprocessor.transformers:
        if name == 'cat':
            original_categorical_cols = cols
            break

    if not original_categorical_cols:
        raise ValueError("Could not find categorical columns in the pipeline's preprocessor.")

    cat_options = {}
    for i, col_name in enumerate(original_categorical_cols):
        cat_options[col_name] = sorted(list(onehot_encoder.categories_[i]))
    return cat_options

cat_options = get_categorical_options_from_pipeline(pipeline)

with st.form('prediction_form'):
    st.header('Input Features')

    # Numerical Inputs
    dose_g_ton = st.number_input('Dose-g/ton', min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    year = st.number_input('Year', min_value=1990, max_value=2025, value=2010, step=1)

    # Categorical Inputs (using dynamically extracted options)
    breed = st.selectbox('Breed', options=cat_options['Breed'])
    location = st.selectbox('Location', options=cat_options['Location'])
    antibiotic_type = st.selectbox('Antibiotic-type', options=cat_options['Antibiotic-type'])
    feed_type = st.selectbox('Feed-type', options=cat_options['Feed-type'])
    anticoccidial = st.selectbox('Anticoccidial', options=cat_options['Anticoccidial']) # Corrected typo here
    housing = st.selectbox('Housing', options=cat_options['Housing'])

    submitted = st.form_submit_button('Predict Outcome')

if submitted:
    # Create a DataFrame from user inputs (raw data)
    input_data_raw = pd.DataFrame({
        'Dose-g/ton': [dose_g_ton],
        'Year': [year],
        'Breed': [breed],
        'Location': [location],
        'Antibiotic-type': [antibiotic_type],
        'Feed-type': [feed_type],
        'Anticoccidial': [anticoccidial], # Corrected typo here
        'Housing': [housing]
    })

    # Make prediction using the full pipeline (it handles preprocessing internally)
    prediction = pipeline.predict(input_data_raw)

    st.subheader('Prediction Result:')
    st.write(f'The predicted Outcome is: {prediction[0]:.2f}')
