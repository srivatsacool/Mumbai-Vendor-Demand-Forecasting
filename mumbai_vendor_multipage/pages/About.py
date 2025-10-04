
import streamlit as st
from utils import load_model_artifacts
st.set_page_config(page_title="About", layout="wide")
st.title("About this Multipage Project")

try:
    model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()
    st.write('Model Type:', model_type)
    st.write('Metadata:', metadata)
    st.write('Feature count (expected):', len(feature_columns) if feature_columns is not None else 'Unknown')
except Exception as e:
    st.warning('Model artifacts not found in working directory. Place rf_model.pkl, feature_columns.pkl etc. in the same folder as this app.')

st.markdown('---')
st.markdown('**How to run**: unzip the project, install requirements from requirements.txt, then `streamlit run app.py`')
