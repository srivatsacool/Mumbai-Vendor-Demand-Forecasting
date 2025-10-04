
import streamlit as st
from utils import load_model_artifacts, load_or_create_shap_explainer
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainability", layout="wide")
st.title("Explainability • SHAP")

model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()

# Try to load historical CSV if present for background
hist_path = "mumbai_vendors_hourly_20250701_20250930.csv"
hist_df = None
if os.path.exists(hist_path):
    hist_df = pd.read_csv(hist_path)
    st.sidebar.success("Historical data loaded for SHAP background")

explainer, loaded = load_or_create_shap_explainer(model, hist_df, feature_columns)

if explainer is None:
    st.error("Could not create or load SHAP explainer. Ensure shap and joblib are available.")
else:
    st.success(f"SHAP explainer ready (loaded_from_disk={loaded})")
    st.markdown('---')
    st.header('Global feature importance (sample background)')
    try:
        sample = hist_df.sample(n=min(200, len(hist_df))) if hist_df is not None else None
        if sample is not None:
            try:
                X = sample[feature_columns].fillna(0)
            except Exception:
                X = sample.select_dtypes(include=[float,int]).fillna(0)
            sv = explainer(X)
            fig = shap.plots.bar(sv, show=False)
            st.pyplot(bbox_inches='tight')
        else:
            st.info('Provide historical CSV to compute global SHAP summary.')
    except Exception as e:
        st.error(f'Global SHAP failed: {e}')

    st.markdown('---')
    st.header('Local explanation (upload a single-row CSV with feature columns)')
    uploaded = st.file_uploader('Upload CSV (single row)', type=['csv'])
    if uploaded is not None:
        row = pd.read_csv(uploaded)
        try:
            Xr = row[feature_columns].fillna(0)
        except Exception:
            Xr = row.select_dtypes(include=[float,int]).fillna(0)
        try:
            sv = explainer(Xr)
            st.subheader('Waterfall (local)')
            st.pyplot(shap.plots.waterfall(sv[0], show=False))
        except Exception as e:
            st.error(f'Local SHAP failed: {e}')

