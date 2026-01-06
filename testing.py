# Gemini test

import streamlit as st
import pandas as pd
import gdown
import os

st.title('Mini AutoML (Cross-Sectional) v1.0')

# Source 1: Local Upload
uploaded_file = st.file_uploader("Upload a '.csv' or '.xlsx' file", type=['csv', 'xlsx'])

df_pp = None # Initialize empty dataframe

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_pp = pd.read_csv(uploaded_file)
        else:
            df_pp = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Upload Error: {e}")

st.write('--- OR ---')

# Source 2: Google Drive
# Changed to text_input for alphanumeric IDs
file_id = st.text_input('Input shared Google Drive file ID (e.g., 1-aBcDe...)')
file_name = st.text_input('Input desired filename (including .csv or .xlsx)')

if file_id and file_name:
    if st.button('Download from Google Drive'):
        try:
            file_url = f'https://drive.google.com/uc?id={file_id}'
            # gdown returns the path as a string
            downloaded_path = gdown.download(file_url, file_name, quiet=True)
            
            if downloaded_path:
                if downloaded_path.endswith('.csv'):
                    df_pp = pd.read_csv(downloaded_path)
                else:
                    df_pp = pd.read_excel(downloaded_path)
                st.success(f"Downloaded: {downloaded_path}")
        except Exception as e:
            st.error(f"Google Drive Error: {e}")

# Display result if data exists from either source
if df_pp is not None:
    st.write("### Dataset Preview")
    st.dataframe(df_pp.head())
