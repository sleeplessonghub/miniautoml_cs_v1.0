import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# Title call
st.title('Mini AutoML (Cross-Sectional) v1.0')

# Session state initializations
if 'df_pp' not in st.session_state:
  st.session_state['df_pp'] = None
if 'file_name_ref' not in st.session_state:
  st.session_state['file_name_ref'] = None

# Dataset upload and conversion to a pandas dataframe
uploaded_file = st.file_uploader("Upload a '.csv' or '.xlsx' file", type = ['csv', 'xlsx'], accept_multiple_files = False)
if uploaded_file:
  try:
    if uploaded_file.name.endswith('.csv'):
      st.session_state['df_pp'] = pd.read_csv(uploaded_file)
      st.session_state['file_name_ref'] = uploaded_file.name
    elif uploaded_file.name.endswith('.xlsx'):
      st.session_state['df_pp'] = pd.read_excel(uploaded_file)
      st.session_state['file_name_ref'] = uploaded_file.name
  except:
    st.error("Uploaded file format must be in either '.csv' or '.xlsx'")
else:
  st.info('Upload a file of the requested format from local to begin the analysis', icon = 'ℹ️')

st.write('OR')

file_id = st.text_input('Input shared Google Drive file ID (e.g. 1Fq32N3GU...)')
file_name = st.text_input("Input shared Google Drive file name (including '.csv'/'.xlsx' extension)")
if file_id and file_name:
  if st.button('Download shared file'):
    file_url = f'https://drive.google.com/uc?id={file_id}'
    try:
      uploaded_file = gdown.download(file_url, file_name, quiet = True)
      try:
        if file_name.endswith('.csv'):
          st.session_state['df_pp'] = pd.read_csv(uploaded_file)
          st.session_state['file_name_ref'] = file_name
        elif file_name.endswith('.xlsx'):
          st.session_state['df_pp'] = pd.read_excel(uploaded_file)
          st.session_state['file_name_ref'] = file_name
      except:
        st.error("Uploaded file format must be in either '.csv' or '.xlsx'")
    except:
      st.error('Potential invalid file ID/name, file potentially not given share access')
else:
  st.info('Link a shared Google Drive file of the requested format to begin the analysis', icon = 'ℹ️')

# Guarded execution block
if st.session_state['df_pp'] is not None:

  # Session state variable assignments
  df_pp = st.session_state['df_pp']
  file_name_ref = st.session_state['file_name_ref']

  # Uploaded file preview
  st.write(f"'{file_name_ref}' Data Preview:")
  st.dataframe(df_pp.head())
else:
  st.subheader('No file upload detected')
