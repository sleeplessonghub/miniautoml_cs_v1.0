import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import textwrap as tw

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
    st.error("Uploaded file format must be in either '.csv' or '.xlsx'", icon = '🛑')
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
        st.error("Uploaded file format must be in either '.csv' or '.xlsx'", icon = '🛑')
    except:
      st.error('Potential invalid file ID/name, file potentially not given share access', icon = '🛑')
else:
  st.info('Link a shared Google Drive file of the requested format to begin the analysis', icon = 'ℹ️')

# Guarded execution block (layer 1)
if st.session_state['df_pp'] is not None:

  # Session state variable assignments
  df_pp = st.session_state['df_pp']
  file_name_ref = st.session_state['file_name_ref']

  # Dataset unusable column cleaning
  original_columns = [col for col in df_pp.columns]
  for col in original_columns:
    if col.startswith('Unnamed:') or len(df_pp) == df_pp[col].isna().sum() or df_pp[col].nunique() == 1:
      df_pp.drop(col, axis = 1, inplace = True)
  
  # Dataset column name/object values leading/trailing white space cleaning
  original_columns_2 = [col for col in df_pp.columns]
  for col in original_columns_2:
    if df_pp[col].dtypes == object:
      df_pp[col] = df_pp[col].str.strip()
    if col != col.strip():
      df_pp.rename(columns = {col: col.strip()}, inplace = True)
  
  # Dataset's variable type specification setup
  st.subheader('---- SETUP ----')
  st.write('✅ — Dataset upload and conversion to pandas dataframe complete!')
  st.write('✅ — Dataset unusable column and white space cleaning complete!')
  st.write(f'{file_name_ref} Preview:')
  st.dataframe(df_pp.head())
  st.write(f'⋯ {len(df_pp)} initial rows for analysis!')
  col_names = [col for col in df_pp.columns]
  col_types = []
  id_count = 0
  unassigned_count = 0
  with st.form('data_type_specification_form'):
    st.write(tw.dedent(
        """
        Specify column data type!

        * Specify 'Nominal' for classification target variable
        * Apply 'Identification' labeling only to a single column
        """
    ).strip())
    for col in col_names:
      data_type = st.selectbox(f"'{col}' column data type is:", ('-', 'Identification', 'Float', 'Integer', 'Ordinal', 'Nominal', 'Drop'), accept_new_options = False)
      if data_type == '-':
        unassigned_count = unassigned_count + 1
      if data_type == 'Identification':
        id_count = id_count + 1
      col_types.append(data_type)
    submitted = st.form_submit_button('Confirm type specification')
  if submitted:
    if unassigned_count > 0:
      st.error("Detected at least 1 column without data type specification", icon = '🛑')
    elif id_count >= 2:
      st.error("'Identification' label has been assigned to 2 or more columns", icon = '🛑')
    else:
      st.write('✅ — Dataset variable type specification complete!') # Guarded execution block (layer 2)

      # Random sampling in the case of large population
      if len(df_pp) > 20000:
        df_pp = df_pp.sample(n = 20000, random_state = 42, ignore_index = True)
        st.write('✅ — Large population size random sampling complete!')
        st.write(f'⋯ {len(df_pp)} rows left post-random sampling!')

else:
  st.subheader('No file upload detected')
