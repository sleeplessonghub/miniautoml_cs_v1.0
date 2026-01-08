import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import textwrap as tw
from sklearn.model_selection import train_test_split

# Title call
st.title('Mini AutoML (Cross-Sectional) v1.0')

# Session state initializations
if 'df_pp' not in st.session_state:
  st.session_state['df_pp'] = None
if 'file_name_ref' not in st.session_state:
  st.session_state['file_name_ref'] = None
if 'submitted_ref' not in st.session_state:
  st.session_state['submitted_ref'] = False

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
  if 'col_names' not in st.session_state:
    st.session_state['col_names'] = None
  col_names = st.session_state['col_names'] = [col for col in df_pp.columns]
  if 'col_types' not in st.session_state:
    st.session_state['col_types'] = None
  col_types = st.session_state['col_types'] = []
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

  if submitted == True:
    submitted_ref = st.session_state['submitted_ref'] = True
  else:
    submitted_ref = st.session_state['submitted_ref'] = False
  
  if submitted_ref == True:
    if unassigned_count > 0:
      st.error("Detected at least 1 column without data type specification", icon = '🛑')
      submitted_ref = False
    elif id_count >= 2:
      st.error("'Identification' label has been assigned to 2 or more columns", icon = '🛑')
      submitted_ref = False
    else:
      st.write('✅ — Dataset variable type specification complete!') # Guarded execution block (layer 2)

      # Random sampling in the case of large population
      if len(df_pp) > 20000:
        df_pp = df_pp.sample(n = 20000, random_state = 42, ignore_index = True)
        st.write('✅ — Large population size random sampling complete!')
        st.write(f'⋯ {len(df_pp)} rows left post-random sampling!')
      
      # Duplicate filtering and irrelevant column dropping
      col_drop_list = []
      for index, value in enumerate(col_types):
        if value == 'Identification':
          df_pp.drop_duplicates(keep = 'first', subset = [col_names[index]], inplace = True)
          df_pp.drop(columns = col_names[index], inplace = True)
          df_pp.reset_index(drop = True, inplace = True)
          col_drop_list.append(col_names[index])
        elif value == 'Drop':
          df_pp.drop(columns = col_names[index], inplace = True)
          col_drop_list.append(col_names[index])
      if col_drop_list:
        st.write('✅ — ID duplicated values removal and targeted column dropping complete!')
        st.write(f"⋯ {len(df_pp)} rows left post-duplicate cleaning and unused column dropping!")
      
      # 'col_names'/'col_types' parallel lists' items update
      for col in col_drop_list:
        col_names.remove(col)
      
      while 'Identification' in col_types:
        col_types.remove('Identification')
      while 'Drop' in col_types:
        col_types.remove('Drop')

      # Train-test sets declaration
      if 'train' not in st.session_state or 'test' not in st.session_state:
        st.session_state['train'] = None
        st.session_state['test'] = None
      train, test = st.session_state['train'], st.session_state['test'] = train_test_split(df_pp, test_size = 0.3, random_state = 42)
      st.write('✅ — Train-test split with a ratio of 70:30 complete!')
      st.write(f'⋯ {len(train)} rows left for training set post-train/test split!')
      st.write(f'⋯ {len(test)} rows left for testing set post-train/test split!')

      # Test output
      st.dataframe(train.head())
      st.write(col_names)
      st.write(col_types)

else:
  st.subheader('No file upload detected')
