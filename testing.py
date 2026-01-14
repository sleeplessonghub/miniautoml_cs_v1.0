# Dependency imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap as tw
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, TargetEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import lightgbm as lgbm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, classification_report
import dalex as dx
import plotly.graph_objects as go

# Title call
st.title('Mini AutoML (Cross-Sectional) v1.0')

# Session state layer initializations
if 'df_pp' not in st.session_state:
  st.session_state['df_pp'] = None # Layer 1 check
if 'submitted_ref' not in st.session_state:
  st.session_state['submitted_ref'] = False # Layer 2 check
if 'submitted_2_ref' not in st.session_state:
  st.session_state['submitted_2_ref'] = False # Layer 3 check

# Dataset upload and conversion to a pandas dataframe
uploaded_file = st.file_uploader("Upload a '.csv' or '.xlsx' file", type = ['csv', 'xlsx'], accept_multiple_files = False)
if uploaded_file:
  try:
    if uploaded_file.name.endswith('.csv'):
      st.session_state['df_pp'] = pd.read_csv(uploaded_file)
      file_name = st.session_state['file_name'] = uploaded_file.name # To be used for new data check for ML (.csv)
    elif uploaded_file.name.endswith('.xlsx'):
      st.session_state['df_pp'] = pd.read_excel(uploaded_file)
      file_name = st.session_state['file_name'] = uploaded_file.name # To be used for new data check for ML (.xlsx)
  except:
    st.error("Uploaded file format must be in either '.csv' or '.xlsx'!", icon = '🛑')
    st.stop()
  st.warning('Do not delete the uploaded file during analysis!', icon = '🚧')
else:
  st.info('Upload a file of the requested format from your device to begin the analysis!', icon = 'ℹ️')

# Guarded execution block (layer 1)
if st.session_state['df_pp'] is not None:

  # Session state variable assignment
  df_pp = st.session_state['df_pp']

  # Dataset unusable column cleaning
  original_columns = [col for col in df_pp.columns]
  for col in original_columns:
    if col.startswith('Unnamed:') or len(df_pp) == df_pp[col].isna().sum() or df_pp[col].nunique() == 1:
      df_pp.drop(col, axis = 1, inplace = True)

  # Dataset column name/object values leading/trailing white space cleaning
  original_columns_2 = [col for col in df_pp.columns]
  for col in original_columns_2:
    if df_pp[col].dtypes == object:
      df_pp[col] = df_pp[col].astype('string') # Bug fix, sometimes boolean can't pass as string
      df_pp[col] = df_pp[col].str.strip()
      df_pp[col] = df_pp[col].astype(object)
      df_pp[col] = df_pp[col].replace(pd.NA, np.nan)
    if col != col.strip():
      df_pp.rename(columns = {col: col.strip()}, inplace = True)

  # Dataset variable type specification
  st.header('⸻ Setup Wizard 🪄')
  st.write('✅ — Dataset upload and conversion to a pandas dataframe complete!')
  st.write('✅ — Dataset unusable column and white space cleaning complete!')
  st.write('Dataset Preview:')
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
    submitted_ref = st.session_state['submitted_ref']

  if submitted_ref == True:
    if unassigned_count > 0:
      st.error('Detected at least 1 column without data type specification!', icon = '🛑')
      submitted_ref = False
    elif id_count >= 2:
      st.error("'Identification' label has been assigned to 2 or more columns!", icon = '🛑')
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
        st.write(f'⋯ {len(df_pp)} rows left post-duplicate cleaning and unused column dropping!')

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

      # Dataset cell missingness handling
      placeholder = 9898989898 # Placeholder value, unlikely to naturally occur in real-world datasets
      df_pp_rows_w_nan = df_pp.isna().any(axis = 1).sum()
      percent_missing = df_pp_rows_w_nan / len(df_pp)
      central_tend_dict = dict()

      if percent_missing <= 0.05:

        train.dropna(inplace = True)
        for index, value in enumerate(col_types):
          if value == 'Float':
            train[col_names[index]] = train[col_names[index]].astype(str)
            train[col_names[index]] = train[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            train[col_names[index]] = train[col_names[index]].replace('', placeholder)
            train[col_names[index]] = train[col_names[index]].astype(float)
            central_tend_dict[col_names[index]] = train[col_names[index]].median()
            if placeholder in train[col_names[index]].values:
              train[col_names[index]] = train[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Integer':
            train[col_names[index]] = train[col_names[index]].astype(str)
            train[col_names[index]] = train[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            train[col_names[index]] = train[col_names[index]].replace('', placeholder)
            train[col_names[index]] = train[col_names[index]].astype(float)
            train[col_names[index]] = train[col_names[index]].astype(int)
            central_tend_dict[col_names[index]] = int(train[col_names[index]].median())
            if placeholder in train[col_names[index]].values:
              train[col_names[index]] = train[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Ordinal':
            train[col_names[index]] = train[col_names[index]].astype(str)
            central_tend_dict[col_names[index]] = train[col_names[index]].mode()[0]
            train[col_names[index]] = train[col_names[index]].replace('', central_tend_dict.get(col_names[index]))
          elif value == 'Nominal':
            train[col_names[index]] = train[col_names[index]].astype(str)
            central_tend_dict[col_names[index]] = train[col_names[index]].mode()[0]
            train[col_names[index]] = train[col_names[index]].replace('', central_tend_dict.get(col_names[index]))
        
        test.dropna(inplace = True)
        for index, value in enumerate(col_types):
          if value == 'Float':
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            test[col_names[index]] = test[col_names[index]].replace('', placeholder)
            test[col_names[index]] = test[col_names[index]].astype(float)
            if placeholder in test[col_names[index]].values:
              test[col_names[index]] = test[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Integer':
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            test[col_names[index]] = test[col_names[index]].replace('', placeholder)
            test[col_names[index]] = test[col_names[index]].astype(float)
            test[col_names[index]] = test[col_names[index]].astype(int)
            if placeholder in test[col_names[index]].values:
              test[col_names[index]] = test[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Ordinal':
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].replace('', central_tend_dict.get(col_names[index]))
          elif value == 'Nominal':
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].replace('', central_tend_dict.get(col_names[index]))
      
      else:

        for index, value in enumerate(col_types):
          if value == 'Float':
            train[col_names[index]] = train[col_names[index]].fillna(placeholder)
            train[col_names[index]] = train[col_names[index]].astype(str)
            train[col_names[index]] = train[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            train[col_names[index]] = train[col_names[index]].replace('', placeholder)
            train[col_names[index]] = train[col_names[index]].astype(float)
            central_tend_dict[col_names[index]] = train[col_names[index]].median()
            if placeholder in train[col_names[index]].values:
              train[col_names[index]] = train[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Integer':
            train[col_names[index]] = train[col_names[index]].fillna(placeholder)
            train[col_names[index]] = train[col_names[index]].astype(str)
            train[col_names[index]] = train[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            train[col_names[index]] = train[col_names[index]].replace('', placeholder)
            train[col_names[index]] = train[col_names[index]].astype(float)
            train[col_names[index]] = train[col_names[index]].astype(int)
            central_tend_dict[col_names[index]] = int(train[col_names[index]].median())
            if placeholder in train[col_names[index]].values:
              train[col_names[index]] = train[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Ordinal':
            train[col_names[index]] = train[col_names[index]].astype(str)
            central_tend_dict[col_names[index]] = train[col_names[index]].mode()[0]
            train[col_names[index]] = train[col_names[index]].fillna(train[col_names[index]].mode()[0])
            train[col_names[index]] = train[col_names[index]].replace('', train[col_names[index]].mode()[0])
          elif value == 'Nominal':
            train[col_names[index]] = train[col_names[index]].astype(str)
            central_tend_dict[col_names[index]] = train[col_names[index]].mode()[0]
            train[col_names[index]] = train[col_names[index]].fillna(train[col_names[index]].mode()[0])
            train[col_names[index]] = train[col_names[index]].replace('', train[col_names[index]].mode()[0])
        
        for index, value in enumerate(col_types):
          if value == 'Float':
            test[col_names[index]] = test[col_names[index]].fillna(placeholder)
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            test[col_names[index]] = test[col_names[index]].replace('', placeholder)
            test[col_names[index]] = test[col_names[index]].astype(float)
            if placeholder in test[col_names[index]].values:
              test[col_names[index]] = test[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Integer':
            test[col_names[index]] = test[col_names[index]].fillna(placeholder)
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].str.replace(r'[^\d.-]', '', regex = True)
            test[col_names[index]] = test[col_names[index]].replace('', placeholder)
            test[col_names[index]] = test[col_names[index]].astype(float)
            test[col_names[index]] = test[col_names[index]].astype(int)
            if placeholder in test[col_names[index]].values:
              test[col_names[index]] = test[col_names[index]].replace(placeholder, central_tend_dict.get(col_names[index]))
          elif value == 'Ordinal':
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].fillna(central_tend_dict.get(col_names[index]))
            test[col_names[index]] = test[col_names[index]].replace('', central_tend_dict.get(col_names[index]))
          elif value == 'Nominal':
            test[col_names[index]] = test[col_names[index]].astype(str)
            test[col_names[index]] = test[col_names[index]].fillna(central_tend_dict.get(col_names[index]))
            test[col_names[index]] = test[col_names[index]].replace('', central_tend_dict.get(col_names[index]))
      
      train.reset_index(drop = True, inplace = True)
      test.reset_index(drop = True, inplace = True)
      if percent_missing > 0:
        st.write('✅ — Dataset missingness handling complete!')
        if percent_missing <= 0.2:
          st.write(f'⋯ {len(train)} rows left for training set post-missingness handling!')
          st.write(f'⋯ {len(test)} rows left for testing set post-missingness handling!')
        elif percent_missing > 0.2:
          st.warning(f'Warning: large missingness detected, imputed missingness rows make up {percent_missing * 100:.2f}% of total rows!', icon = '🚧')
          st.write(f'⋯ {len(train)} rows left for training set post-missingness handling!')
          st.write(f'⋯ {len(test)} rows left for testing set post-missingness handling!')
      
      # Numerical list preparation for outlier handling
      col_names_num = []
      for index, value in enumerate(col_types):
        if value == 'Float' or value == 'Integer':
          col_names_num.append(col_names[index])
      
      # Outlier handling and power transformation
      transformer = PowerTransformer(method = 'yeo-johnson', standardize = False)
      train_cutoff_dict_low = dict()
      train_cutoff_dict_high = dict()
      skew = 0
      kurt_fish = 0
      outlier_handling_check = 0
      for index, value in enumerate(col_names_num):
        train_cutoff_dict_low[col_names_num[index]] = train[col_names_num[index]].quantile(0.0015)
        train_cutoff_dict_high[col_names_num[index]] = train[col_names_num[index]].quantile(0.9985)
        skew = train[col_names_num[index]].skew()
        kurt_fish = train[col_names_num[index]].kurtosis()
        if kurt_fish >= 3.00:
          train = train[train[col_names_num[index]] > train_cutoff_dict_low.get(col_names_num[index])]
          train = train[train[col_names_num[index]] < train_cutoff_dict_high.get(col_names_num[index])]
          outlier_handling_check = outlier_handling_check + 1
          if abs(skew) >= 1.00:
            train[col_names_num[index]] = transformer.fit_transform(train[[col_names_num[index]]]).flatten()
            test[col_names_num[index]] = transformer.transform(test[[col_names_num[index]]]).flatten()
            outlier_handling_check = outlier_handling_check + 1
        elif abs(skew) >= 1.00:
          train[col_names_num[index]] = transformer.fit_transform(train[[col_names_num[index]]]).flatten()
          test[col_names_num[index]] = transformer.transform(test[[col_names_num[index]]]).flatten()
          outlier_handling_check = outlier_handling_check + 1
      
      train.reset_index(drop = True, inplace = True)
      test.reset_index(drop = True, inplace = True)
      if outlier_handling_check > 0:
        st.write('✅ — Dataset outlier handling complete!')
        st.write(f'⋯ {len(train)} rows left for training set post-outlier handling!')
        st.write(f'⋯ {len(test)} rows left for testing set post-outlier handling!')

      # Target variable selection
      st.write('Target Variable Selection:')
      train_info = pd.DataFrame({'Variables': train.columns, 'Non-Null Count': train.count(numeric_only = False), 'Data Type': train.dtypes}).reset_index(drop = True)
      st.dataframe(train_info, hide_index = True)
      unassigned_count_2 = 0
      target = None
      target_class = None
      is_object = None
      with st.form('target_variable_selection_form'):
        st.write(tw.dedent(
            """
            Select a target variable for machine learning!

            * 'object' variables are either Ordinal/Nominal according to the previous type specification
            * Categorical target variables are always treated as Nominal
            * One-vs-Rest (OvR) encoding would be applied to categorical targets with more than 2 classes
            * User must select a class 1 label for the chosen categorical target variable's classes
            """
        ).strip())
        target_options = ['-'] + train.columns.tolist()
        target = st.selectbox('Select a target variable:', (target_options), accept_new_options = False)
        if target == '-':
          unassigned_count_2 = unassigned_count_2 + 1
        if target != '-':
          if train[target].dtypes == float or train[target].dtypes == int:
            is_object = False
          elif train[target].dtypes == object:
            st.write(train[target].value_counts())
            target_class_options = ['-'] + train[target].unique().tolist()
            target_class = st.selectbox('Select a class 1 label:', (target_class_options), accept_new_options = False)
            if target_class == '-':
              unassigned_count_2 = unassigned_count_2 + 1
            is_object = True
        submitted_2 = st.form_submit_button('Confirm target variable/class assignment')
      
      if submitted_2 == True:
        submitted_2_ref = st.session_state['submitted_2_ref'] = True
      else:
        submitted_2_ref = st.session_state['submitted_2_ref']
      
      if submitted_2_ref == True:
        if unassigned_count_2 > 0:
          st.error('Detected target variable/class without dropdown selection!', icon = '🛑')
        else:
          st.write('✅ — Target variable assignment complete!') # Guarded execution block (layer 3)

          # One-vs-Rest (OVR) binary encoding for categorical target
          if is_object == True:
            train[f'{target}_{target_class}'] = [x if x == target_class else 0 for x in train[target]]
            test[f'{target}_{target_class}'] = [x if x == target_class else 0 for x in test[target]]
            train[f'{target}_{target_class}'] = train[f'{target}_{target_class}'].map({target_class: 1, 0: 0})
            test[f'{target}_{target_class}'] = test[f'{target}_{target_class}'].map({target_class: 1, 0: 0})
            train = train.drop(target, axis = 1)
            test = test.drop(target, axis = 1)
            target_encoded = f'{target}_{target_class}'
            idx = col_names.index(target)
            del col_names[idx]
            del col_types[idx] # 'col_names'/'col_types' lists are no longer in use for manipulation past this line as they're no longer parallel
            st.write('✅ — Categorical target binary encoding complete!')
          
          # Single cardinality categorical variables cleaning
          col_names_2 = [col for col in train.columns]
          for col in col_names_2:
            if train[col].dtypes == object and train[col].nunique() == 1:
              train.drop(col, axis = 1, inplace = True)
              test.drop(col, axis = 1, inplace = True)
          col_names_2 = [col for col in train.columns]
          
          # One Hot Encoding (OHE) for low cardinality categorical features
          col_names_lc = []
          for col in col_names_2:
            if train[col].dtypes == object and train[col].nunique() <= 5:
              col_names_lc.append(col)
          oh_encoder = OneHotEncoder(drop = 'first', sparse_output = False, dtype = int, handle_unknown = 'ignore')
          oh_encoded_train = oh_encoder.fit_transform(train[col_names_lc])
          oh_encoded_test = oh_encoder.transform(test[col_names_lc])
          oh_encoded_train_df = pd.DataFrame(oh_encoded_train, columns = oh_encoder.get_feature_names_out(col_names_lc))
          oh_encoded_test_df = pd.DataFrame(oh_encoded_test, columns = oh_encoder.get_feature_names_out(col_names_lc))
          train = pd.concat([train, oh_encoded_train_df], axis = 1)
          test = pd.concat([test, oh_encoded_test_df], axis = 1)
          train.drop(col_names_lc, axis = 1, inplace = True)
          test.drop(col_names_lc, axis = 1, inplace = True)
          if col_names_lc:
            st.write('✅ — One hot encoding complete!')
          
          # Target encoding for high cardinality categorical features
          col_names_2 = [col for col in train.columns]
          col_names_hc = []
          for col in col_names_2:
            if train[col].dtypes == object and train[col].nunique() > 5:
              col_names_hc.append(col)
          if is_object == False:
            t_encoder = TargetEncoder(target_type = 'continuous', smooth = 'auto', random_state = 42)
          elif is_object == True:
            t_encoder = TargetEncoder(target_type = 'binary', smooth = 'auto', random_state = 42)
          for col in col_names_hc:
            train[col] = t_encoder.fit_transform(train[[col]], train[[target_encoded if is_object == True else target]]).flatten()
            test[col] = t_encoder.transform(test[[col]]).flatten()
          
          train.reset_index(drop = True, inplace = True)
          test.reset_index(drop = True, inplace = True)
          st.write('✅ — Target encoding complete!')

          # Feature/target split
          dep_var = target_encoded if is_object == True else target # Fuck it -> tired of checking condition everytime target is called
          feature_train = train.drop(columns = dep_var)
          target_train = train[[dep_var]]
          feature_test = test.drop(columns = dep_var)
          target_test = test[[dep_var]]
          st.write('✅ — Feature/target split complete!')
          st.write(f'⋯ {len(feature_train)} rows left for feature (train) set post-feature/target split!')
          st.write(f'⋯ {len(target_train)} rows left for target (train) set post-feature/target split!')
          st.write(f'⋯ {len(feature_test)} rows left for feature (test) set post-feature/target split!')
          st.write(f'⋯ {len(target_test)} rows left for target (test) set post-feature/target split!')

          # Z-score standardization of numerical variables
          scaler = StandardScaler()
          if dep_var not in col_names_num and is_object == False:
            col_names_num.append(dep_var)
          for col in col_names_hc:
            col_names_num.append(col)
          
          for col in col_names_num:
            if col == dep_var:
              target_train[col] = scaler.fit_transform(target_train[[col]]).flatten()
              target_test[col] = scaler.transform(target_test[[col]]).flatten()
            else:
              feature_train[col] = scaler.fit_transform(feature_train[[col]]).flatten()
              feature_test[col] = scaler.transform(feature_test[[col]]).flatten()
          if col_names_num:
            st.write('✅ — Z-score standardization complete!')
          
          # Undersampling to handle imbalanced categorical target
          if is_object == True:
            undersampler = RandomUnderSampler(random_state = 42)
            feature_train_balanced, target_train_balanced = undersampler.fit_resample(feature_train, target_train)
            resampled = True
            st.write('✅ — Undersampling for imbalanced target complete!')
            st.write(f'⋯ {len(feature_train_balanced)} rows left for feature (train-balanced) set post-undersampling!')
            st.write(f'⋯ {len(target_train_balanced)} rows left for target (train-balanced) set post-undersampling!')
          else:
            resampled = False
          
          # Variance Inflation Factor (VIF) check for multicollinearity
          if dep_var in col_names_num and is_object == False:
            col_names_num.remove(dep_var)
          
          vif_df_switch = False
          if len(col_names_num) > 1:
            intercept = add_constant(feature_train[col_names_num])
            vif_df = pd.DataFrame([vif(intercept.values, x) for x in range(intercept.shape[1])], index = intercept.columns).reset_index()
            vif_df.columns = ['Features', 'VIF Score']
            vif_df = vif_df.loc[vif_df.Features != 'const']
            vif_df = vif_df.sort_values(by = 'VIF Score', ascending = False)
            vif_df = vif_df.reset_index(drop = True)
            vif_df_switch = True
          
          if vif_df_switch == True:
            vif_df = vif_df[vif_df['VIF Score'] >= 5]
            col_names_num_vif = [x for x in vif_df['Features']]
            feature_train.drop(columns = col_names_num_vif, inplace = True)
            feature_test.drop(columns = col_names_num_vif, inplace = True)
            if resampled == True:
              feature_train_balanced.drop(columns = col_names_num_vif, inplace = True)
            st.write('✅ — VIF multicollinearity diagnostic complete!')

          # Executing machine learning algorithms and evaluation metrics
          st.divider()
          st.header('⸻ Machine Learning 📊')

          if is_object == False: # Regression modeling

            # New data check
            if 'file_name_check' not in st.session_state:
              file_name_check = st.session_state['file_name_check'] = None
            else:
              file_name_check = st.session_state['file_name_check']

            # Linear model, linear regression
            if file_name_check != file_name:
              ln = LinearRegression()
              ln.fit(feature_train, target_train)
              ln_pred = ln.predict(feature_test)
              r2_ln = st.session_state['r2_ln'] = r2_score(target_test, ln_pred)
              rmse_ln = st.session_state['rmse_ln'] = root_mean_squared_error(target_test, ln_pred)
              mae_ln = st.session_state['mae_ln'] = mean_absolute_error(target_test, ln_pred)
              mape_ln = st.session_state['mape_ln'] = mean_absolute_percentage_error(target_test, ln_pred)
              st.write('✅ — Linear regression fitted!')
            elif file_name_check == file_name:
              st.write('✅ — Linear regression fitted!')

            # Tree-based model, decision tree regressor
            if file_name_check != file_name:
              dt_reg = DecisionTreeRegressor(random_state = 42)
              dt_reg.fit(feature_train, target_train)
              dt_reg_pred = dt_reg.predict(feature_test)
              r2_dt_reg = st.session_state['r2_dt_reg'] = r2_score(target_test, dt_reg_pred)
              rmse_dt_reg = st.session_state['rmse_dt_reg'] = root_mean_squared_error(target_test, dt_reg_pred)
              mae_dt_reg = st.session_state['mae_dt_reg'] = mean_absolute_error(target_test, dt_reg_pred)
              mape_dt_reg = st.session_state['mape_dt_reg'] = mean_absolute_percentage_error(target_test, dt_reg_pred)
              st.write('✅ — Decision tree regressor fitted!')
            elif file_name_check == file_name:
              st.write('✅ — Decision tree regressor fitted!')

            # Ensemble model, light gradient boosting machine regressor
            if file_name_check != file_name:
              lgbm_reg = lgbm.LGBMRegressor(random_state = 42, n_jobs = -1)
              lgbm_reg.fit(feature_train, target_train, eval_set = [(feature_test, target_test)], callbacks = [lgbm.early_stopping(stopping_rounds = 3)])
              lgbm_reg_pred = lgbm_reg.predict(feature_test)
              r2_lgbm_reg = st.session_state['r2_lgbm_reg'] = r2_score(target_test, lgbm_reg_pred)
              rmse_lgbm_reg = st.session_state['rmse_lgbm_reg'] = root_mean_squared_error(target_test, lgbm_reg_pred)
              mae_lgbm_reg = st.session_state['mae_lgbm_reg'] = mean_absolute_error(target_test, lgbm_reg_pred)
              mape_lgbm_reg = st.session_state['mape_lgbm_reg'] = mean_absolute_percentage_error(target_test, lgbm_reg_pred)
              st.write('✅ — Light gradient boosting machine regressor fitted!')
            elif file_name_check == file_name:
              st.write('✅ — Light gradient boosting machine regressor fitted!')

            # Regression report
            st.write('#### Output Statistics')
            
            st.text(tw.dedent(
                f'''
                > Models Used

                • Linear Model — Linear Regression
                • Tree-Based Model — Decision Tree Regressor (DT)
                • Ensemble Model — Light Gradient Boosting Machine Regressor (LGBM)

                > Train/Test Sets Sample Size Check

                • Feature (Train) Sample Size (n): {len(feature_train)}
                • Target (Train) Sample Size (n): {len(target_train)}
                • Feature (Test) Sample Size (n): {len(feature_test)}
                • Target (Test) Sample Size (n): {len(target_test)}

                > Train/Test Sets Dimensionality Check

                • Feature (Train) Column Count: {len(feature_train.columns)}
                • Target (Train) Column Count: {len(target_train.columns)}
                • Feature (Test) Column Count: {len(feature_test.columns)}
                • Target (Test) Column Count: {len(target_test.columns)}

                > Model Fit Evaluation Metrics (Test Set Predictions)

                ---- Coefficient of Determination (R2 Score - Unit: Percentage)
                • Linear Regression - R2 Score: {r2_ln * 100:.2f}%
                • DT Regressor - R2 Score: {r2_dt_reg * 100:.2f}%
                • LGBM Regressor - R2 Score: {r2_lgbm_reg * 100:.2f}%

                ---- Root Mean Squared Error (RMSE - Unit: Z-Score)
                • Linear Regression - RMSE: {rmse_ln:.4f}
                • DT Regressor - RMSE: {rmse_dt_reg:.4f}
                • LGBM Regressor - RMSE: {rmse_lgbm_reg:.4f}

                ---- Mean Absolute Error (MAE - Unit: Z-Score)
                • Linear Regression - MAE: {mae_ln:.4f}
                • DT Regressor - MAE: {mae_dt_reg:.4f}
                • LGBM Regressor - MAE: {mae_lgbm_reg:.4f}

                ---- Mean Absolute Percentage Error (MAPE - Unit: Percentage)
                • Linear Regression - MAPE: {mape_ln * 100:.2f}%
                • DT Regressor - MAPE: {mape_dt_reg * 100:.2f}%
                • LGBM Regressor - MAPE: {mape_lgbm_reg * 100:.2f}%
                '''
            ).strip())

            # Regression best model explainer (dalex)
            model_names = ['XAI: Linear Regression', 'XAI: DT Regressor', 'XAI: LGBM Regressor']
            model_fits = [ln, dt_reg, lgbm_reg]
            model_rmses = [rmse_ln, rmse_dt_reg, rmse_lgbm_reg]

            best_model_rmse = min(model_rmses)
            best_model_fit = model_fits[model_rmses.index(best_model_rmse)]
            best_model_name = model_names[model_fits.index(best_model_fit)]

            best_model_explainer = dx.Explainer(best_model_fit, feature_train, target_train, label = best_model_name, verbose = False)

            st.text(tw.dedent(
                f'''
                > Explainable Artificial Intelligence (XAI)

                • Best Model - {best_model_name[5:]}
                • Evaluation Metric for Determination of Best Model - Root Mean Squared Error (RMSE) at {best_model_rmse:.4f}
                • Loss Function - Root Mean Squared Error (RMSE)
                '''
            ).strip())

            st.write('• Permutation Feature Importance (PFI):')
            pfi = best_model_explainer.model_parts(random_state = 42)
            pfi_fig: go.Figure = pfi.plot(show = False)
            pfi_fig_ss = st.session_state['pfi_fig_ss'] = pfi_fig.update_layout(height = 300 if len(feature_train.columns) >= 6 else 250,
                                                                                width = None,
                                                                                autosize = True,
                                                                                title_font_size = 16,
                                                                                font = dict(size = 11 if len(feature_train.columns) >= 6 else 13))
            st.plotly_chart(pfi_fig_ss, width = 'stretch', config = {'displayModeBar': False})

            st.write('• Partial Dependence Plots (PDPs):')
            pdp = best_model_explainer.model_profile(random_state = 42, verbose = False)
            pdp_fig: go.Figure = pdp.plot(show = False)
            pdp_fig_ss = st.session_state['pdp_fig_ss'] = pdp_fig.update_layout(showlegend = False,
                                                                                height = 600,
                                                                                width = None,
                                                                                autosize = False,
                                                                                title_x = 0.5,
                                                                                margin = dict(l = 50))
            st.plotly_chart(pdp_fig_ss, width = 'stretch', config = {'displayModeBar': False})
          
          elif is_object == True: # Classification modeling

            # Linear model, logistic regression
            logit = LogisticRegression(random_state = 42)
            logit.fit(feature_train, target_train)
            logit_pred = logit.predict(feature_test)
            logit_metrics = classification_report(target_test, logit_pred)
            st.write('✅ — Logistic regression fitted!')

            # Linear model, logistic regression (resampled)
            logit_rs = LogisticRegression(random_state = 42)
            logit_rs.fit(feature_train_balanced, target_train_balanced)
            logit_rs_pred = logit_rs.predict(feature_test)
            logit_rs_metrics = classification_report(target_test, logit_rs_pred)
            st.write('✅ — Logistic regression (undersampled) fitted!')

            # Tree-based model, decision tree classifier
            dt_class = DecisionTreeClassifier(random_state = 42)
            dt_class.fit(feature_train, target_train)
            dt_class_pred = dt_class.predict(feature_test)
            dt_class_metrics = classification_report(target_test, dt_class_pred)
            st.write('✅ — Decision tree classifier fitted!')

            # Tree-based model, decision tree classifier (resampled)
            dt_class_rs = DecisionTreeClassifier(random_state = 42)
            dt_class_rs.fit(feature_train_balanced, target_train_balanced)
            dt_class_rs_pred = dt_class_rs.predict(feature_test)
            dt_class_rs_metrics = classification_report(target_test, dt_class_rs_pred)
            st.write('✅ — Decision tree classifier (undersampled) fitted!')

            # Ensemble model, light gradient boosting machine classifier
            lgbm_class = lgbm.LGBMClassifier(random_state = 42, n_jobs = -1)
            lgbm_class.fit(feature_train, target_train, eval_set = [(feature_test, target_test)], callbacks = [lgbm.early_stopping(stopping_rounds = 3)])
            lgbm_class_pred = lgbm_class.predict(feature_test)
            lgbm_class_metrics = classification_report(target_test, lgbm_class_pred)
            st.write('✅ — Light gradient boosting machine classifier fitted!')

            # Ensemble model, light gradient boosting machine classifier (resampled)
            lgbm_class_rs = lgbm.LGBMClassifier(random_state = 42, n_jobs = -1)
            lgbm_class_rs.fit(feature_train_balanced, target_train_balanced, eval_set = [(feature_test, target_test)], callbacks = [lgbm.early_stopping(stopping_rounds = 3)])
            lgbm_class_rs_pred = lgbm_class_rs.predict(feature_test)
            lgbm_class_rs_metrics = classification_report(target_test, lgbm_class_rs_pred)
            st.write('✅ — Light gradient boosting machine classifier (undersampled) fitted!')

            # Classification report
            st.write('#### Output Statistics')
            
            st.text(tw.dedent(
                f'''
                > Models Used
                
                • Linear Model — Logistic Regression
                • Tree-Based Model — Decision Tree Classifier (DT)
                • Ensemble Model — Light Gradient Boosting Machine Classifier (LGBM)

                > Train/Test Sets Sample Size Check

                • Feature (Train) Sample Size (n): {len(feature_train)}
                • Target (Train) Sample Size (n): {len(target_train)}
                • Feature (Train-Balanced) Sample Size (n): {len(feature_train_balanced)}
                • Target (Train-Balanced) Sample Size (n): {len(target_train_balanced)}
                • Feature (Test) Sample Size (n): {len(feature_test)}
                • Target (Test) Sample Size (n): {len(target_test)}

                > Train/Test Sets Dimensionality Check

                • Feature (Train) Column Count: {len(feature_train.columns)}
                • Target (Train) Column Count: {len(target_train.columns)}
                • Feature (Train-Balanced) Column Count: {len(feature_train_balanced.columns)}
                • Target (Train-Balanced) Column Count: {len(target_train_balanced.columns)}
                • Feature (Test) Column Count: {len(feature_test.columns)}
                • Target (Test) Column Count: {len(target_test.columns)}
                '''
            ).strip())

            st.write('---- Classification Reports (Test Set Predictions)')
            st.write('• Logistic Regression:')
            st.code(logit_metrics, language = None, width = 513)
            st.write('• Logistic Regression (Undersampled):')
            st.code(logit_rs_metrics, language = None, width = 513)
            st.write('• DT Classifier:')
            st.code(dt_class_metrics, language = None, width = 513)
            st.write('• DT Classifier (Undersampled):')
            st.code(dt_class_rs_metrics, language = None, width = 513)
            st.write('• LGBM Classifier:')
            st.code(lgbm_class_metrics, language = None, width = 513)
            st.write('• LGBM Classifier (Undersampled):')
            st.code(lgbm_class_rs_metrics, language = None, width = 513)

          # E

else:
  st.subheader('No file upload detected')
  
