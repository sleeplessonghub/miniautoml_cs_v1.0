st.title('Mini AutoML (Cross-Sectional) v1.0 - Streamlit')

uploaded_file = st.file_uploader("Upload a '.csv' or '.xlsx' file", type = ['csv', 'xlsx'], accept_multiple_files = False)

if uploaded_file is not None:
  try:
    if uploaded_file.name.endswith('.csv'):
      df_pp = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
      df_pp = pd.read_excel(uploaded_file)

    st.write(df_pp)

  except Exception as e:
    st.error(f"Error: {e}")
else:
  st.info('Upload a file to begin the analysis', icon = 'ℹ️')
