import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data: pd.DataFrame):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame
    and display results in Streamlit app.
    """
    st.header("Exploratory Data Analysis (EDA)")

    # Show data info
    st.subheader("Data Overview")
    st.write(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    if st.checkbox("Show raw data"):
        st.dataframe(data)

    # Data types
    st.subheader("Data Types")
    dtype_df = pd.DataFrame(data.dtypes, columns=["Data Type"])
    dtype_df["# Missing"] = data.isnull().sum()
    st.dataframe(dtype_df)

    # Summary statistics for numeric columns
    st.subheader("Summary Statistics (Numerical Columns)")
    numerical_cols = data.select_dtypes(include='number').columns
    if len(numerical_cols) > 0:
        st.write(data[numerical_cols].describe().T)
    else:
        st.write("No numerical columns found.")

    # Summary statistics for categorical columns
    st.subheader("Summary Statistics (Categorical Columns)")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_summary = pd.DataFrame()
        cat_summary['Unique Values'] = data[categorical_cols].nunique()
        cat_summary['Most Frequent'] = data[categorical_cols].mode().loc[0]
        cat_summary['Frequency'] = data[categorical_cols].apply(lambda x: x.value_counts().iloc[0])
        st.write(cat_summary)
    else:
        st.write("No categorical columns found.")

    # Missing values visualization
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        fig, ax = plt.subplots()
        missing_values.plot.bar(ax=ax)
        ax.set_ylabel("Count Missing")
        ax.set_title("Missing Values per Column")
        st.pyplot(fig)
    else:
        st.write("No missing values found.")

    # Distribution plots for numerical columns
    if len(numerical_cols) > 0:
        st.subheader("Distribution of Numerical Features")
        selected_num_cols = st.multiselect("Select numerical columns to plot distributions:", numerical_cols.tolist(), default=numerical_cols.tolist()[:3])
        for col in selected_num_cols:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(data[col].dropna(), kde=True, ax=ax, color='cornflowerblue')
            ax.set_title(f'Distribution for {col}')
            st.pyplot(fig)

    # Count plots for categorical columns
    if len(categorical_cols) > 0:
        st.subheader("Count Plots for Categorical Features")
        selected_cat_cols = st.multiselect("Select categorical columns for count plots:", categorical_cols.tolist(), default=categorical_cols.tolist()[:3])
        for col in selected_cat_cols:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.countplot(data=data, x=col, order=data[col].value_counts().index, ax=ax, palette='pastel')
            ax.set_title(f'Count Plot for {col}')
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    # Correlation matrix for numerical features
    if len(numerical_cols) > 1:
        st.subheader("Correlation Matrix (Numerical Features)")
        corr = data[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, square=True)
        st.pyplot(fig)
    else:
        st.write("Not enough numerical columns to show correlation matrix.")
