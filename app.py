import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Title of the app
st.title('Financial Budget Analysis Tool')

# Introduction
st.markdown("""
This tool allows you to analyze your financial budget and sales data. You can upload a CSV file containing your data,
view visualizations, and even predict future sales. Simply follow the instructions below.
""")

# Step 1: Upload CSV data
st.header("Step 1: Upload Your Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of your data:")
    st.write(df)
    
    required_columns = ['Product', 'Budget', 'Sales']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}")
    else:
        # Calculate difference between budget and sales
        df['Difference'] = df['Sales'] - df['Budget']
        
        # Display the DataFrame with the new Difference column
        st.write("Data with Difference column:")
        st.write(df)
        
        # Step 2: Data visualization
        st.header("Step 2: Visualize Your Data")
        st.write("This chart shows the Budget vs. Sales for each product.")
        fig, ax = plt.subplots()
        ax.bar(df['Product'], df['Budget'], label='Budget')
        ax.bar(df['Product'], df['Sales'], bottom=df['Budget'], label='Sales')
        ax.set_ylabel('Amount')
        ax.legend()
        st.pyplot(fig)
        
        # Step 3: Predictive Analytics
        st.header("Step 3: Predict Future Sales")
        st.write("Select a product and the number of months to predict future sales.")

        # User input for prediction
        product_to_predict = st.selectbox("Select a product to predict future sales", df['Product'])
        months_to_predict = st.slider("Select number of months for prediction", 1, 12, 3)
        
        # Example predictive model
        product_data = df[df['Product'] == product_to_predict]
        X = np.arange(len(product_data)).reshape(-1, 1)  # Example feature: month index
        y = product_data['Sales'].values.reshape(-1, 1)  # Target: sales

        # Fit a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future sales
        future_X = np.arange(len(product_data), len(product_data) + months_to_predict).reshape(-1, 1)
        future_sales = model.predict(future_X)
        
        # Display prediction results
        prediction_results = pd.DataFrame({
            "Month": [f"Month {i+1}" for i in range(months_to_predict)],
            "Predicted Sales": future_sales.flatten()
        })
        st.write("Predicted future sales:")
        st.write(prediction_results)

        # Data visualization of predictions
        st.write("Future Sales Predictions")
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(product_data)), product_data['Sales'], label='Historical Sales')
        ax.plot(np.arange(len(product_data), len(product_data) + months_to_predict), future_sales, label='Predicted Sales', linestyle='--')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig)
else:
    st.write("Please upload a CSV file to start the analysis.")
