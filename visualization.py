import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to train the model and make predictions
def train_model(data):
    X = data[['sales', 'market_trend']].shift(1).dropna()
    y = data['sales'][1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, predictions

# Function to display metrics
def display_metrics(y_test, predictions):
    
    st.write(f'Prediction Difference: {np.sqrt(mean_squared_error(y_test, predictions)):.3f}')
    

# Streamlit app
st.title('Enhanced Demand Forecasting Application')

# Data upload
st.write('## Upload your dataset')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('### Dataset Preview')
    st.write(data.head())

    # Display basic statistics
    st.write('### Data Statistics')
    st.write(data.describe())

    # Data visualization
    st.write('### Data Visualization')
    st.line_chart(data['sales'])

    # Training the model
    model, X_train, X_test, y_train, y_test, predictions = train_model(data)
    
    # Display model performance metrics
    st.write('## Model Performance Metrics')
    display_metrics(y_test, predictions)

    # Plot the results
    st.write('### Forecast vs Actual')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.index, y_test, label='Actual Sales')
    ax.plot(y_test.index, predictions, label='Forecasted Sales')
    ax.legend()
    st.pyplot(fig)

    # User input for prediction
    st.write('## Make a Prediction')
    sales_input = st.number_input('Previous Day Sales', value=20)
    market_trend_input = st.number_input('Market Trend', value=0.0)

    # Make a single prediction
    input_data = np.array([[sales_input, market_trend_input]])
    prediction = model.predict(input_data)[0]
    st.write(f'Predicted Sales: {prediction:.2f}')

    # Export predictions
    st.write('## Export Predictions')
    if st.button('Export'):
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        predictions_df.to_csv('predictions.csv')
        st.write('Predictions exported successfully!')

    # Historical analysis
    st.write('## Historical Data Analysis')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, ax=ax)
    st.pyplot(fig)

    # Scenario analysis
    st.write('## Scenario Analysis')
    scenario_sales = st.number_input('Scenario Sales Input', value=20)
    scenario_trend = st.number_input('Scenario Trend Input', value=0.0)
    scenario_prediction = model.predict(np.array([[scenario_sales, scenario_trend]]))[0]
    st.write(f'Scenario Predicted Sales: {scenario_prediction:.2f}')

else:
    st.write('Please upload a CSV file to proceed.')
