
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.header(
    'Car residual values prediction'
)

st.subheader(
    'The Residual Values'
)

st.write(
    """
    The Residual Values (RV) model is a model to predict car values for given times. There are several car conditions having
    effects to future prices. In the model, transformed categorical features via One-Hot Encoding and scaled numeric
    features via normalization and polynomial transformer. The core model is used CatBoost regression.
    """
)

st.subheader(
    'To use model prediction, please following below steps:'
)

st.write(
    """
    1. From the left side of this page, there is an area to input several car conditions. \n
    2. To input car conditions that needed to be predicted. \n
    3. See the results below.
    """
)

# Load development data
df = pd.read_csv('carData.csv')
df = df[df['vehicleType'] == 'car']

# Create sidebar
st.sidebar.header(
    'Input car condition'
)

# Create function for car features
def userInputFeatures():
    carBrand = st.sidebar.selectbox(
        'Car Brand', (np.unique(df['carBrand']))
    )

    modelChoice = df['carModel'].loc[df['carBrand'] == carBrand]
    modelChoice = np.unique(modelChoice)

    carModel = st.sidebar.selectbox(
        'Car Model', modelChoice
    )

    fuelType = st.sidebar.selectbox(
        'Fuel Type', (np.unique(df['fuelType']))
    )

    gearType = st.sidebar.selectbox(
        'Gear Type', (np.unique(df['gearType']))
    )

    owner = st.sidebar.selectbox(
        'Owner', (np.unique(df['owner']))
    )

    age = st.sidebar.slider(
        'Age',
        1, 10, step = 1
    )

    data = {
        'carModel': [carModel],
        'carBrand': [carBrand],
        'fuelType': [fuelType],
        'gearType': [gearType],
        'owner': [owner],
        'kmDriven': [10000],
        'age': [age]
    }

    data = pd.DataFrame(data)

    estimateAge = int(data['age']) #Get age to repeat
    features = pd.DataFrame(
        np.repeat(
            data.values,
            estimateAge,
            axis = 0
        ),
        columns = data.columns
    )

    features['age'] = features.index + 1
    features['kmDriven'] = features['kmDriven'].cumsum()

    return features

inputData = userInputFeatures()

# Load model
model = pickle.load(
    open('model.pk', 'rb')
)

# Prediction
results = np.exp(
    model.predict(inputData)
)

resultsTable = pd.DataFrame(
    results,
    columns = ['Future prices']
)

resultsTable['Year'] = resultsTable.index + 1
resultsTable.set_index('Year', inplace = True)

#Display results
st.table(resultsTable)

# Plot results
st.line_chart(resultsTable)
