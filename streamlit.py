import pandas as pd
import numpy as np
import joblib
import pickle
from currency_converter import CurrencyConverter
import streamlit as st
from babel.numbers import format_currency


with open('encoders.pkl','rb') as file:
    encoders=pickle.load(file)
regmodel = joblib.load('randomforestmodel.pkl')
cluspca=joblib.load('clupca.pkl')
clusss=joblib.load('cluscaler.pkl')
clusmod=joblib.load('clukmeans.pkl')
st.title("Diamond Price Predication and Clustering")
col1, col2 = st.columns(2)
with col1:
    Carat_slider = st.slider("Carat (drag)", 0.10, 5.20)
with col2:
    Carat_input = st.number_input("Carat (type)", 0.10, 5.20, Carat_slider)

cut_options = encoders['cut'].categories_[0].tolist()
cut_input = st.selectbox("Select Cut", cut_options)
clarity_options=encoders['clarity'].categories_[0].tolist()
clarity_inpurt=st.selectbox("Select Clarity",clarity_options)
color_options=encoders['color'].categories_[0].tolist()
color_inpurt=st.selectbox("Select Color",color_options)

col1, col2 = st.columns(2)
with col1:
    Table_slider = st.slider("Table (drag)", 43.0, 95.0)
with col2:
    Table_input = st.number_input("Table (type)", 43.0, 95.0, Table_slider)

col1, col2 = st.columns(2)
with col1:
    X_slider = st.slider("X (drag)", 3.00, 11.00)
with col2:
    X_input = st.number_input("X (type)", 3.00, 11.00, X_slider)

col1, col2 = st.columns(2)
with col1:
    Y_slider = st.slider("Y (drag)", 3.00, 11.00)
with col2:
    Y_input = st.number_input("Y (type)", 3.00, 11.00, Y_slider)

col1, col2 = st.columns(2)
with col1:
    Z_slider = st.slider("Z (drag)", 1.00, 31.00)
with col2:
    Z_input = st.number_input("Z (type)", 1.00, 31.00, Z_slider)
depth_value = round((Z_input / ((X_input + Y_input) / 2)) * 100, 1)
Dim_ratio=(X_input+Y_input)/(2*Z_input)
volume_value = X_input * Y_input * Z_input
new_data = pd.DataFrame([{
    'carat': Carat_input,
    'cut': cut_input,
    'clarity': clarity_inpurt,
    'color': color_inpurt,
    'x': X_input,
    'y': Y_input,
    'z': Z_input,
    'depth': depth_value,
    'table': Table_input,
    'dim_ratio': Dim_ratio
}])

new_data['cut_encoded']=encoders['cut'].transform([new_data['cut']])[0][0]
new_data['clarity_encoded']=encoders['clarity'].transform([new_data['clarity']])[0][0]
new_data['color_encoded']=encoders['color'].transform([new_data['color']])[0][0]
model_df=new_data[['carat','cut_encoded', 'clarity_encoded', 'color_encoded','depth', 'table']]
price=regmodel.predict(model_df)
c = CurrencyConverter()
inr_price=c.convert(price,'USD','INR')
formatted_price = format_currency(inr_price, "INR", locale="en_IN")
new_data['log_carat']=np.log1p(new_data['carat'])
clusterdata=new_data[['log_carat','cut_encoded', 'clarity_encoded', 'color_encoded','table','depth','dim_ratio' ]]
clusst=clusss.transform(clusterdata)
cluspcaa=cluspca.transform(clusst)
pred=clusmod.predict(cluspcaa)[0]
cluster_labels = {
    1: "Mid-Range Balanced Diamonds",
    2: "Affordable Small Diamonds",
    0: "Premium Heavy Diamonds"
}
label = cluster_labels[pred]
st.metric("Belongs to Cluster",f"{label}", f"{formatted_price}",border=True)#:,.2f
