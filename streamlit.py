import pandas as pd
import numpy as np
import joblib
import io
import os
from currency_converter import CurrencyConverter
import streamlit as st
from babel.numbers import format_currency
import boto3

bucket_name = "forestclassification"  
AWS_ACCESS_KEY_ID= os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY= os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION=  os.getenv("AWS_DEFAULT_REGION")
ENCODER_KEY=os.getenv("ENCODER_KEY")         
REG_MODEL_KEY=os.getenv("REG_MODEL_KEY")         
PCA_KEY=os.getenv("PCA_KEY")         
SCALER_KEY=os.getenv("SCALER_KEY")         
CLUSTER_MODEL_KEY=os.getenv("CLUSTER_MODEL_KEY")         


s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

@st.cache_resource
def load_all_from_s3():
    # Load Encoders
    encoders_obj = s3.get_object(Bucket=bucket_name, Key=ENCODER_KEY)
    encoders = joblib.load(io.BytesIO(encoders_obj['Body'].read()))
    #Regression Model
    reg_obj=s3.get_object(Bucket=bucket_name,Key=REG_MODEL_KEY)
    regmodel=joblib.load(io.BytesIO(reg_obj['Body'].read()))
    #PCA For Clustering
    pca_obj=s3.get_object(Bucket=bucket_name,Key=PCA_KEY)
    cluspca=joblib.load(io.BytesIO(pca_obj['Body'].read()))
    #Standard Scaler For Clustering
    ss_obj=s3.get_object(Bucket=bucket_name,Key=SCALER_KEY)
    clusss=joblib.load(io.BytesIO(ss_obj['Body'].read()))
    #Kmeans Model Clustering
    cm_obj=s3.get_object(Bucket=bucket_name,Key=CLUSTER_MODEL_KEY)
    clusmod=joblib.load(io.BytesIO(cm_obj['Body'].read()))

    return encoders, regmodel, cluspca, clusss,clusmod


encoders, regmodel, cluspca, clusss,clusmod=load_all_from_s3()

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

