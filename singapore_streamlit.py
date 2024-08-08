from cycler import V
from decimal import Decimal
import singapore
import streamlit as st
import numpy as np
st.title(":blue[SINGAPORE RESALE FLAT PRICES PREDICTION]")
with st.sidebar:
    pred_year=st.radio("Select any year duration to predict the flat price",("Home","1990 - 1999","2000 - 2011","2012 - 2014","2014 - 2016","2017 - present"))
if pred_year == "Home":
    st.image("singapore_resale_price_predict/flat1.jpg",width=800)
elif pred_year == "1990 - 1999":
    with st.form(key="my_form"):
        town_values=singapore.resale1990_df['town'].unique()
        flat_value=singapore.resale1990_df['flat_type'].unique()
        storey_value=singapore.resale1990_df['storey_range'].unique()
        flat_model_value=singapore.resale1990_df['flat_model'].unique()
        year_value=singapore.resale1990['year'].unique()
        Town=st.selectbox("choose any town",town_values)
        flatType=st.selectbox("choose any Type of flat",flat_value)
        storeyRange=st.selectbox("choose any storey range",storey_value)
        flatModel=st.selectbox("choose any Model",flat_model_value)
        leaseDate=st.number_input("Enter the lease commenced year in YYYY format",min_value=1967,max_value=1997)
        Year=st.selectbox("choose any year",year_value)
        floorArea=st.number_input("Enter the Floor area in square meters",min_value=25,max_value=310)
        submit=st.form_submit_button(label='Predict Price')
        if submit:
             predicted_value=singapore.resale1990_pred(Town,flatType,storeyRange,floorArea,flatModel,leaseDate,Year)
             predicted_price=predicted_value[0]
             st.write(f"Predicted selling price is: {predicted_price:,.2f}")
elif pred_year == "2000 - 2011":
    with st.form(key="my_form"):
        town_values=singapore.resale2000_df['town'].unique()
        flat_value=singapore.resale2000_df['flat_type'].unique()
        storey_value=singapore.resale2000_df['storey_range'].unique()
        flat_model_value=singapore.resale2000_df['flat_model'].unique()
        year_value=singapore.resale2000['year'].unique()
        Town=st.selectbox("choose any town",town_values)
        flatType=st.selectbox("choose any Type of flat",flat_value)
        storeyRange=st.selectbox("choose any storey range",storey_value)
        flatModel=st.selectbox("choose any Model",flat_model_value)
        leaseDate=st.number_input("Enter the lease commenced year in YYYY format",min_value=1967,max_value=1997)
        Year=st.selectbox("choose any year",year_value)
        floorArea=st.number_input("Enter the Floor area in square meters",min_value=25,max_value=310)
        submit=st.form_submit_button(label='Predict Price')
        if submit:
             predicted_value=singapore.resale2000_pred(Town,flatType,storeyRange,floorArea,flatModel,leaseDate,Year)
             predicted_price=predicted_value[0]
             st.write(f"Predicted selling price is: {predicted_price:,.2f}")
elif pred_year == "2012 - 2014":
    with st.form(key="my_form"):
        town_values=singapore.resale2012_df['town'].unique()
        flat_value=singapore.resale2012_df['flat_type'].unique()
        storey_value=singapore.resale2012_df['storey_range'].unique()
        flat_model_value=singapore.resale2012_df['flat_model'].unique()
        year_value=singapore.resale2012['year'].unique()
        Town=st.selectbox("choose any town",town_values)
        flatType=st.selectbox("choose any Type of flat",flat_value)
        storeyRange=st.selectbox("choose any storey range",storey_value)
        flatModel=st.selectbox("choose any Model",flat_model_value)
        leaseDate=st.number_input("Enter the lease commenced year in YYYY format",min_value=1967,max_value=1997)
        Year=st.selectbox("choose any year",year_value)
        floorArea=st.number_input("Enter the Floor area in square meters",min_value=25,max_value=310)
        submit=st.form_submit_button(label='Predict Price')
        if submit:
             predicted_value=singapore.resale2012_pred(Town,flatType,storeyRange,floorArea,flatModel,leaseDate,Year)
             predicted_price=predicted_value[0]
             st.write(f"Predicted selling price is: {predicted_price:,.2f}")
elif pred_year == "2014 - 2016":
    with st.form(key="my_form"):
        town_values=singapore.resale2015_df['town'].unique()
        flat_value=singapore.resale2015_df['flat_type'].unique()
        storey_value=singapore.resale2015_df['storey_range'].unique()
        flat_model_value=singapore.resale2015_df['flat_model'].unique()
        year_value=singapore.resale2015['year'].unique()
        Town=st.selectbox("choose any town",town_values)
        flatType=st.selectbox("choose any Type of flat",flat_value)
        storeyRange=st.selectbox("choose any storey range",storey_value)
        flatModel=st.selectbox("choose any Model",flat_model_value)
        leaseDate=st.number_input("Enter the lease commenced year in YYYY format",min_value=1967,max_value=1997)
        Year=st.selectbox("choose any year",year_value)
        floorArea=st.number_input("Enter the Floor area in square meters",min_value=25,max_value=310)
        submit=st.form_submit_button(label='Predict Price')
        if submit:
             predicted_value=singapore.resale2015_pred(Town,flatType,storeyRange,floorArea,flatModel,leaseDate,Year)
             predicted_price=predicted_value[0]
             st.write(f"Predicted selling price is: {predicted_price:,.2f}")
elif pred_year == "2017 - present":
    with st.form(key="my_form"):
        town_values=singapore.resale2017_df['town'].unique()
        flat_value=singapore.resale2017_df['flat_type'].unique()
        storey_value=singapore.resale2017_df['storey_range'].unique()
        flat_model_value=singapore.resale2017_df['flat_model'].unique()
        year_value=singapore.resale2017['year'].unique()
        Town=st.selectbox("choose any town",town_values)
        flatType=st.selectbox("choose any Type of flat",flat_value)
        storeyRange=st.selectbox("choose any storey range",storey_value)
        flatModel=st.selectbox("choose any Model",flat_model_value)
        leaseDate=st.number_input("Enter the lease commenced year in YYYY format",min_value=1967,max_value=1997)
        Year=st.selectbox("choose any year",year_value)
        floorArea=st.number_input("Enter the Floor area in square meters",min_value=25,max_value=310)
        submit=st.form_submit_button(label='Predict Price')
        if submit:
             predicted_value=singapore.resale2017_pred(Town,flatType,storeyRange,floorArea,flatModel,leaseDate,Year)
             predicted_price=predicted_value[0]
             st.write(f"Predicted selling price is: {predicted_price:,.2f}")
