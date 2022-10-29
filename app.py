import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load("E:\TMLC\ML\Project 1\model\model.pkl")

st.set_page_config(page_title="Accident Severity Prediction App",
                   layout="wide")


# creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday",
               "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']


options_cause = ['No distancing', 'Changing lane to the right',
                 'Changing lane to the left', 'Driving carelessly',
                 'No priority to vehicle', 'Moving Backward',
                 'No priority to pedestrian', 'Other', 'Overtaking',
                 'Driving under the influence of drugs', 'Driving to the left',
                 'Getting off the vehicle improperly', 'Driving at high speed',
                 'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
                 'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
                        'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
                        'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
                        'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

option_light = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                'Darkness - lights unlit']

options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
                 'other', 'Double carriageway (median)', 'One way',
                 'Two-way (divided with solid lines road marking)', 'Unknown']


options_accident_area = ['Residential areas', 'Office areas', 'Recreational areas',
                         'Industrial area', 'Other', 'Church areas', 'Market areas', 'Rural village areas', 'Outside rural areas', 'Hospital areas', 'School areas', 'Unknown']

options_driving_experience = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No licence', 'Below 1yr', 'Unknown']

options_junction_type = ['No junction', 'Y Shape', 'Crossing',
                         'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape']

options_service_year = ['Above 10yr', '5-10yr',
                        '1-2yr', '2-5yr', 'Unknown', 'Below 1yr']

options_educational_level = ['Above high school', 'Junior high school',
                             'Elementary school', 'High school', 'Unknown', 'Illiterate', 'Writing & reading']

features = ['minute', 'hour', 'day_of_week', 'accident_cause', 'casualties', 'vehicles_involved', 'driver_age', 'vehicle_type',
            'light_condition', 'lanes', 'accident_area', 'driving_experience', 'junction_type', 'service_year', 'educational_level']


st.markdown("<h1 style='text-align: center;'>Road Traffic Accident Severity Prediction</h1>",
            unsafe_allow_html=True)
st.image("https://t3.ftcdn.net/jpg/03/72/46/46/360_F_372464646_Ks082AREONEjY5XYhWSexdDGFQ9tHr8S.jpg", use_column_width=True)

st.sidebar.title("About this application")
st.sidebar.write("""
        The aim of creating this app is to predict the severity of accident on the given features.
        
        
        The data is collected from Addis Ababa Sub-city police departments for master's research work.
        """)

st.sidebar.info("#### by: Smit Karanjia")

def main():
    with st.form('prediction_form'):

        st.header("Enter the input for following features:")

        minute = st.slider("Pickup minute: ", 0, 59, value=0, format="%d")
        hour = st.slider("Pickup hours: ", 0, 23, value=0, format="%d")
        day_of_week = st.selectbox("select day : ", options=options_day)
        accident_cause = st.selectbox(
            "Select Accident Cause: ", options=options_cause)
        casualties = st.slider("Hour of Accident: ", 1,
                               8, value=0, format="%d")
        vehicles_involved = st.slider(
            "vehicles involved : ", 1, 7, value=0, format="%d")
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        vehicle_type = st.selectbox(
            "Select Vehicle Type: ", options=options_vehicle_type)
        light_condition = st.selectbox(
            "light condition: ", options=option_light)
        lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        accident_area = st.selectbox(
            "Select Accident area: ", options=options_accident_area)
        driving_experience = st.selectbox(
            "Select Driving experience: ", options=options_driving_experience)
        junction_type = st.selectbox(
            "Select Junction type: ", options=options_junction_type)
        service_year = st.selectbox(
            "Select Service year of vehicle: ", options=options_service_year)
        educational_level = st.selectbox(
            "Select Educational level of Driver: ", options=options_educational_level)

        submit = st.form_submit_button("Predict")

    if submit:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        driver_age = ordinal_encoder(driver_age, options_age)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        light_condition = ordinal_encoder(light_condition, option_light)
        lanes = ordinal_encoder(lanes, options_lanes)
        accident_area = ordinal_encoder(accident_area, options_accident_area)
        driving_experience = ordinal_encoder(
            driving_experience, options_driving_experience)
        junction_type = ordinal_encoder(junction_type, options_junction_type)
        service_year = ordinal_encoder(service_year, options_service_year)
        educational_level = ordinal_encoder(
            educational_level, options_educational_level)

        data = np.array([minute, hour, day_of_week, accident_cause, casualties,
                         vehicles_involved, driver_age, vehicle_type, light_condition, lanes, 
                         accident_area, driving_experience, junction_type, service_year, educational_level]).reshape(1, -1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity of accident is:  {pred[0]}")


if __name__ == '__main__':
    main()
