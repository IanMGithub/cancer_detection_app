import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

columns = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]


def get_clean_data():
    df = pd.read_csv('data/data.csv', header=None, names=columns)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave Points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal Dimension (mean)", "fractal_dimension_mean"),

        ("Radius (SE)", "radius_se"),
        ("Texture (SE)", "texture_se"),
        ("Perimeter (SE)", "perimeter_se"),
        ("Area (SE)", "area_se"),
        ("Smoothness (SE)", "smoothness_se"),
        ("Compactness (SE)", "compactness_se"),
        ("Concavity (SE)", "concavity_se"),
        ("Concave Points (SE)", "concave_points_se"),
        ("Symmetry (SE)", "symmetry_se"),
        ("Fractal Dimension (SE)", "fractal_dimension_se"),

        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave Points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal Dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model()


@st.cache_resource
def load_scaler():
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler()


def add_predictions(input_data):
    model = load_model()
    scaler = load_scaler()
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)
    st.subheader("Cell cluster prediction: ")
    st.write("The cell cluster has been diagnosed as: ")

    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

    st.write("Benign Classification Probability: ",
             model.predict_proba(input_array_scaled)[0][0])
    st.write("Malignant Classification Probability: ",
             model.predict_proba(input_array_scaled)[0][1])


def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis', 'id'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter',
                  'Area', 'Smoothness', 'Compactness', 'Concavity',
                  'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'], input_data['area_mean'], input_data['smoothness_mean'], input_data[
                'compactness_mean'], input_data['concavity_mean'], input_data['concave_points_mean'], input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'], input_data['smoothness_se'], input_data[
                'compactness_se'], input_data['concavity_se'], input_data['concave_points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'], input_data['area_worst'], input_data['smoothness_worst'], input_data[
                'compactness_worst'], input_data['concavity_worst'], input_data['concave_points_worst'], input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def main():
    # page config
    st.set_page_config(
        page_title="Breast Cancer Tumour Diagnosis",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Takes measurements from a patient tissue sample and uses a logistic regression model to predict weather a breast tumour is benign or malignant. Measurements can be updated using the left hand slider.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)


main()
