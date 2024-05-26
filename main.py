import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import numpy as np
from Dashboard import dashboard

def custom_css():
    """
    Inject custom CSS styles into Streamlit app.
    """
    custom_styles = """
    <style>
        body {
            background-color: #f0f0f0; /* Set background color */
        }
        .container {
            max-width: 800px; /* Set maximum width for content */
            margin: auto; /* Center content horizontally */
            padding: 200px; /* Add padding to content */
            background-color: #ffffff; /* Set background color for content */
            border-radius: 10px; /* Add rounded corners to content */
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1); /* Add shadow to content */
        }
    </style>
    """
    st.markdown(custom_styles, unsafe_allow_html=True)

def home():
    st.markdown("""
    <style>
    /* Center align the title */
    .title {
        text-align: center;
    }
    /* Add padding to the left side of the content */
    .content {
        padding-left: 180px;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 20px !important;
    }
    /* Increase font size of titles */
    .content h2 {
        font-size: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title"> <h1 style="color: #F63366;"> Morocco Rivers Forecasting App </h1> </div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content">
    
    ## Bienvenue!
    Cette application utilise l'apprentissage automatique pour prédire les niveaux des rivières en temps réel au Maroc.

    <p style='text-align: center; color: #F63366; font-weight:900; font-size: 35px;'>---------------------------------------------------------------------------------------------------------</p>

    ## Caractéristiques principales
    - Prédiction en temps réel des niveaux des rivières au Maroc.
    - Utilisation d'un modèle LSTM pour la prédiction.
    - Sauvegarde des résultats dans une base de données locale.

    <p style='text-align: center; color: #F63366; font-weight:900; font-size: 35px;'>---------------------------------------------------------------------------------------------------------</p>

    ## Détails techniques
    - **Modèle utilisé:** LSTM (Long Short-Term Memory).
    - **Précision de prédiction:** Elevée.
    - **Temps de réponse:** Moins de 10 secondes en moyenne par prédiction.

    <p style='text-align: center; color: #F63366; font-weight:900; font-size: 35px;'>---------------------------------------------------------------------------------------------------------</p>

    ## Comment utiliser l'application
    1. Sélectionnez l'option 'Prédiction des niveaux' pour entrer les données et obtenir une prédiction des niveaux des rivières.
    2. Sélectionnez l'option 'Dashboard' pour visualiser les données en temps réel.

    <p style='text-align: center; color: #F63366; font-weight:900; font-size: 35px;'>---------------------------------------------------------------------------------------------------------</p>

    ## À propos de l'application
    Cette application a été développée dans le cadre d'un projet visant à prédire les niveaux des rivières au Maroc en utilisant des techniques avancées de machine learning.
    Elle utilise TensorFlow et Scikit-Learn pour les prétraitements et l'apprentissage automatique, et stocke les résultats dans une base de données locale.
    Le modèle utilisé est un réseau de neurones LSTM, connu pour ses capacités à traiter les données séquentielles.

    <p style='text-align: center; color: #F63366; font-weight:900; font-size: 35px;'>---------------------------------------------------------------------------------------------------------</p>

    ## Comment utiliser l'application
    Pour commencer, sélectionnez l'une des options du menu en haut de la page.
    
    </div>
    """, unsafe_allow_html=True)

def about():
    st.markdown("""
    <style>
    /* Center align the title */
    .title {
        text-align: center;
    }
    /* Add padding to the left side of the content */
    .content {
        padding-left: 120px;
    }
    /* Increase font size for all text */
    .stMarkdown p, .stMarkdown li {
        font-size: 20px !important;
    }
    /* Increase font size of titles */
    .content h5, .content h3 {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'> <h1> À propos de l'application </h1> </div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="content">
    <br />
    <h5>Ce projet vise à surveiller les rivières susceptibles de s'assécher dans le futur dans notre royaume du Maroc.</h5>
    <h5>Nous utilisons l'intelligence artificielle pour prédire et résoudre les problèmes liés à ces rivières à l'avenir.</h5>
    <h5>Grâce à ces prévisions, nous espérons trouver des solutions pour préserver nos ressources en eau.</h5>
    <h3>Auteurs:</h3>
    
    - AMGAROU Salma <br/>
    - BOUADIF ABDELKRIM <br/>
    - HAFDAOUI Hamza  <br/>
    - IDRISSI HAMZA 

    <h3>Supervisé par:</h3>
    
    - Professeur Mostafa Ezziyyani
    <h3>Date de création:</h3>  
    
    - Mardi 26 mai 2024

    <h3>Technologies utilisées:</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.header("")
    with col2:
        st.image("images/scikit-learn.svg",  width=260)
    with col3:
        st.header("")
        st.image("images/tensorflow-ar21.svg",  width=450)
    with col4:
        pass
    st.markdown("""
    <div class="content">
    
    <h3>Langages de programmation:</h3>
    <br />
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        st.header("")
        st.image("images/python.svg",width=350)
        st.header("")
    with col3:
        pass
    with col4:
        pass
    with col5:
        pass

    st.markdown("""
    <div class="content">
    
    <h3>Framework:</h3>
    <br />
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        st.image("images/streamlit.png",width=350)
        st.header("")
    with col3:
        pass
    with col4:
        pass
    with col5:
        pass

    st.markdown("""
    <div class="content">
    
    <h3>Systèmes d'exploitation:</h3>
    
    - Unix  <br />
    - MacOS <br />
    - Windows
    """, unsafe_allow_html=True)

def load_data():
    source = pd.read_csv('Data/RIVER_DATA.csv', sep=';')
    source['Year'] = pd.to_datetime(source['Year'], format='%Y')
    df_ = pd.DataFrame(source.values, index=source['Year'], columns=source.columns)
    df__ = df_.drop(columns=['Name', 'Near_city', 'River', 'Year'])

    for column in df__.columns:
        df__[column] = pd.to_numeric(df__[column], errors='coerce')

    impute = SimpleImputer(strategy='mean')
    df = pd.DataFrame(impute.fit_transform(df__), columns=df__.columns)

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df__.columns)

    return df

def train_model():
    st.file_uploader("Choose a file")
    if st.button('Train the model LSTM'):
        # Load data
        df = load_data()

        # Split data into features and target
        y = df['Res_capacityM3']
        x = df.drop(columns=['Res_capacityM3'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

        # Reshape data for LSTM input
        x_train = np.reshape(x_train.values, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test.values, (x_test.shape[0], x_test.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1], 1), return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Train the model
        training = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.33)

        # Save the model
        with open('pretrained_models/lstm_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        with open('pretrained_models/model_params.pkl', 'wb') as fid:
            pickle.dump({'model': model.get_weights(), 'training_history': training.history}, fid, pickle.HIGHEST_PROTOCOL)

        st.success("Model trained successfully!")

        test_loss, test_mae = model.evaluate(x_test, y_test)

        st.info(f"Test Loss: {test_loss}")
        st.info(f"Test MAE: {test_mae}")

        fig, axs = plt.subplots(2)
        fig.suptitle('Training Metrics')
        axs[0].plot(training.history['loss'], label='Train Loss')
        axs[0].plot(training.history['val_loss'], label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[1].plot(training.history['mae'], label='Train MAE')
        axs[1].plot(training.history['val_mae'], label='Validation MAE')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('MAE')
        axs[1].legend()
        st.pyplot(fig)

def make_predictions():
    # Load the model
    with open('pretrained_models/model_params.pkl', 'rb') as f:
        model_params = pickle.load(f)

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(15, 1), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Set the trained parameters
    model.set_weights(model_params['model'])

    # Load the data
    source = pd.read_csv('Data/RIVER_DATA.csv', sep=';')

    river_name = st.text_input("Enter the name of the river:")
    prediction_year = st.number_input("Enter the prediction year:", value=2024)

    if st.button('Predict'):
        river_data = source[source['River'] == river_name]

        if len(river_data) > 0:
            if len(river_data[river_data['Year'] == prediction_year]) > 0:
                user_data = river_data[river_data['Year'] == prediction_year]
            else:
                user_data = river_data.tail(1)  # Use the latest available data for that river
        else:
            st.error("No data found for the specified river in Morocco.")
            return

        # Drop irrelevant columns and concatenate the year information to the input data
        user_input_processed = user_data.drop(columns=['Name', 'Near_city', 'River'])
        user_input_processed['Year'] = prediction_year

        # Reshape user input for LSTM input
        user_input_reshaped = np.reshape(user_input_processed.values, (1, user_input_processed.shape[1], 1))

        # Make prediction using the trained model
        predicted_capacity = model.predict(user_input_reshaped)

        st.success(f"Predicted reservoir capacity for River {river_name} in {prediction_year}: {predicted_capacity[0][0]:.2f} m³")

        def create_time_series_plot():
            scaler = MinMaxScaler()
            river_data['Res_capacityM3_normalized'] = river_data[['Res_capacityM3']]

            fig = px.line(river_data, x='Year', y='Res_capacityM3_normalized', title=f"Time Series of Reservoir Capacity for {river_name} in Morocco")
            fig.update_xaxes(title='Year')

            # Manually set the range of the y-axis
            fig.update_yaxes(title='Normalized Reservoir Capacity', range=[0, 800])  # Adjust the range as needed

            return fig

        # Display a spinner while the plot is being generated
        with st.spinner('Generating plot...'):
            # Create the time series plot
            fig = create_time_series_plot()

        # Display the plot using Streamlit
        st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide", page_title="Morocco Rivers Forecasting App")
    choices = option_menu(
        menu_title=None,
        options=["Home", "Prédiction des niveaux", "Dashboard", "À propos"],
        icons=["house", "water", "graph-up-arrow", "info-circle"],
        menu_icon="cast",
        orientation="horizontal"
    )
    custom_css()

    if choices == "Home":
        home()
    elif choices == "À propos":
        about()
    elif choices == "Prédiction des niveaux":
        st.title("Reservoir Capacity Prediction")
        option = st.sidebar.selectbox("Choose an option:", ["Train Model", "Make Predictions"])

        if option == "Train Model":
            train_model()
        elif option == "Make Predictions":
            make_predictions()
    elif choices == "Dashboard":
        dashboard()

if __name__ == '__main__':
    main()
