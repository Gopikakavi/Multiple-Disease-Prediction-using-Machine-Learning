import os
import pickle
import streamlit as st
from  streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/svm_diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Assume you have a cancer prediction model saved as 'cancer_model.sav'
# You would need to train and save this model beforehand
try:
    cancer_model = pickle.load(open(f'{working_dir}/saved_models/cancer.sav', 'rb'))
except FileNotFoundError:
    st.error("Cancer prediction model not found. Please ensure 'cancer_model.sav' is in the 'saved_models' directory.")
    cancer_model = None # Set to None or handle appropriately if the model is crucial


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Cancer Prediction'], # Added Cancer Prediction
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', '⚕️'], # Added an icon for cancer
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        # Convert all inputs to float and handle potential errors
        try:
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = 'Please enter valid numeric values for all fields.'
        except Exception as e:
            diab_diagnosis = f"An error occurred during prediction: {e}"

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        try:
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
        except ValueError:
            heart_diagnosis = 'Please enter valid numeric values for all fields.'
        except Exception as e:
            heart_diagnosis = f"An error occurred during prediction: {e}"

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        try:
            user_input = [float(x) for x in user_input]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = 'Please enter valid numeric values for all fields.'
        except Exception as e:
            parkinsons_diagnosis = f"An error occurred during prediction: {e}"

    st.success(parkinsons_diagnosis)


# ---
# Cancer Prediction Page
# ---
if selected == 'Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')

    if cancer_model is None:
        st.warning("Cannot perform prediction as the cancer model was not loaded. Please check the 'saved_models' directory.")
    else:
        # Define the columns for input fields, splitting them for better layout
        cols = st.columns(3)

        # Map attribute names to more user-friendly labels
        attributes = {
            "radius_mean": "Radius Mean", "texture_mean": "Texture Mean", "perimeter_mean": "Perimeter Mean",
            "area_mean": "Area Mean", "smoothness_mean": "Smoothness Mean", "compactness_mean": "Compactness Mean",
            "concavity_mean": "Concavity Mean", "concave points_mean": "Concave Points Mean",
            "symmetry_mean": "Symmetry Mean", "fractal_dimension_mean": "Fractal Dimension Mean",
            "radius_se": "Radius SE", "texture_se": "Texture SE", "perimeter_se": "Perimeter SE",
            "area_se": "Area SE", "smoothness_se": "Smoothness SE", "compactness_se": "Compactness SE",
            "concavity_se": "Concavity SE", "concave points_se": "Concave Points SE",
            "symmetry_se": "Symmetry SE", "fractal_dimension_se": "Fractal Dimension SE",
            "radius_worst": "Radius Worst", "texture_worst": "Texture Worst", "perimeter_worst": "Perimeter Worst",
            "area_worst": "Area Worst", "smoothness_worst": "Smoothness Worst",
            "compactness_worst": "Compactness Worst", "concavity_worst": "Concavity Worst",
            "concave points_worst": "Concave Points Worst", "symmetry_worst": "Symmetry Worst",
            "fractal_dimension_worst": "Fractal Dimension Worst"
        }

        # Collect user input for all attributes
        user_input_cancer = {}
        # We'll skip 'id' as it's an identifier, not a feature for prediction
        # 'diagnosis' is the target variable, not an input feature

        attribute_names_ordered = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
            "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
            "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
            "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        # Use a loop to create input fields dynamically
        for i, attr in enumerate(attribute_names_ordered):
            with cols[i % 3]: # Distribute inputs across 3 columns
                user_input_cancer[attr] = st.text_input(attributes[attr])

        cancer_diagnosis = ''

        if st.button('Cancer Test Result'):
            # Convert user inputs to a list in the correct order for prediction
            input_data = []
            valid_input = True
            for attr in attribute_names_ordered:
                try:
                    input_data.append(float(user_input_cancer[attr]))
                except ValueError:
                    valid_input = False
                    break # Exit loop if any input is invalid

            if not valid_input:
                cancer_diagnosis = 'Please enter valid numeric values for all fields.'
            else:
                try:
                    cancer_prediction = cancer_model.predict([input_data])

                    if cancer_prediction[0] == 1: # Assuming 1 for malignant, 0 for benign
                        cancer_diagnosis = 'The person is predicted to have Malignant Cancer.'
                    else:
                        cancer_diagnosis = 'The person is predicted to have Benign Cancer.'
                except Exception as e:
                    cancer_diagnosis = f"An error occurred during prediction: {e}"

        st.success(cancer_diagnosis)