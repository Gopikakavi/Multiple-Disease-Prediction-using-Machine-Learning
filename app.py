import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# --- Set page configuration ---
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# --- Get the working directory ---
working_dir = os.path.dirname(os.path.abspath(__file__))
saved_models_dir = os.path.join(working_dir, 'saved_models')

# --- Function to load models safely ---
def load_model(model_name):
    model_path = os.path.join(saved_models_dir, model_name)
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_name}' not found at '{model_path}'. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return None

# --- Loading all saved models ---
diabetes_model = load_model('svm_diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')
cancer_model = load_model('cancer.sav') # Assuming cancer model is saved as 'cancer.sav'
liver_model = load_model('liver.sav')
kidney_model = load_model('kidney.sav')


# --- Sidebar for navigation ---
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Cancer Prediction',
                            'Liver Disease Prediction',
                            'Kidney Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', '⚕️', '🫁', '🫄'], # Updated icons
                           default_index=0) # Default to Diabetes Prediction

# --- Diabetes Prediction Page ---
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', value='0')
    with col2:
        Glucose = st.text_input('Glucose Level', value='0')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', value='0')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', value='0')
    with col2:
        Insulin = st.text_input('Insulin Level', value='0')
    with col3:
        BMI = st.text_input('BMI value', value='0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', value='0')
    with col2:
        Age = st.text_input('Age of the Person', value='0')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        if diabetes_model is not None:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
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
        else:
            diab_diagnosis = "Diabetes prediction model not loaded."
    st.success(diab_diagnosis)

# --- Heart Disease Prediction Page ---
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age', value='0')
    with col2:
        sex = st.text_input('Sex (0=Female, 1=Male)', value='0') # Assuming 0/1 encoding
    with col3:
        cp = st.text_input('Chest Pain types (0-3)', value='0') # Assuming 0,1,2,3 encoding
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', value='0')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', value='0')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)', value='0')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0-2)', value='0')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved', value='0')
    with col3:
        exang = st.text_input('Exercise Induced Angina (0=No, 1=Yes)', value='0')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise', value='0')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)', value='0')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy (0-3)', value='0')
    with col1:
        thal = st.text_input('thal: 0=normal, 1=fixed defect, 2=reversable defect', value='0') # Clarified encoding

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        if heart_disease_model is not None:
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
        else:
            heart_diagnosis = "Heart disease prediction model not loaded."
    st.success(heart_diagnosis)

# --- Parkinson's Prediction Page ---
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', value='0')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', value='0')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', value='0')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', value='0')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', value='0')
    with col1:
        RAP = st.text_input('MDVP:RAP', value='0')
    with col2:
        PPQ = st.text_input('MDVP:PPQ', value='0')
    with col3:
        DDP = st.text_input('Jitter:DDP', value='0')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', value='0')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', value='0')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', value='0')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', value='0')
    with col3:
        APQ = st.text_input('MDVP:APQ', value='0')
    with col4:
        DDA = st.text_input('Shimmer:DDA', value='0')
    with col5:
        NHR = st.text_input('NHR', value='0')
    with col1:
        HNR = st.text_input('HNR', value='0')
    with col2:
        RPDE = st.text_input('RPDE', value='0')
    with col3:
        DFA = st.text_input('DFA', value='0')
    with col4:
        spread1 = st.text_input('spread1', value='0')
    with col5:
        spread2 = st.text_input('spread2', value='0')
    with col1:
        D2 = st.text_input('D2', value='0')
    with col2:
        PPE = st.text_input('PPE', value='0')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        if parkinsons_model is not None:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
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
        else:
            parkinsons_diagnosis = "Parkinson's prediction model not loaded."
    st.success(parkinsons_diagnosis)

# --- Cancer Prediction Page ---
if selected == 'Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')

    if cancer_model is None:
        st.warning("Cannot perform prediction as the cancer model was not loaded. Please check the 'saved_models' directory.")
    else:
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

        # Ensure the order of attributes for prediction matches your model's training
        attribute_names_ordered = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
            "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
            "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
            "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
            "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        user_input_cancer = {}
        cols = st.columns(3) # Use 3 columns for better layout

        for i, attr in enumerate(attribute_names_ordered):
            with cols[i % 3]:
                user_input_cancer[attr] = st.text_input(attributes[attr], value='0.0') # Default value

        cancer_diagnosis = ''
        if st.button('Cancer Test Result'):
            input_data = []
            valid_input = True
            for attr in attribute_names_ordered:
                try:
                    input_data.append(float(user_input_cancer[attr]))
                except ValueError:
                    valid_input = False
                    break

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

# --- Liver Disease Prediction Page ---
if selected == 'Liver Disease Prediction':
    st.title('Liver Disease Prediction using ML')

    if liver_model is None:
        st.warning("Cannot perform prediction as the liver disease model was not loaded. Please check the 'saved_models' directory.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.text_input('Age', value='0')
        with col2:
            Gender = st.selectbox('Gender', ['Male', 'Female'])
        with col3:
            Total_Bilirubin = st.text_input('Total Bilirubin', value='0.0')
        with col1:
            Direct_Bilirubin = st.text_input('Direct Bilirubin', value='0.0')
        with col2:
            Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase', value='0')
        with col3:
            Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase', value='0')
        with col1:
            Aspartate_Aminotransferase = st.text_input('Aspartate Aminotransferase', value='0')
        with col2:
            Total_Protiens = st.text_input('Total Protiens', value='0.0')
        with col3:
            Albumin = st.text_input('Albumin', value='0.0')
        with col1:
            Albumin_and_Globulin_Ratio = st.text_input('Albumin and Globulin Ratio', value='0.0')

        liver_diagnosis = ''
        if st.button('Liver Disease Test Result'):
            # IMPORTANT: Ensure this mapping matches your trained model's encoding for 'Gender'
            gender_numeric = 1 if Gender == 'Male' else 0 # Assuming Male:1, Female:0

            # This list order MUST match the feature order your liver_model expects
            user_input_liver = [
                Age, gender_numeric, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                Albumin, Albumin_and_Globulin_Ratio
            ]

            try:
                processed_input = [float(x) for x in user_input_liver]
                liver_prediction = liver_model.predict([processed_input])

                if liver_prediction[0] == 1: # Assuming 1 for liver disease, 0 for no liver disease
                    liver_diagnosis = 'The person is predicted to have Liver Disease.'
                else:
                    liver_diagnosis = 'The person is predicted not to have Liver Disease.'
            except ValueError:
                liver_diagnosis = 'Please enter valid numeric values for all fields.'
            except Exception as e:
                liver_diagnosis = f"An error occurred during prediction: {e}"
        st.success(liver_diagnosis)

# --- Kidney Disease Prediction Page ---
if selected == 'Kidney Disease Prediction':
    st.title('Kidney Disease Prediction using ML')

    if kidney_model is None:
        st.warning("Cannot perform prediction as the kidney disease model was not loaded. Please check the 'saved_models' directory.")
    else:
        col1, col2, col3 = st.columns(3)

        # The order of these inputs MUST match the order of features your kidney_model was trained on.
        # Ensure correct default values for numeric fields and choices for selectboxes.
        with col1:
            age_kidney = st.text_input('Age', value='0')
        with col2:
            bp = st.text_input('Blood Pressure (mm/Hg)', value='0')
        with col3:
            sg = st.text_input('Specific Gravity', value='0.0')
        with col1:
            al = st.text_input('Albumin', value='0')
        with col2:
            su = st.text_input('Sugar', value='0')
        with col3:
            rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal']) # Categorical
        with col1:
            pc = st.selectbox('Pus Cells', ['normal', 'abnormal']) # Categorical
        with col2:
            pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent']) # Categorical
        with col3:
            ba = st.selectbox('Bacteria', ['present', 'notpresent']) # Categorical
        with col1:
            bgr = st.text_input('Blood Glucose Random (mg/dl)', value='0')
        with col2:
            bu = st.text_input('Blood Urea (mg/dl)', value='0')
        with col3:
            sc = st.text_input('Serum Creatinine (mg/dl)', value='0.0')
        with col1:
            sod = st.text_input('Sodium (mEq/L)', value='0.0')
        with col2:
            pot = st.text_input('Potassium (mEq/L)', value='0.0')
        with col3:
            hemo = st.text_input('Hemoglobin (g)', value='0.0')
        with col1:
            pcv = st.text_input('Packed Cell Volume', value='0')
        with col2:
            wc = st.text_input('White Blood Cell Count (cells/cmm)', value='0')
        with col3:
            rc = st.text_input('Red Blood Cell Count (millions/cmm)', value='0.0')
        with col1:
            htn = st.selectbox('Hypertension', ['yes', 'no']) # Categorical
        with col2:
            dm = st.selectbox('Diabetes Mellitus', ['yes', 'no']) # Categorical
        with col3:
            cad = st.selectbox('Coronary Artery Disease', ['yes', 'no']) # Categorical
        with col1:
            appet = st.selectbox('Appetite', ['good', 'poor']) # Categorical
        with col2:
            pe = st.selectbox('Pedal Edema', ['yes', 'no']) # Categorical
        with col3:
            ane = st.selectbox('Anemia', ['yes', 'no']) # Categorical


        kidney_diagnosis = ''
        if st.button('Kidney Disease Test Result'):
            # Convert categorical inputs to numeric - IMPORTANT: Match your model's training encoding
            rbc_numeric = 1 if rbc == 'abnormal' else 0 # Assuming normal:0, abnormal:1
            pc_numeric = 1 if pc == 'abnormal' else 0
            pcc_numeric = 1 if pcc == 'present' else 0 # Assuming notpresent:0, present:1
            ba_numeric = 1 if ba == 'present' else 0
            htn_numeric = 1 if htn == 'yes' else 0 # Assuming no:0, yes:1
            dm_numeric = 1 if dm == 'yes' else 0
            cad_numeric = 1 if cad == 'yes' else 0
            appet_numeric = 1 if appet == 'poor' else 0 # Assuming good:0, poor:1
            pe_numeric = 1 if pe == 'yes' else 0
            ane_numeric = 1 if ane == 'yes' else 0

            # This list order MUST match the feature order your kidney_model expects
            user_input_kidney = [
                age_kidney, bp, sg, al, su, rbc_numeric, pc_numeric, pcc_numeric, ba_numeric,
                bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn_numeric, dm_numeric, cad_numeric,
                appet_numeric, pe_numeric, ane_numeric
            ]

            try:
                processed_input = [float(x) for x in user_input_kidney]
                kidney_prediction = kidney_model.predict([processed_input])

                if kidney_prediction[0] == 1: # Assuming 1 for CKD, 0 for no CKD
                    kidney_diagnosis = 'The person is predicted to have Chronic Kidney Disease (CKD).'
                else:
                    kidney_diagnosis = 'The person is predicted not to have Chronic Kidney Disease (CKD).'
            except ValueError:
                kidney_diagnosis = 'Please enter valid numeric values for all fields.'
            except Exception as e:
                kidney_diagnosis = f"An error occurred during prediction: {e}"
        st.success(kidney_diagnosis)