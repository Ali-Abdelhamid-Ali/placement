import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

@st.cache_data  
def load_model():
    url = "https://github.com/Ali-Abdelhamid-Ali/placement/blob/main/placement.pkl?raw=true"
    response = requests.get(url)
    
    if response.status_code == 200:
        model = pickle.load(io.BytesIO(response.content))  
        return model
    else:
        st.error("Failed to download the model.")
        return None

best_model = load_model()

if best_model:
    st.write("Model loaded successfully!")
else:
    st.write("Failed to load model.")


def classify_placement(CGPA, Internships, Projects, Workshops_Certifications, AptitudeTestScore,
                       SoftSkillsRating, ExtracurricularActivities, PlacementTraining, SSC_Marks, HSC_Marks):
    input_data = (CGPA, Internships, Projects, Workshops_Certifications, AptitudeTestScore,
                  SoftSkillsRating, ExtracurricularActivities, PlacementTraining, SSC_Marks, HSC_Marks)
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = best_model.predict(input_data_reshaped)
    
    return prediction

def visualize_data(CGPA, prediction):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Not Placed', 'Placed'], [1 - prediction, prediction], color=['red', 'green'])
    ax.set_title(f'Placement Prediction for CGPA: {CGPA}')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

def main():
    st.title("Placement Prediction App")
    st.write("Predicts student placement success based on academic performance and skills.")

    CGPA = st.slider("CGPA", 0.0, 10.0, 7.0, 0.01)
    Internships = st.slider("Number of Internships", 0, 20, 3)
    Projects = st.slider("Number of Projects", 0, 100, 10)
    Workshops_Certifications = st.slider("Number of Workshops and Certifications", 0, 10, 4)
    AptitudeTestScore = st.slider("Aptitude Test Score", 0, 100, 70)
    SoftSkillsRating = st.slider("Soft Skills Rating", 0, 5, 2)
    ExtracurricularActivities = st.slider("Extracurricular Activities", 0, 5, 1)
    PlacementTraining = st.slider("Placement Training", 1, 8, 0)
    SSC_Marks = st.slider("SSC Marks", 0, 100, 75)
    HSC_Marks = st.slider("HSC Marks", 0, 100, 80)
    
    if st.button("Predict"):
        prediction = classify_placement(CGPA, Internships, Projects, Workshops_Certifications, AptitudeTestScore,
                                       SoftSkillsRating, ExtracurricularActivities, PlacementTraining, SSC_Marks, HSC_Marks)
        
        if prediction[0] == 1:
            st.write("The student will be placed.")
        else:
            
            st.markdown("The student will <span style='color:red;'>NOT</span> be placed.", unsafe_allow_html=True)
        
        
        visualize_data(CGPA, prediction[0])

main()
