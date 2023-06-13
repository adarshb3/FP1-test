import streamlit as st
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import numpy as np
import PyPDF2

# Load a dummy dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Convert the feature data to a list of strings
X_text = [str(features) for features in X]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Train the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the SVM model
model = SVC()
model.fit(X_train_vectorized, y_train)

# Save the trained model and vectorizer to files
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the model and vectorizer from files
loaded_model = joblib.load('svm_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app code
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Job Acceptance Prediction")
    
    uploaded_file = st.file_uploader("Upload your resume", type="pdf")
    
    if uploaded_file is not None:
        resume_contents = extract_text_from_pdf(uploaded_file)
        resume_vectorized = loaded_vectorizer.transform([resume_contents])
        prediction = loaded_model.predict(resume_vectorized)
        
        st.write("Prediction:", prediction[0])
        st.write("Confidence Score:", np.max(model.decision_function(resume_vectorized)))
    
if __name__ == "__main__":
    main()
