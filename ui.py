import io
import streamlit as st
import cv2
import numpy as np
import keras
from PIL import Image

model = keras.models.load_model('plant_disease_classification_plant_village.keras')

Labels = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy", "No Leaf Image", "Healthy", "Powdery Mildew", "Healthy", "Cercospora Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy", "Black Rot", "Esca (Black Measles)", "Leaf Blight (Isariopsis Leaf Spot)", "Healthy", "Haunglongbing (Citrus Greening)", "Bacterial Spot", "Healthy", "Bacterial Spot", "Healthy", "Early Blight", "Late Blight", "Healthy", "Healthy", "Healthy", "Powdery Mildew", "Leaf Scorch", "Healthy", "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", "Spider Mites (Two Spotted Spider Mite)", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"]


def preprocess_input_image(image):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    return img


def app():
    st.set_page_config(layout="wide")
    st.subheader("Hi! 👋")
    st.title("Welcome to LeafMD 🌱")
    st.write("🌿 Harnessing the power of advanced machine learning, our web application accurately predicts plant "
             "leaf diseases from images.")

    # Column layout
    col1, col2 = st.columns([1, 2])

    # Left column for image upload and prediction
    with col1:
        st.subheader("Get Predictions :")
        # Add your image upload and prediction code here
        uploaded_file = st.file_uploader("Choose an image", type=["png", "JPG", "jpeg"])
        if uploaded_file is not None:
            if st.button("Predict"):
                processed_img = preprocess_input_image(uploaded_file)
                # get the prediction
                prediction = model.predict(processed_img)
                prediction_class = np.argmax(prediction)
                percentage = "{:.2f}".format(prediction[0][prediction_class]*100)
                predicted_label = Labels[prediction_class]
                st.toast("Prediction Successful")
                st.success("Prediction : {}".format(predicted_label))
                st.success("There is a {}% chance that your plant is dealing with the above given prediction.".format(percentage))

                st.session_state['predicted_label'] = True

    # Right column for displaying supported plants
    with col2:
        st.subheader("We currently support the following plants:")
        plants = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Bell Pepper", "Potato",
                  "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato", ]
        html_code = "<div style='column-count: 3;'>"
        for plant in plants:
            html_code += f"<li>{plant}</li>"
        html_code += "</div>"
        st.write(html_code, unsafe_allow_html=True)
        if uploaded_file is not None and st.session_state.get('predicted_label', False):
            st.success("Thank You for using LeafMD !")


if __name__ == '__main__':
    app()
