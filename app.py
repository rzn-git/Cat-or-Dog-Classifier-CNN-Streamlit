import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('CNN_model/cnn_cat_or_dog_v4.0.keras')

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize image to match the model's input size
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to the same scale as training data
    return img_array

# Streamlit app
st.title("Cat or Dog Classifier")

st.write("Upload an image to classify it as a cat or dog.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img_array = preprocess_image(image_file)

    # Make a prediction
    prediction = model.predict(img_array)

    # Interpret the result and display it
    if prediction[0] > 0.5:
        st.write(f"**Prediction:** Dog")
        st.write(f"**Confidence:** {prediction[0][0]*100:.2f}%")
    else:
        st.write(f"**Prediction:** Cat")
        st.write(f"**Confidence:** {(1 - prediction[0][0])*100:.2f}%")
