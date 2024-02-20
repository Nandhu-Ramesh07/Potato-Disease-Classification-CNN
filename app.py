import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def identify(img_path, model):

    class_names = {
        0: 'Early blight',
        1: 'Late blight',
        2: 'Healthy'
    }

    # Open the image and convert it to RGB mode
    img = Image.open(img_path).convert('RGB')
    
    # Resize the image to the required dimensions
    img = img.resize((256, 256))
    
    # Convert the image to a numpy array and normalize the pixel values
    img_array = np.array(img) / 255.0
    
    # Expand the dimensions to match the model's input shape
    input_img = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    res = model.predict(input_img)
    
    # Get the index of the predicted class
    predicted_class_index = np.argmax(res)
    
    # Get the name of the predicted class from the class_names dictionary
    predicted_class_name = class_names.get(predicted_class_index, 'Unknown')
    
    return predicted_class_name

# Streamlit App
st.title("Potato-Disease-Classification")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    model = load_model('potatoes_CNN.h5')
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
    st.write("")

    if st.button("Predict the Disease"):
        result = identify(uploaded_file, model)
        st.subheader("Image Result") 
        st.write(f"**{result}**")
