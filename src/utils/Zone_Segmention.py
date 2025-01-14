import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Upload Image
st.title("Select a Zone of the Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
     
    image = Image.open(uploaded_file)     # Load the image

    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(image)     # Convert PIL Image to np.array

    height, width, _ = img_array.shape

    st.subheader("Define Crop Area")
    x_start = st.slider("Start X", 0, width, 0)
    x_end = st.slider("End X", 0, width, width)
    y_start = st.slider("Start Y", 0, height, 0)
    y_end = st.slider("End Y", 0, height, height)

    if x_start < x_end and y_start < y_end:

        cropped_image = img_array[y_start:y_end, x_start:x_end] # Crop the image

        # Display the cropped area
        st.subheader("Cropped Image") # Display the cropped area
        st.image(cropped_image, use_column_width=True)
    else:
        st.error("Invalid crop area")
