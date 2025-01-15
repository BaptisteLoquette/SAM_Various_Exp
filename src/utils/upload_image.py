import streamlit as st
from PIL import Image
import numpy as np

def upload_image(uploaded_file):
    """
    Upload Image Fonction using streamlit
    Args:
        uploaded_file: File uploaded through streamlit
    Returns:
        tuple: (Image PIL, numpy array)
    """
    try:
        # Convertir l'image téléchargée en objet PIL Image
        image = Image.open(uploaded_file)
        image = image.resize((512, 512))
        
        # Convertir l'image en tableau numpy
        image_array = np.array(image)
        
        return image, image_array
        
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image: {str(e)}")
        return None, None