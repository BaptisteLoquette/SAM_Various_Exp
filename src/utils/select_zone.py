from streamlit_extras.image_selector import image_selector
from streamlit_extras.image_selector import show_selection
from PIL import Image
import streamlit as st

def select_bbox_zone(image):
    """
    Slect Zone Fonction either with lasso or box
    Args:
        image (PIL.Image): Image on which we'll select the wanted zone
    Returns:
        dict: Json result of selection
    """
    selection_type = st.radio(
        "Selection Type", ["lasso", "box"], index=1, horizontal=True
    )

    selection = image_selector(image=image, selection_type=selection_type, width = 500, height = 500)
    if selection:
        #show_selection(image, selection)
        return selection
    return None