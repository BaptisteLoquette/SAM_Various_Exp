import matplotlib.pyplot as plt
import numpy as np
from utils.sam_utils import PREDICTOR
from PIL import Image
import streamlit as st
from  utils.select_zone import select_bbox_zone
from utils.upload_image import upload_image
from utils.sam_utils import PREDICTOR
from streamlit_extras.image_selector import show_selection

def show_segmentation(image_np, predicted_masks):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    # Assuming predicted_masks is a binary mask with the same height and width as image_np
    ax.imshow(predicted_masks, cmap='jet', alpha=0.5)  # Overlay the mask with transparency
    ax.axis('off')  # Hide axes
    st.pyplot(fig)  # Display the figure in Streamlit

def main():
    """
    Main function to display the Segmented Image using different methods
    """

    st.title("Image Visulation")

    uploaded_file = st.file_uploader("Choose an Image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Passer le fichier téléchargé à la fonction upload_image
        result = upload_image(uploaded_file)
        if result:
            image, image_array = result

        PREDICTOR.set_image(image_array)

        json_results = select_bbox_zone(image)

        pos_bboxes = json_results["selection"]["box"]

        if pos_bboxes:
            xs = pos_bboxes[0]['x']
            ys = pos_bboxes[0]['y']

            margin = 50  # pixels de marge

            # Inverser l'ordre des coordonnées x et y pour la sélection
            bbox_x = [
                max(0, int(np.ceil(ys[0])) - margin),
                min(image_array.shape[1], int(np.ceil(ys[1])) + margin)
            ]
            bbox_y = [
                max(0, int(np.ceil(xs[0])) - margin),
                min(image_array.shape[0], int(np.ceil(xs[1])) + margin)
            ]

            input_box = np.array([bbox_x[0], bbox_y[0], bbox_x[1], bbox_y[1]])


            masks, scores, logits = PREDICTOR.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            show_segmentation(image_array, masks[0])

            # Découper l'image avec les coordonnées correctes
            bbox_array = image_array[bbox_y[0]:bbox_y[1], bbox_x[0]:bbox_x[1], :]

            st.image(bbox_array)

            # Convertir en uint8 pour l'affichage
            predicted_masks = masks[0]
            predicted_masks = (predicted_masks * 255).astype(np.uint8)
            predicted_masks = predicted_masks.reshape(predicted_masks.shape[0], predicted_masks.shape[1], 1)
            predicted_masks = np.repeat(predicted_masks, 3, axis=2)

            st.image(predicted_masks, caption="Mask Prectidted")
            
if __name__ == "__main__":
    main()
