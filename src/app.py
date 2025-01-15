import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from  utils.select_zone import select_bbox_zone
from utils.upload_image import upload_image
from utils.sam_utils import get_sam_model

def show_segmentation(image_np, predicted_masks):
    fig, ax = plt.subplots(figsize=(20, 20))
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

        st.session_state.sam_model_choice = False

        sam_model_type = st.radio(
        "Choose SAM Model Size", ["vit_b", "vit_l", "vit_h"], index=None, horizontal=True)

        if sam_model_type:
            st.session_state.sam_model_choice = True
            PREDICTOR = get_sam_model(model_type = sam_model_type)
            PREDICTOR.set_image(image_array)

        if st.session_state.sam_model_choice:
            st.session_state.task_selection = None

            task_selection = st.radio(
            "Choose task", ["full_image", "bbox"], index=None, horizontal=True)

            if task_selection == "full_image":
                st.session_state.task_selection = "full_image"

            if task_selection == "bbox":
                st.session_state.task_selection = "bbox"
                
            if st.session_state.task_selection == "full_image":
                st.image(image_array)

                # Créer une boîte englobante pour l'image entière
                h, w = image_array.shape[:2]
                box = np.array([0, 0, w, h])
                
                masks, scores, logits = PREDICTOR.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False,
                )

                show_segmentation(image_array, masks[0])

                
            elif st.session_state.task_selection == "bbox":
                st.session_state.task_selection = "bbox"
                # Select Zone
                json_results = select_bbox_zone(image)

                pos_bboxes = json_results["selection"]["box"]

                if pos_bboxes:
                    xs = pos_bboxes[0]['x']
                    ys = pos_bboxes[0]['y']

                    # Inverser l'ordre des coordonnées x et y pour la sélection
                    bbox_x = [
                        max(0, int(np.ceil(ys[0]))),
                        min(image_array.shape[1], int(np.ceil(ys[1])))
                    ]
                    bbox_y = [
                        max(0, int(np.ceil(xs[0]))),
                        min(image_array.shape[0], int(np.ceil(xs[1])))
                    ]

                    # Découper l'image avec les bonnes coordonnées
                    bbox_array = image_array[bbox_y[0]:bbox_y[1], bbox_x[0]:bbox_x[1], :]


                    masks, scores, logits = PREDICTOR.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox_array[None, :],
                        multimask_output=False,
                    )

                    show_segmentation(bbox_array, masks[0])

                    #st.image(bbox_array)

                    # Convertir en uint8 pour l'affichage
                    predicted_masks = masks[0]
                    predicted_masks = (predicted_masks * 255).astype(np.uint8)
                    predicted_masks = predicted_masks.reshape(predicted_masks.shape[0], predicted_masks.shape[1], 1)
                    predicted_masks = np.repeat(predicted_masks, 3, axis=2)

                    st.image(predicted_masks, caption="Mask Prectidted")
            
if __name__ == "__main__":
    main()
