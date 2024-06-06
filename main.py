from utils import SmoothIrregular
from streamlit_option_menu import option_menu
import streamlit as st
from morphology import Morphology
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import joblib
import cv2
import os
from codecarbon import EmissionsTracker
from ultralytics import YOLO
import matplotlib.pyplot as plt
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
import tempfile
from pydantic.v1 import BaseSettings
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pandas as pd

st.set_page_config(page_title="BlueRev", layout="wide",
                   page_icon=("🪱"))
st.markdown("""
<style>
.first_titre {
    font-size:60px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    # text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
    text-align:left
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
</style>
""", unsafe_allow_html=True)


def run(labels):
    st.set_option("deprecation.showfileUploaderEncoding", False)

    if "files" not in st.session_state:
        st.session_state["files"] = []
        st.session_state["annotation_files"] = []
        st.session_state["image_index"] = 0

    def refresh():
        st.session_state["image_index"] = 0

    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')

    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index

    files = st.sidebar.file_uploader(label="Upload Images", type=[
                                     'jpg', 'jpeg', 'png'],
                                     accept_multiple_files=True)

    if files:
        st.session_state["files"] = files

    # Sidebar: show status
    n_files = len(st.session_state["files"])
    n_annotate_files = len(st.session_state["annotation_files"])
    st.sidebar.write("Total files:", n_files)
    st.sidebar.write("Total annotated files:", n_annotate_files)
    st.sidebar.write("Remaining files:", n_files - n_annotate_files)

    if st.session_state["files"]:
        file_select = st.sidebar.selectbox(
            "Files",
            st.session_state["files"],
            index=st.session_state["image_index"],
            on_change=go_to_image,
            key="file",
        )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=previous_image)
    with col2:
        st.button(label="Next image", on_click=next_image)

    st.sidebar.button(label="Refresh", on_click=refresh)

    # Main content: annotate images
    if st.session_state["files"]:
        img_file = st.session_state["files"][st.session_state["image_index"]]

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_file.read())
            temp_file_path = tmp.name

        im = ImageManager(temp_file_path)
        img = im.get_img()
        resized_img = im.resizing_img()
        resized_rects = im.get_resized_rects()
        rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

        def annotate():
            im.save_annotation()
            image_annotate_file_name = os.path.splitext(img_file.name)[
                0] + ".xml"
            if image_annotate_file_name not in st.session_state["annotation_files"]:
                st.session_state["annotation_files"].append(
                    image_annotate_file_name)
            next_image()

        if rects:
            st.button(label="Save")
            preview_imgs = im.init_annotation(rects)

            for i, prev_img in enumerate(preview_imgs):
                prev_img[0].thumbnail((200, 200))
                col1, col2 = st.columns(2)
                with col1:
                    morpho = Morphology()
                    image_array = np.array(Image.open(img_file))
                    scale, obj1 = morpho.get_scale(image_array)
                    obj = morpho.get_obj(np.array(prev_img[0]))
                    img_width = morpho.calculate_width(
                        image=image_array, scale=scale)
                    img_height = morpho.calculate_length(
                        image=image_array, scale=scale, obj=obj)
                    col1.image(prev_img[0])
                st.write("---")
                st.subheader("Resume")
                col11, col21 = st.columns(2)
                col11.metric("Width(micro-metre)", round(img_width, 2))
                col21.metric("Height(micro-metre)", round(img_height, 2))
                # im.set_annotation(i, select_label)

        # Clean up the temporary file
        os.remove(temp_file_path)


def get_data(uploaded_file):

    summarize = {
        "file_name": None,
        "kind": None,
        "width": None,
        "height": None,
        "oesophagus_width": None,
        "oesophagus_height": None,
        "tail_width": None,
        "tail_height": None,
        "color": None,
        "shape": None,
        "carbon_footprint": None

    }

    image = Image.open(uploaded_file)
    image_array = np.array(image)
    image_gray = image.convert("L")
    image_resized = image_gray.resize((64, 64), Image.LANCZOS)
    img_for_waste_model = np.array(image_resized)
    img_for_waste_model = img_for_waste_model.reshape(-1, 64, 64, 1)

    # SmoothIrregular
    smoothIrregular_class = SmoothIrregular()
    load_m_smooth = smoothIrregular_class.load_model(model_path="./models/Corp_lisse_ou_irregulier.h5",
                                                     scaler_path="./models/Corp_lisse_ou_irregulier.pkl")
    smoothIrregular_label = smoothIrregular_class.predict(image_array)

    # load_m_waste
    load_m_waste = tf.keras.models.load_model(
        "./models/nematode_dechet_ou_pas.h5")

    waste_or_not = load_m_waste.predict(img_for_waste_model)

    # nema_cope
    load_m_nema_cope = tf.keras.models.load_model(
        "./models/nema_cope.h5")
    nemar_or_cope = int(load_m_nema_cope.predict(
        img_for_waste_model)[0][0])
    if nemar_or_cope == 1:
        nemar_or_cope_string = "nématode"

    elif nemar_or_cope == 0:
        nemar_or_cope_string = "copépode"

    # tail oesophagus
    model = YOLO('./models/best.pt')
    results = model.predict(image)
    detected_obj = {}
    for i, result in enumerate(results):
        if hasattr(result, 'boxes') and result.boxes is not None:
            for j, (bbox, conf, cls) in enumerate(zip(result.boxes.xyxy,
                                                      result.boxes.conf,
                                                      result.boxes.cls)):
                x_min, y_min, x_max, y_max = map(int, bbox)
                class_name = [value for key,
                              value in model.names.items()][int(cls)]
                if class_name in ['oesophagus', 'tail']:
                    object_image = image_array[y_min:y_max,
                                               x_min:x_max]
                    scale, _ = morpho.get_scale(image_array)
                    obj = morpho.get_obj(np.array(object_image))
                    lenght = morpho.calculate_length(object_image,
                                                     scale,
                                                     obj)
                    witdh = morpho.calculate_width(object_image,
                                                   scale)
                    detected_obj[class_name] = [witdh, lenght]


menu = option_menu("", ["Home", "Blueview", "Laboratory",
                        "M.Learning", "---", "About"],
                   icons=['house', 'view-list', 'code-square',
                          'diagram-3-fill', None, 'file-earmark-person-fill'],
                   menu_icon="None", default_index=0,
                   styles={
    "container": {"padding": "20!important", "background-color": "",
                  "width": "100%", "gap": "20px",
                  "display": "flex"},
    "icon": {"color": "orange", "font-size": "24px"},
    "nav-link": {"font-size": "13px", "text-align": "left", "margin": "5px",
                 "--hover-color": "",
                 "display": "flex",
                 "align-items": "center"
                 },
    "nav-link-selected": {"background-color": "",
                          "font-size": "13px"},
}, orientation="horizontal"
)

if menu == "Home":
    st.markdown('<p class="first_titre">BleuRev Platform</p>',
                unsafe_allow_html=True)
    st.write("---")
    st.markdown('<p class="section">Le projet ISBLUE BlueRevolution \
        propose une nouvelle méthode pour étudier la \
        meiofaune en utilisant des technologies de \
        pointe qui incluent l’intelligence artificielle \
        et l’imagerie. Ce projet est piloté par \
        l’ENIB/Lab-STICC et l’Ifremer et inclut un réseau \
        international important de scientifiques multidisciplinaires.</p>',
                unsafe_allow_html=True)
    st.write("---")
    st.subheader("Teams")
    st.write(
        "• [BAGUIAN Harouna/GitHub](https://github.com/BAGUIAN01)")
    st.write(
        "• [Ashley HUYN/GitHub]()")
    st.write(
        "• [Hiba BOURDOUKH/GitLab](https://git.enib.fr/h2bourdo)")
    st.write(
        "• [Victor MOLEZ/GitHub]()")
    st.write(
        "• [Pedro MASI BURGOS/GitHub]()")


elif menu == "Blueview":
    with st.sidebar:
        uploaded_files = st.file_uploader(label="", type=['jpg',
                                                          'jpeg',
                                                          'png'],
                                          accept_multiple_files=True)
        st.write("---")
        files_names = [file.name for file in uploaded_files if uploaded_files]
        select_files = st.selectbox("files",
                                    options=files_names,
                                    )
        select_nema_cope = st.selectbox("Select nématode or copépode",
                                        options=["______Select here________",
                                                 "Nématode",
                                                 "Copépode"],)

        st.write("---")
        save_btn = st.button(("💾 save"))
        if save_btn:
            st.toast("Saving...", icon=("🪱"))
    # st.write(type(uploaded_file))

    if uploaded_files:
        file = uploaded_files[files_names.index(select_files)]
        summarize = {
            "file_name": None,
            "kind": None,
            "width": None,
            "height": None,
            "oesophagus_width": None,
            "oesophagus_height": None,
            "tail_width": None,
            "tail_height": None,
            "color": None,
            "shape": None,
            "carbon_footprint": None

        }
        uploaded_file = file
        summarize["file_name"] = uploaded_file.name
        image = Image.open(uploaded_file)
        model = YOLO('./models/best.pt')
        results = model.predict(image)
        draw = ImageDraw.Draw(image)

        font = ImageFont.load_default()
        for result in results:
            for box, label in zip(result.boxes.xyxy, result.names):
                xmin, ymin, xmax, ymax = box[:4]
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red")
                draw.text((xmin, ymin),
                          result.names[label], fill="red", font=font)

        col11, col22 = st.columns(2)
        col11.subheader("Original image")
        col11.image(uploaded_file)
        col22.subheader("Predicted Image with Boxes")
        col22.image(image, use_column_width=False)

        if select_nema_cope == "Nématode":
            summarize["kind"] = "Nématode"
            tracker = EmissionsTracker(project_name="bluerevCope")
            tracker.start()
            emissions = []
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            model = YOLO('./models/best.pt')
            results = model.predict(image)
            draw = ImageDraw.Draw(image)
            emissions.append(1000*tracker.stop())

            tracker.start()
            smoothIrregular_class = SmoothIrregular()
            load_m_smooth = smoothIrregular_class.load_model(model_path="./models/Corp_lisse_ou_irregulier.h5",
                                                             scaler_path="./models/Corp_lisse_ou_irregulier.pkl")
            smoothIrregular_label = smoothIrregular_class.predict(image_array)
            emissions.append(1000*tracker.stop())

            tracker.start()
            load_m_waste = tf.keras.models.load_model(
                "./models/nematode_dechet_ou_pas.h5")

            load_m_nema_cope = tf.keras.models.load_model(
                "./models/nema_cope.h5")
            image_gray = image.convert("L")
            image_resized = image_gray.resize((64, 64), Image.LANCZOS)
            img_for_waste_model = np.array(image_resized)
            img_for_waste_model = img_for_waste_model.reshape(-1, 64, 64, 1)
            waste_or_not = load_m_waste.predict(img_for_waste_model)
            nemar_or_cope = int(load_m_nema_cope.predict(
                img_for_waste_model)[0][0])
            if nemar_or_cope == 1:
                nemar_or_cope_string = "nématode"

            elif nemar_or_cope == 0:
                nemar_or_cope_string = "copépode"
            emissions.append(1000*tracker.stop())

            tracker.start()
            morpho = Morphology()
            scale, obj = morpho.get_scale(image_array)
            img_width = morpho.calculate_width(image=image_array, scale=scale)
            img_height = morpho.calculate_length(
                image=image_array, scale=scale, obj=obj)
            summarize["width"] = img_width
            summarize["height"] = img_height
            detected_obj = {}
            for i, result in enumerate(results):
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for j, (bbox, conf, cls) in enumerate(zip(result.boxes.xyxy,
                                                              result.boxes.conf,
                                                              result.boxes.cls)):
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        class_name = [value for key,
                                      value in model.names.items()][int(cls)]
                        if class_name in ['oesophagus', 'tail']:
                            object_image = image_array[y_min:y_max,
                                                       x_min:x_max]
                            scale, _ = morpho.get_scale(image_array)
                            obj = morpho.get_obj(np.array(object_image))
                            lenght = morpho.calculate_length(object_image,
                                                             scale,
                                                             obj)
                            witdh = morpho.calculate_width(object_image,
                                                           scale)
                            detected_obj[class_name] = [witdh, lenght]

            emissions.append(1000*tracker.stop())
            st.write("---")
            st.subheader(f"Resume of {select_nema_cope}")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Width(micro-metre)", round(img_width, 2))
            col2.metric("Length(micro-metre)", round(img_height, 2))
            col3.metric("Color", "dark")
            col4.metric("Shape", smoothIrregular_label)
            summarize["shape"] = smoothIrregular_label
            st.write("---")
            col5, col6, col7, col8 = st.columns(4)

            try:
                col8.metric("Width oesophagus(micro-metre)",
                            str(round(detected_obj["oesophagus"][0], 2)))
                col6.metric("Length oesophagus(micro-metre)",
                            str(round(detected_obj["oesophagus"][1], 2)))
                summarize["oesophagus_width"] = round(
                    detected_obj["oesophagus"][0], 2)
                summarize["oesophagus_height"] = round(
                    detected_obj["oesophagus"][1], 2)

            except Exception as e:
                col8.metric("Width oesophagus(micro-metre)",
                            None)
                col6.metric("Length oesophagus(micro-metre)",
                            None)

            try:
                col5.metric("Length tail(micro-metre)",
                            str(round(detected_obj["tail"][1], 2)))

                col7.metric("Width tail(micro-metre)",
                            str(round(detected_obj["tail"][0], 2)))
                summarize["tail_width"] = round(detected_obj["tail"][0], 2)
                summarize["tail_height"] = round(detected_obj["tail"][1], 2)
            except Exception as e:
                col5.metric("Length tail(micro-metre)",
                            None)

                col7.metric("Width tail(micro-metre)",
                            None)

            # col7.metric("Niveau d'enrollement", "1")
            st.write("---")
            st.subheader("carbon footprint")
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Emissions in g CO₂", str(round(np.sum(emissions), 6)))
            st.write("---")
            summarize["carbon_footprint"] = round(np.sum(emissions), 6)
            summarize_df = pd.DataFrame([summarize])
            profile = ProfileReport(summarize_df, title="Profiling Report")

            st_profile_report(profile)

        elif select_nema_cope == "Copépode":
            load_m = tf.keras.models.load_model(
                "./models/classifier_copepode_gonade.h5")
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            morpho = Morphology()
            img_width = morpho.calculate_width(image=image_array)
            img_height = morpho.calculate_length(image=image_array)

            img = image.resize((150, 150))
            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            copepode_class = int(load_m.predict(images, batch_size=10)[0][0])
            if copepode_class == 0:
                copepode_class_text = "without egg"
            elif copepode_class == 1:
                copepode_class_text = "with egg"

            st.write("---")
            st.subheader(f"Resume of {select_nema_cope}")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Width", round(img_width, 2))
            col2.metric("Length", round(img_height, 2))
            col3.metric("Color", "dark")
            col4.metric("Copepode", copepode_class_text)
            st.write("---")

elif menu == "Laboratory":
    custom_labels = ["", "Width", "Height"]
    run(custom_labels)
