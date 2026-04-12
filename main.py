import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image
import matplotlib.pyplot as plt
import wikipediaapi


def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def get_wikipedia_info(label):
    try:
        label = label.replace("_", " ").title()
        wiki_ro = wikipediaapi.Wikipedia(language="ro", user_agent="atestat-app/1.0")
        page_ro = wiki_ro.page(label)
        if page_ro.exists():
            return page_ro.summary[:500]
        wiki_en = wikipediaapi.Wikipedia(language="en", user_agent="atestat-app/1.0")
        page_en = wiki_en.page(label)
        if page_en.exists():
            return page_en.summary[:500]
        return None
    except:
        return None

def plot_predictions_chart(predictions):
    labels = [label for _, label, _ in predictions]
    scores = [score for _, _, score in predictions]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, scores, color=["#FF4B4B", "#FF8C00", "#FFC300"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Scor")
    ax.set_title("Predicții")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2%}", va="center")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Clasificator AI de imagini", page_icon="🖼️", layout="centered")

    st.title("Clasificator AI de imagini")
    st.write("Încarcă o imagine și lasă inteligența artificială să îți spună ce este în ea!")

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(uploaded_file, caption="Imagine încărcată", use_container_width=True)
        btn = st.button("Clasifică imaginea")

        if btn:
            with st.spinner("Se analizează imaginea..."):
                predictions = classify_image(model, pil_image)

                if predictions:
                    st.subheader("Predicții")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

                    st.subheader("Grafic predicții")
                    fig_chart = plot_predictions_chart(predictions)
                    st.pyplot(fig_chart)

                    st.subheader("Informații Wikipedia")
                    top_label = predictions[0][1].replace("_", " ")
                    with st.spinner("Se caută informații..."):
                        info = get_wikipedia_info(top_label)
                    if info:
                        st.write(info)
                    else:
                        st.write("Nu s-au găsit informații pe Wikipedia.")


if __name__ == "__main__":
    main()