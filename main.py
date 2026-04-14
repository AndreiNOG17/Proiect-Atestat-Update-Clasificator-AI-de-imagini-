import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import wikipediaapi
import time

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    img = np.array(img, dtype=np.float32)
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
        label_cautat = label.replace("_", " ").title()
        wiki_ro = wikipediaapi.Wikipedia(language="ro", user_agent="atestat-app/1.0")
        page_ro = wiki_ro.page(label_cautat)
        if page_ro.exists():
            return page_ro.summary[:500], page_ro.fullurl
        wiki_en = wikipediaapi.Wikipedia(language="en", user_agent="atestat-app/1.0")
        page_en = wiki_en.page(label_cautat)
        if page_en.exists():
            langlinks = page_en.langlinks
            if "ro" in langlinks:
                titlu_ro = langlinks["ro"].title
                page_ro2 = wiki_ro.page(titlu_ro)
                if page_ro2.exists():
                    return page_ro2.summary[:500], page_ro2.fullurl
            return page_en.summary[:500], page_en.fullurl
        return None, None
    except:
        return None, None

def get_image_stats(image):
    img_rgb = image.convert("RGB")
    stat = ImageStat.Stat(img_rgb)
    brightness = sum(stat.mean) / 3
    return round(brightness, 1)

def plot_predictions_chart(predictions):
    labels = [label for _, label, _ in predictions]
    scores = [score for _, _, score in predictions]
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#5C4FC7", "#8B80E0", "#C5C0F0"]
    bars = ax.barh(labels, scores, color=colors, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Scor", fontsize=11)
    ax.set_title("Predicții", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2%}", va="center", fontsize=10)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    return fig

def confidence_label(score):
    if score >= 0.90:
        return "🟢 Foarte ridicată"
    elif score >= 0.60:
        return "🟡 Medie"
    else:
        return "🔴 Scăzută"

def main():
    st.set_page_config(page_title="Clasificator AI de imagini", page_icon="🔍", layout="centered")

    st.markdown("""
        <style>
        .stButton > button {background-color: #5C4FC7; color: white; border-radius: 8px; border: none; width: 100%;}
        .stButton > button:hover {background-color: #4a3eb5;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🔍 Clasificator AI de imagini")
    st.caption("Încarcă o imagine și lasă inteligența artificială să îți spună ce este în ea.")

    with st.expander("ℹ️ Despre"):
        st.write("**Nume:** Cicortaș Andrei")
        st.write("**An:** 2026")
        st.write("**Proiect de atestat**")

    st.divider()

    if "istoric" not in st.session_state:
        st.session_state.istoric = []

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Alege o imagine...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(uploaded_file, caption="Imagine încărcată", use_container_width=True)

        w, h = pil_image.size
        brightness = get_image_stats(pil_image)

        c1, c2, c3 = st.columns(3)
        c1.caption(f"📄 {uploaded_file.name}")
        c2.caption(f"📐 {w}×{h} px")
        c3.caption(f"☀️ Luminozitate: {brightness}/255")

        st.divider()

        btn = st.button("Clasifică imaginea")
        if btn:
            start_time = time.time()

            with st.spinner("Se analizează imaginea..."):
                predictions = classify_image(model, pil_image)

            elapsed = time.time() - start_time

            if predictions:
                top_label = predictions[0][1].replace("_", " ").title()
                top_score = predictions[0][2]

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicție principală", top_label)
                col2.metric("Confidență", f"{top_score:.1%}")
                col3.metric("Clase analizate", "1000")

                st.progress(float(top_score), text=f"{confidence_label(top_score)} · {top_score:.1%}")
                st.caption(f"Timp de procesare: {elapsed:.2f} secunde")

                st.divider()

                st.subheader("Predicții")
                for _, label, score in predictions:
                    st.write(f"**{label.replace('_', ' ').title()}**: {score:.2%}")

                st.subheader("Grafic predicții")
                fig_chart = plot_predictions_chart(predictions)
                st.pyplot(fig_chart)

                st.subheader("Informații Wikipedia")
                with st.spinner("Se caută informații..."):
                    info, url = get_wikipedia_info(predictions[0][1])
                if info:
                    st.info(info)
                    st.markdown(f"[Citește mai mult pe Wikipedia →]({url})")
                else:
                    st.write("Nu s-au găsit informații pe Wikipedia.")

                st.session_state.istoric.insert(0, {
                    "nume": uploaded_file.name,
                    "predictie": top_label,
                    "confidenta": f"{top_score:.1%}"
                })
                st.session_state.istoric = st.session_state.istoric[:3]

    if st.session_state.istoric:
        st.divider()
        st.subheader("Istoric sesiune")
        for item in st.session_state.istoric:
            st.write(f"🖼️ **{item['nume']}** → {item['predictie']} ({item['confidenta']})")

if __name__ == "__main__":
    main()
