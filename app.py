import pandas as pd
import streamlit as st
from clickbait_detector.inference import ClickBaitPredictor

st.set_page_config(page_title="Vietnamese Clickbait Detection", layout="centered")

@st.cache_resource
def load_predictor():
    predictor = ClickBaitPredictor(config_dir=r".\configs\phobert-base-v2",
                                   weight_path=r".\artifacts\models\last.pth",
                                   max_len=256, threshold=0.5)
    return predictor

predictor = load_predictor()
st.title("Vietnamese Clickbait Detection")
st.write("Choose one of the prediction methods below.")
tab1, tab2, tab3 = st.tabs(["Predict from Title", "Predict from URL", "Predict from CSV"])

with tab1:
    with st.form("title_form"):
        title = st.text_input("Enter the article title")
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not title.strip():
            st.warning("You haven't entered a title.")
        else:
            with st.spinner("Predicting..."):
                result = predictor.predict_one_title(title)
            if result is None:
                st.error("Unpredictable.")
            else:
                st.subheader("Results")
                st.write(f"**Sentence:** {result['Sentence']}")
                st.write(f"**Score:** {result['Score']}")
                if result["Label"] == "clickbait":
                    st.error(f"Prediction: {result['Label']}")
                else:
                    st.success(f"Prediction: {result['Label']}")

with tab2:
    with st.form("url_form"):
        url = st.text_input("Enter the article url")
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not url.strip():
            st.warning("You haven't entered a URL.")
        else:
            with st.spinner("Predicting..."):
                result = predictor.predict_url(url)
            if result is None:
                st.error("Unpredictable.")
            else:
                st.subheader("Results")
                st.write(f"**Sentence:** {result['Sentence']}")
                st.write(f"**Score:** {result['Score']}")
                if result["Label"] == "clickbait":
                    st.error(f"Prediction: {result['Label']}")
                else:
                    st.success(f"Prediction: {result['Label']}")

with tab3:
    uploaded_file = st.file_uploader("Drag and drop or browse the CSV or EXCEL file.", type=["csv", "xlsx"],
                                     help="Streamlit supports both Browse and Drag & Drop.")
    if uploaded_file is not None:
        suffix = uploaded_file.name.split('.')[-1]
        if suffix.lower() == "xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success("File read successful!")
        st.write("The first 5 lines of the file")
        st.write(df.head())
        if st.button("Predict File"):
            with st.spinner("Predicting..."):
                results = predictor.predict_file(df)
            st.dataframe(results)

