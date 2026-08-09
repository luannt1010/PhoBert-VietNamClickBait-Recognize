import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Vietnamese Clickbait Detection", layout="centered")

st.title("Vietnamese Clickbait Detection")
st.write("Choose one of the prediction methods below.")
tab1, tab2, tab3 = st.tabs(["Predict from Title", "Predict from URL", "Predict from Upload File"])

with tab1:
    with st.form("title_form"):
        title = st.text_input("Enter the article title")
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not title.strip():
            st.warning("You haven't entered a title.")
        else:
            with st.spinner("Predicting..."):
                response = requests.post(f"{API_URL}/predict/title", json={"title": title})
            if response.status_code == 200:
                result = response.json()
                st.subheader("Results")
                st.write(f"**Sentence:** {result['sentence']}")
                st.write(f"**Score:** {result['score']}")
                if result["label"] == "clickbait":
                    st.error(f"Prediction: {result['label']}")
                else:
                    st.success(f"Prediction: {result['label']}")
            else:
                st.error(response.json()["detail"])

with tab2:
    with st.form("url_form"):
        url = st.text_input("Enter the article url")
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not url.strip():
            st.warning("You haven't entered a URL.")
        else:
            with st.spinner("Predicting..."):
                response = requests.post(f"{API_URL}/predict/url", json={"url": url})
            if response.status_code == 200:
                result = response.json()
                st.subheader("Results")
                st.write(f"**Sentence:** {result['sentence']}")
                st.write(f"**Score:** {result['score']}")
                if result["label"] == "clickbait":
                    st.error(f"Prediction: {result['label']}")
                else:
                    st.success(f"Prediction: {result['label']}")
            else:
                st.error(response.json()["detail"])

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
                response = requests.post(f"{API_URL}/predict/file", files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)})
            if response.status_code == 200:
                result = response.json()
                st.dataframe(pd.DataFrame(result))
            else:
                st.error(response.json()["detail"])

