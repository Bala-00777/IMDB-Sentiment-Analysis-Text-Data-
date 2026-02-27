import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review below:")

user_input = st.text_area("Review Text")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)

        if prediction[0] == "positive":
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
    else:
        st.warning("Please enter some text.")