import streamlit as st
import pickle
import re
import base64
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="AI-Powered Fake News Analyzer",
    page_icon="📰"  # You can use an emoji or a custom image URL
)

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction



# Function to set background
def set_background(local_image_path):
    with open(local_image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_background("Image.jpg")  # Change to your image path

def set_styles():
    st.markdown(
        """
        <style>
        h1, h2 {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



if __name__ == '__main__':
    #set_styles()  # Apply the CSS styles
    
    st.title('AI-Powered Fake News Analyzer')
    st.subheader("Is This News Real? Enter Below")

    sentence = st.text_area("", "Paste Your News Article Here", height=200)
    predict_btt = st.button("🚀 Check Authenticity")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Reliable')
        if prediction_class == [1]:
            st.warning('Unreliable')