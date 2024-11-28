import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
from time import sleep

st.title("Carrefour :blue[ChatBot] 🤖")

st.caption("This is a simple Carrefour query bot PoC.")
st.caption("Author: Ashner Novilla :sunglasses:")

@st.cache_data
def data_import():
    """
    Import dataset from Google Sheets and preprocess it.
    """
    csv_url = "https://docs.google.com/spreadsheets/d/1b_pcI_IDs93QfdnoStfa8KJZj8eg_mjE8QiZXingLgE/export?format=csv&gid=515934021"

    # Read the CSV
    df = pd.read_csv(csv_url, sep=',', engine='python')

    # Drop missing values
    df.dropna(inplace=True)

    return df

@st.cache_data
def train_vectorizer(df):
    """
    Train a TF-IDF vectorizer on the questions and answers from the dataset.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(np.concatenate((df['Question'].values, df['Answer'].values)))

    question_vectors = vectorizer.transform(df['Question'].values)
    return vectorizer, question_vectors

def find_closest_response(input_question, vectorizer, question_vectors, df):
    """
    Find the closest response from the dataset using cosine similarity.
    """
    input_question_vector = vectorizer.transform([input_question])
    similarities = cosine_similarity(input_question_vector, question_vectors)
    closest_index = np.argmax(similarities, axis=1)[0]
    return df['Answer'].iloc[closest_index]

def stream_data(response_chat):
    """
    Stream response one word at a time.
    """
    for word in response_chat.split(" "):
        yield word + " "
        sleep(0.02)

# Load data and train vectorizer
df = data_import()
vectorizer, question_vectors = train_vectorizer(df)

# Initialize the conversation history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Input prompt
prompt = st.chat_input("Ask the bot something (type 'quit' to stop)")
if prompt:
    if prompt.lower() == "quit":
        st.write("**Chatbot session ended. Refresh the page to start a new conversation.**")
    else:
        # Add the user prompt to the history
        st.session_state["history"].append({"role": "user", "message": prompt})

        # Generate the bot's response based on input
        bot_response = find_closest_response(prompt, vectorizer, question_vectors, df)
        st.session_state["history"].append({"role": "bot", "message": bot_response})

# Display the conversation history with streaming
for entry in st.session_state["history"]:
    if entry["role"] == "user":
        st.chat_message("user").write(entry["message"])
    elif entry["role"] == "bot":
        # Stream the bot's response dynamically
        placeholder = st.chat_message("assistant").empty()
        streamed_text = ""
        for chunk in stream_data(entry["message"]):
            streamed_text += chunk
            placeholder.write(f"**Bot:** {streamed_text}")
