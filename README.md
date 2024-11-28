# Chatbot with TF-IDF and Streamlit

This project implements a simple chatbot using TF-IDF (Term Frequency-Inverse Document Frequency) for text similarity and Streamlit for the web interface. The chatbot responds to user questions by finding the most relevant answer from a dataset stored in a Google Sheet.

## Features

- TF-IDF-based Similarity: Finds the closest matching response to user input using cosine similarity.
- Streaming Responses: Simulates typing by streaming the bot's response one word at a time.
- Session History: Maintains a chat history using Streamlit's session state.
- Dynamic Dataset: Automatically imports and preprocesses data from a Google Sheet.

## Prerequisites
- Python 3.7+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - streamlit

## Install dependencies 
- pip install pandas numpy scikit-learn streamlit.


## Dataset
The chatbot uses a dataset hosted on Google Sheets. The dataset must have the following columns:

Question: A column containing example questions.
Answer: A column containing corresponding answers.

The dataset is loaded from the URL specified in the csv_url variable in the data_import function.

## How It Works
1. Data Import: The dataset is fetched and preprocessed (missing values are dropped).

2. TF-IDF Vectorizer: A TfidfVectorizer is trained on both the questions and answers.

3. Cosine Similarity: The chatbot calculates the similarity between the user's input and dataset questions to find the closest match.

4. Streaming Responses: Bot responses are displayed word by word with a short delay for a typing effect.

5. Session History: The chat history is saved and displayed using Streamlit's session state.

## How to Run
1. Save the script as chatbot.py.

2. Launch the app using Streamlit:
- streamlit run chatbot.py

3. Open the app in your browser at the URL provided by Streamlit (e.g., http://localhost:8501).


The chatbot interface includes:

A text input box for user queries.
Streaming responses for enhanced user experience.
A persistent conversation history.

## Example Dataset Format
![image](https://github.com/user-attachments/assets/8cbb9ac8-fab1-4753-adb5-17c2f5739904)

