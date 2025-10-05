import streamlit as st
import pandas as pd
from src.predict import predict, load_model 

# Load trained model
model = load_model() 

# Load questions from CSV
df = pd.read_csv('data/100_Unique_QA_Dataset.csv')

# Pick random questions for sidebar
sidebar_questions = df['question'].sample(10).tolist()  # pick 10 random questions

# Sidebar display
st.sidebar.title("Sample Questions")
for q in sidebar_questions:
    st.sidebar.write(q)

# Main area
st.title("RNN Q&A Chatbot")
user_question = st.text_input("Ask your question here:")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.write("Please enter a valid question.")
    else:
        answer = predict(model, user_question)
        st.write(f"Answer: {answer}")
