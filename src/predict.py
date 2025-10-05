import torch
from src.dataset import text_to_indices, vocab
from src.model import QAModel
import os

def load_model(model_path='data/qa_model.pth'):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'qa_model.pth')
    model = QAModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model

def predict(model, question, threshold=0.5):
    #convert question to numerical indices
    numerical_question = text_to_indices(question, vocab)

    #convert to tensor and add batch dimension
    question_tensor = torch.tensor(numerical_question).unsqueeze(0)  # Shape: (1, seq_length)

    #model from streamlit app
    output = model(question_tensor)

    #apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)

    #get the predicted index and its probability
    value, index = torch.max(probabilities, dim=1)

    if value <threshold:
        answer = list(vocab.keys())[index]
        return f"Sorry, I don't know the answer to that question.?"
    else:
        answer = list(vocab.keys())[index]
        return f'{answer} (Confidence: {value.item():.2f})'
