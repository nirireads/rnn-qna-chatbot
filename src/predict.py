from dataset import text_to_indices, vocab
from src.model import QAModel
import torch

def predict(model, question, threshold=0.5):
    #convert question to numerical indices
    numerical_question = text_to_indices(question, vocab)

    #convert to tensor and add batch dimension
    question_tensor = torch.tensor(numerical_question).unsqueeze(0)  # Shape: (1, seq_length)

    #send to model
    model = QAModel(vocab_size=len(vocab))
    output = model(question_tensor)

    #apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)

    #get the predicted index and its probability
    value, index = torch.max(probabilities, dim=1)

    if value <threshold:
        print("Sorry, I don't know the answer to that question.")
    else:
        answer = list(vocab.keys())[index]
        print(f'Answer: {answer} (Confidence: {value.item():.2f})')
        