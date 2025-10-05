from model import QAModel
import torch
from dataset import vocab, dataloader

# Hyperparameters
LR = 0.001
EPOCHS = 20

# Initialize model, loss function, and optimizer
model = QAModel(len(vocab))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0

    for question, answer in dataloader:
        optimizer.zero_grad()
        output = model(question)
        # print(f"question: {question} and answer: {answer}")
        loss = criterion(output, answer[0])
        # print(f"output: {output} and loss: {loss}")
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
    
    print(f'Epoch: {epoch} , Loss: {total_loss}')

# Save the trained model
torch.save(model.state_dict(), 'data/qa_model.pth')