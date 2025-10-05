from config import LR, EPOCHS
from src.model import QAModel
import torch
from dataset import vocab, dataloader

# Initialize model, loss function, and optimizer
model = QAModel(vocab_size=len(vocab))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0

    for question, answer in dataloader:
        optimizer.zero_grad()
        output = model(question)
        loss = criterion(output, answer[0])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f'Epoch: {epoch} , Loss: {total_loss}')

# Save the trained model
torch.save(model.state_dict(), 'data/qa_model.pth')