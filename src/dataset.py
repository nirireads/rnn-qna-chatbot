import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#load data
df = pd.read_csv('data/100_Unique_QA_Dataset.csv')

#tokenizer function
def tokenize(text):
    text = text.lower()
    text = text.replace('?','')
    text = text.replace('.','')
    text = text.replace(',','')
    text = text.replace('!','')
    text = text.replace('"','')
    tokens = text.split()
    return tokens

#build vocabulary from dataframe
vocab = {'<UNK>':0}

def build_vocab(row):
    #have row['question'] and row['answer']
    tokenized_question = tokenize(row['question'])
    tokenized_answer = tokenize(row['answer'])
    merged_tokens = tokenized_question + tokenized_answer

    for token in merged_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

# calling build_vocab on each row
df.apply(build_vocab, axis=1)

#convert words to numerical indices
def text_to_indices(text,vocab):
    indexed_text = []
    tokenized_text = tokenize(text)

    for token in tokenized_text:
        if(token in vocab):
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab['<UNK>'])
    return indexed_text


# Define custom dataset class
class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        # convert question and answer to numerical indices 
        numerical_question = text_to_indices(self.df.iloc[index]['question'],self.vocab)
        numerical_answer = text_to_indices(self.df.iloc[index]['answer'], self.vocab)

        return torch.tensor(numerical_question), torch.tensor(numerical_answer)
    
# Create dataset instance
dataset = QADataset(df, vocab)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for question,answer in dataloader:
#   print(question, answer)
# tensor([[  1,   2,   3, 222,   5, 223, 224, 225]]) tensor([[226]])