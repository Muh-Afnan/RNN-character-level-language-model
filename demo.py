from src.rnn import *
from torch import nn
from torch.utils.data import DataLoader

with open(
    "para.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

tokenizer = CharTokenizer()
dataset = tokenizer.seq_generator(text=text,window=6)
vocab_size = len(tokenizer.vocab)
gen = DataGenerator()
x,y = gen.generate_pair(dataset)
x= torch.tensor(x,dtype=torch.long)
y=torch.tensor(y,dtype=torch.long)
x_train_len = int(
    len(x) * 0.8
)

x_train = x[:x_train_len]
y_train = y[:x_train_len]

x_test = x[x_train_len:]
y_test = y[x_train_len:]
train_dataset = CustomDataset(
    x_train,
    y_train
)

test_dataset = CustomDataset(
    x_test,
    y_test
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False
)

loss_fn = nn.CrossEntropyLoss()
epochs = 500
model = RNNFromScratch(vocab_size=vocab_size,embedding_dim=8,hidden_size=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.fit(train_loader,epochs,loss_fn,optimizer)
