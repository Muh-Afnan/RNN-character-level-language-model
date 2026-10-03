from src.rnn import *
from torch import nn
from torch.utils.data import DataLoader
import urllib.request


# with open(
#     "para.txt",
#     "r",
#     encoding="utf-8"
# ) as file:

#     text = file.read()

url = "https://www.gutenberg.org/files/11/11-0.txt"  # Alice in Wonderland
text = urllib.request.urlopen(url).read().decode('utf-8')
text = text[1000:50000]  # skip the header, take ~49k characters

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
epochs = 1000
model = RNNFromScratch(vocab_size=vocab_size,embedding_dim=8,hidden_size=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.fit(train_loader,epochs,loss_fn,optimizer)


def generate(seed, length, model:RNNFromScratch,tokenizer:CharTokenizer):
    start = tokenizer.preprocess(seed)
    start_idx = tokenizer.char2idx_map[start]
    string = start
    h = None
    for i in range(length):
        start_tensor = torch.tensor([[start_idx]])
        next_idx,h = model.predict(start_tensor,h)
        char = tokenizer.idx2char_map[next_idx.item()]
        string = string+char
        print(f"Input: {start}, Output: {char}")
        print(f"Full String:{string}")
        start_idx = next_idx
        start = tokenizer.idx2char_map[next_idx.item()]
        print("I ran")

generate(seed="f",length=50,model=model,tokenizer=tokenizer)