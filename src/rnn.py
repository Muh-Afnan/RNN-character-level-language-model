import torch
from torch import nn
from torch.utils.data import Dataloader,Dataset


class CharTokenizer():
    def __init__(self):
        self.vocab = None
        self.char2idx_map = None
        self.idx2char_map = None

    def preprocess(self,text:str):
        text = text.lower()
        chars = list(text)
        return chars

    def char2idx(self,text:str):
        chars = self.preprocess(text)
        self.vocab = sorted(set(chars))
        self.char2idx_map  = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char_map = {idx: char for char, idx in self.char2idx_map.items()}
        return self.vocab, self.char2idx_map, self.idx2char_map
    
    def idx2char(self,idx:int):
        return self.idx2char_map[idx]

    def seq_generator(self,text:str,window:int):
        text_pro = self.preprocess(text)
        vocab, char2idx_map, idx2char_map = self.char2idx(text)
        seqs = []
        for idx in range(len(text_pro) - window):
            input = text_pro[idx:idx+window]
            output = text_pro[idx+1:idx+window+1]
            input_idx = [char2idx_map[char] for char in input]
            output_idx = [char2idx_map[char] for char in output]
            seq = (input_idx,output_idx)
            seqs.append(seq)
        return seqs

class RNNFromScratch(nn.Module):
    def __init__(self,vocab_size:int, embedding_dim:int, hidden_size:int,):
        super().__init__()
        if min(vocab_size, embedding_dim,hidden_size)<=0:
            raise ValueError("All sizes must be positive.")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.W_xh = nn.Parameter(torch.empty(embedding_dim,hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.empty(hidden_size,vocab_size))
        self.b_y = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.orthogonal_(self.W_hh)
        nn.init.xavier_uniform_(self.W_hy)

    def forward(self,x: torch.Tensor,h0: torch.Tensor | None = None,) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(
                "x must have shape (batch_size, seq_len)."
            )

        if x.dtype != torch.long:
            raise TypeError(
                "x must contain character IDs as torch.long."
            )

        batch_size, seq_len = x.shape
        if seq_len == 0:
            raise ValueError("Input sequences cannot be empty.")
        embedded = self.embedding(x)

        if h0 is None:
            h = embedded.new_zeros((batch_size, self.hidden_size))
        else:
            if h0.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    "h0 must have shape (batch_size, hidden_size)."
                )

            if (
                h0.device != embedded.device
                or h0.dtype != embedded.dtype
            ):
                raise ValueError(
                    "h0 must match the embedding device and dtype."
                )

            h = h0
        hidden_states = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h = torch.tanh(
                x_t @ self.W_xh
                + h @ self.W_hh
                + self.b_h
            )
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states, dim=1)
        logits = hidden_states @ self.W_hy + self.b_y
        return logits, h

    def fit(self,dataloader,epochs,loss_fn,optimizer):
        epoch_losses=[]
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                prediction,h = self(x_batch)
                pred = prediction.reshape(-1, self.vocab_size)
                y_batch = y_batch.reshape(-1, )
                loss = loss_fn(pred,y_batch)
                loss.backward()
                optimizer.step()
                total_loss +=loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)

            if (epoch +1)%10 ==0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {total_loss:.4f}"
                )
            # if scheduler is not None:
            #     if isinstance(
            #         scheduler,
            #         torch.optim.lr_scheduler.ReduceLROnPlateau
            #     ):
            #         scheduler.step(avg_loss)
            #     else:
            #         scheduler.step()
        return epoch_losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            logits = self(X)
            predictions = torch.argmax(logits,dim=1)
        return predictions


class CustomDataset(Dataset):
    def __init__(self,x,y,tranform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = tranform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if self.transform:
            x = self.transform(x)
        return x, y

class DataGenerator():
    @staticmethod
    def generate_pair(pairs:list[tuple]):
        x = []
        y = []
        for input,output in pairs:
            x.append(input)
            y.append(output)
        return x, y