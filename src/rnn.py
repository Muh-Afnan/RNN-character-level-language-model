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



    
            
