# Day 37 — RNN: Character-Level Language Model

## Problem
Feedforward networks take fixed-size input processed all at once.
Natural language is sequential — meaning depends on order, context,
and position. Build an RNN from scratch that processes text one
character at a time while carrying memory forward.

## Core Questions
- Why do feedforward networks fail on variable-length sequences?
- What does the hidden state h actually carry?
- How does one shared W_hh give a network memory across time?
- Why does the model learn spelling but fail at grammar?

## Requirements
- CharTokenizer: char2idx, idx2char, sliding-window sequences
- RNNFromScratch: manual W_xh, W_hh, W_hy as nn.Parameter
- Training loop with reshape for CrossEntropyLoss
- Autoregressive text generation with hidden state carried forward