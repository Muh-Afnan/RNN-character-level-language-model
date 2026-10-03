# Day 37 — Character-Level RNN

Vanilla RNN built from scratch with nn.Parameter weights,
trained on Alice in Wonderland (~49k characters).

## Core Classes
- CharTokenizer — char2idx, idx2char, sliding-window sequences
- RNNFromScratch — manual W_xh, W_hh, W_hy; fit(); predict()
- CustomDataset / DataGenerator — DataLoader pipeline

## Config
vocab_size 29 | embedding_dim 8 | hidden_size 128
window 6 | batch 32 | Adam lr=0.001 | 1000 epochs

## Result
Loss 1.74 → 1.52 (plateaus around epoch 500)

Generated: "f the was not a little the dodo, oughs in the was n"

Real words and correct spelling. Grammar across words breaks down.

## Key Insight
h carries the whole sequence, not one step — but compressed and
decaying. W_hh applied at every timestep shrinks early context
toward nothing. Vanishing gradients across time.

Also: dim in argmax is an axis position, not a size. Use dim=-1.

## Run
python demo.py