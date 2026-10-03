# Approach — Day 37

## The RNN Cell
h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
y_t = h_t @ W_hy + b_y

Three matrices, all shared across every timestep:
- W_xh (embedding_dim, hidden)  — input into hidden space
- W_hh (hidden, hidden)         — the memory connection
- W_hy (hidden, vocab)          — hidden into vocabulary scores

## Shape Flow
x_batch              (32, 6)
    ↓ nn.Embedding(29, 8)
embedded             (32, 6, 8)
    ↓ x_t = embedded[:, t, :]
x_t                  (32, 8)
    ↓ @ W_xh (8, 128) + h @ W_hh (128, 128) + b_h (128,)
h                    (32, 128)
    ↓ stack 6 timesteps at dim=1
hidden_states        (32, 6, 128)
    ↓ @ W_hy (128, 29) + b_y (29,)
logits               (32, 6, 29)
    ↓ reshape(-1, 29)
                     (192, 29)
targets (32, 6) → reshape(-1) → (192,)

192 = 32 × 6. Every timestep of every sequence is an independent
training signal. One forward pass yields 192 of them.

## Batched Matrix Multiplication
(32, 6, 128) @ (128, 29) — leading dims are batch, only the last
two multiply. The same W_hy applies to all 192 hidden states,
the same way a conv filter is shared across image positions.

## Training Data
Sliding window, input and target shifted by one:
"hell" → "ello". Position 0 sees 'h' predicts 'e', position 1
sees 'e' predicts 'l', and so on.

## Initialization
Xavier on W_xh and W_hy. Orthogonal on W_hh specifically —
orthogonal matrices have eigenvalues of magnitude 1, so repeated
multiplication across timesteps neither shrinks nor explodes
the signal.

## Parameters (vocab=29, embed=8, hidden=128)
embedding   29 × 8    =    232
W_xh         8 × 128  =  1,024
W_hh       128 × 128  = 16,384   ← 76% of the model
b_h              128  =    128
W_hy       128 × 29   =  3,712
b_y               29  =     29
                        21,509