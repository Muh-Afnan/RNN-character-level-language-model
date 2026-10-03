# Learnings — Day 37

## Why Feedforward Fails on Text
A standard network takes everything at once with fixed input size.
Language is sequential — order, context, and position carry meaning.
The RNN processes one element at a time and carries state forward.

## What h Actually Holds
h is the memory. It is NOT one step back — h_t is computed from
h_{t-1}, which came from h_{t-2}, all the way to h_0. So h carries
a compressed trace of the entire sequence so far.

The real limitation is that this history is heavily compressed,
and it decays. Information from early steps gets squeezed through
many tanh activations and many multiplications by W_hh.

## Broadcasting Rule
Align shapes from the right, pad the shorter on the left with 1s.
Each pair must match or one must be 1.
(32, 128) + (128,)  works — 128 matches 128
(32, 128) + (32,)   fails  — 32 vs 128

## argmax dim is a Position, Not a Size
Passing dim=vocab_size (29) threw "Dimension out of range" —
dim names an axis (0, 1, 2), not how many entries it holds.
dim=-1 is safest: always the last axis regardless of shape.

## Autoregressive Generation
The model's own output becomes its next input. h must be carried
forward between steps — without it, every character is predicted
from a zero hidden state and the model has amnesia between steps.

Training uses all 192 predictions against 192 targets.
Generation uses argmax to select one character. Different purposes.

## The Failure Mode — and Why
Output: "f the was not a little the dodo, oughs in the was n"

Real English words, correct spelling, correct comma and spacing.
Grammar across words falls apart completely.

Cause: W_hh. The same matrix multiplies h at every timestep, so
its effect compounds across the sequence. Values shrink step after
step and early context vanishes — the same vanishing gradient from
Day 29, now across time instead of across layers.

W_hh is also 76% of the model's parameters and the only square
matrix, because memory must map hidden → hidden.

This is exactly the problem LSTM was built to solve.

## Bugs Fixed
- Case mismatch: defined self.w_xh, used self.W_xh — AttributeError
- forward() returns (logits, h); fit() treated it as a tensor
- Reshaped logits but forgot to flatten targets
- argmax on dim=1 (sequence axis, size 1) returned all zeros
- Passed an int where a (1,1) tensor was needed in generate()
- predict() accepted h0 but never forwarded it to self()
- Printed the seed instead of the current input each generation step