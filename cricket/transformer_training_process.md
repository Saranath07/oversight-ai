# Cricket Transformer: Training and Inference Process

This document provides a mathematical explanation of how the Cricket Transformer model trains and makes predictions, with a focus on the sequence-to-sequence mechanism and how multiple overs are handled.

## Sequence Structure: Match History to Over Prediction

### Input-Output Relationship

The Cricket Transformer uses **128 previous balls** from match history to predict **one complete 6-ball over**:

- **Input**: 128 balls from match history ($\mathbf{X} \in \mathbb{R}^{128 \times 18}$)
- **Output**: 6 balls in an over ($\mathbf{Y} \in \mathbb{N}^6$)

This is a many-to-many sequence transformation, but with different sequence lengths (128 → 6).

## Sequential Over Prediction Flow

For predicting multiple overs in sequence, the model operates with a sliding window approach:

### First Over Prediction
1. Use the 128 most recent balls from match history
2. Predict all 6 balls of the next over
3. Add these 6 predicted balls to the match history

### Second Over Prediction
1. Use the most recent 128 balls, which now includes the 6 balls predicted for the first over
2. Predict all 6 balls of the second over
3. Add these 6 new predicted balls to the match history

### Third Over Prediction
1. Use the most recent 128 balls, which now includes the 12 balls predicted from the first two overs
2. Predict all 6 balls of the third over
3. Continue this pattern

```
                 [128 balls match history] → [6 balls prediction for over 1]
[122 balls from history + 6 predicted balls] → [6 balls prediction for over 2]
[116 balls from history + 12 predicted balls] → [6 balls prediction for over 3]
```

This sliding window approach means that as the model predicts more overs, it begins to rely more on its own predictions as part of the input for future predictions.

## Match Simulation Flow Diagram

```
Initial match history: [b₁, b₂, b₃, ..., b₁₂₇, b₁₂₈]
                                                 ↓
                             Predict over 1: [p₁, p₂, p₃, p₄, p₅, p₆]
                                                 ↓
Updated match history: [b₇, b₈, ..., b₁₂₈, p₁, p₂, p₃, p₄, p₅, p₆]
                                                 ↓
                             Predict over 2: [p₇, p₈, p₉, p₁₀, p₁₁, p₁₂]
                                                 ↓
Updated match history: [b₁₃, ..., b₁₂₈, p₁, p₂, ..., p₁₂]
                                                 ↓
                             Predict over 3: [p₁₃, p₁₄, p₁₅, p₁₆, p₁₇, p₁₈]
```

## Training Data Structure

During training, each example consists of:

1. **Input**: Match history sequence $\mathbf{X} \in \mathbb{R}^{128 \times 18}$ (128 previous balls)
2. **Target**: Over sequence $\mathbf{Y} \in \mathbb{N}^6$ (6 balls to predict)
3. **Context**: Current over context $\mathbf{c} \in \mathbb{R}^{20}$ (match situation)

## Ball-by-Ball vs. Full Over Prediction

The Cricket Transformer handles prediction differently during training versus inference:

### Training: Parallel Prediction with Causal Masking

During training, the model processes the entire over in parallel, but with **causal masking** to preserve the sequential nature:

```
# Input from encoder: 128 encoded ball vectors
M ∈ ℝ^(128×512)

# Decoder input (shifted target sequence with START token)
tgt_input = [<START>, y₁, y₂, y₃, y₄, y₅] ∈ ℕ^6

# Decoder output (logits for each position)
logits ∈ ℝ^(6×24)

# Actual target outputs
tgt_output = [y₁, y₂, y₃, y₄, y₅, <END>] ∈ ℕ^6
```

Even though the computation happens in parallel, each position in the decoder can only attend to previous positions due to the causal mask:

- Position 1 can only see the START token
- Position 2 can see START and ball 1
- Position 3 can see START, ball 1, and ball 2
- And so on...

### Inference: Ball-by-Ball Autoregressive Generation

During inference (prediction time), the model generates one ball at a time:

1. Encode all 128 balls from match history → $\mathbf{M} \in \mathbb{R}^{128 \times 512}$
2. Initialize with START token → $\mathbf{tgt} = [1]$
3. Predict first ball → $\hat{y}_1$
4. Update sequence → $\mathbf{tgt} = [1, \hat{y}_1]$
5. Predict second ball → $\hat{y}_2$
6. Continue until all 6 balls are predicted

At each step, the model uses:
- The complete encoded match history (128 balls)
- All previously predicted balls in the current over

## Error Accumulation in Extended Prediction

When predicting multiple overs in sequence, the model's own predictions are fed back as input, which can lead to error accumulation over time. This is why the model's performance might degrade the further it predicts into the future.

To mitigate this effect, the model uses both:

1. **Teacher Forcing Training**: Learning from ground truth data
2. **Autonomous Training**: Learning to handle its own prediction errors

The autonomous training mode is particularly important for match simulation where the model predicts many consecutive overs.

```python
# Example autonomous training from trainer.py
with torch.no_grad():
    generated = model.generate(
        histories=batch['histories'],
        contexts=batch['contexts'],
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        max_length=batch['target_outputs'].shape[1],
        history_mask=batch['history_mask'],
        temperature=0.8
    )

# Use generated sequence as input for training
target_inputs = generated[:, :target_seq_len]
logits = model(
    histories=batch['histories'],
    contexts=batch['contexts'],
    target_inputs=target_inputs,
    history_mask=batch['history_mask'],
    target_mask=batch['target_mask']
)
```

## Full Match Simulation

For simulating entire cricket matches, the process follows:

1. Start with the initial 128 balls from the actual match
2. Predict the next over (6 balls)
3. Add these predictions to the match history
4. Update match state (score, wickets, etc.)
5. Predict the next over using the updated history
6. Continue until the match concludes

At each step, the most recent 128 balls (which increasingly include the model's own predictions) are used as input for predicting the next over.

## Causal Masking

The key to preventing the decoder from "cheating" by looking at future balls is the causal mask in the decoder's self-attention.

### Self-Attention Causal Mask

The causal mask ensures that when predicting ball $i$, the model can only attend to balls 1 through $i-1$:

$$\mathbf{mask}_{\text{causal}} \in \mathbb{R}^{6 \times 6}$$

$$\mathbf{mask}_{\text{causal}}[i, j] = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}$$

For a 6-ball over, the causal mask looks like:

$$\mathbf{mask}_{\text{causal}} = 
\begin{bmatrix}
0 & -\infty & -\infty & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty & -\infty & -\infty \\
0 & 0 & 0 & -\infty & -\infty & -\infty \\
0 & 0 & 0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}$$

This mask is applied to the self-attention scores:

$$\mathbf{S}_{\text{masked}} = \mathbf{S} + \mathbf{mask}_{\text{causal}}$$

When applied to the attention scores before the softmax, the $-\infty$ values in the mask become zeros in the attention weights, effectively blocking information flow from future positions.

## Summary: Match Simulation Flow

The complete flow for predicting multiple overs:

1. **Initial Input**: Start with 128 balls from match history
2. **First Over**: Predict 6 balls using the 128-ball history
3. **Update History**: Slide the window to include the 6 newly predicted balls
4. **Next Over**: Use the updated 128-ball history (which now includes some predicted balls)
5. **Repeat**: Continue the process, with the history gradually containing more predicted balls

This approach allows the model to simulate match progression while maintaining the appropriate sequential dependencies between balls and overs.