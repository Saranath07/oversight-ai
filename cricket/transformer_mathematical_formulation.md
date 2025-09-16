# Mathematical Formulation of Cricket Transformer

This document provides a comprehensive mathematical formulation of the Cricket Transformer model, focusing on the sequence processing aspects with precise dimensions.

## Fixed Dimensions

| Parameter | Value |
|-----------|-------|
| Match history sequence length | 128 |
| Target sequence length | 6 |
| Ball vector dimension | 18 |
| Model dimension | 512 |
| Context vector dimension | 20 |
| Vocabulary size | 24 |
| Number of attention heads | 8 |
| Head dimension | 64 (512 ÷ 8) |
| Feed-forward dimension | 2048 |
| Dropout rate | 0.1 |

## Encoder

### Input

Match history sequence: $\mathbf{X} \in \mathbb{R}^{128 \times 18}$

Each row $\mathbf{x}_i \in \mathbb{R}^{18}$ represents features of a single ball in the match history.

### Output

Encoded memory: $\mathbf{M} \in \mathbb{R}^{128 \times 512}$

### Transformation Function

$$f_{\text{encoder}}(\mathbf{X}) = \mathbf{M}$$

Step-by-step transformations:

1. **Input projection**: 
   $$\mathbf{X}_{\text{proj}} = \mathbf{X} \mathbf{W}_{\text{in}} \cdot \sqrt{512} \in \mathbb{R}^{128 \times 512}$$
   
   Where $\mathbf{W}_{\text{in}} \in \mathbb{R}^{18 \times 512}$ and $\sqrt{512} = 22.63$

2. **Add positional encoding**: 
   $$\mathbf{X}_{\text{pos}} = \mathbf{X}_{\text{proj}} + \text{PE}(128, 512) \in \mathbb{R}^{128 \times 512}$$
   
   Where positional encoding $\text{PE}$ for position $i$ and dimension $j$ is:
   $$\text{PE}_{(i, 2j)} = \sin(i/10000^{2j/512})$$
   $$\text{PE}_{(i, 2j+1)} = \cos(i/10000^{2j/512})$$

3. **Apply dropout**: 
   $$\mathbf{X}_{\text{drop}} = \text{Dropout}(\mathbf{X}_{\text{pos}}, p=0.1) \in \mathbb{R}^{128 \times 512}$$

4. **Self-Attention Mechanism** (detailed):
   
   a. **Linear projections** for each attention head $h \in \{1, 2, ..., 8\}$:
   $$\mathbf{Q}_h = \mathbf{X}_{\text{drop}} \mathbf{W}_h^Q \in \mathbb{R}^{128 \times 64}$$
   $$\mathbf{K}_h = \mathbf{X}_{\text{drop}} \mathbf{W}_h^K \in \mathbb{R}^{128 \times 64}$$
   $$\mathbf{V}_h = \mathbf{X}_{\text{drop}} \mathbf{W}_h^V \in \mathbb{R}^{128 \times 64}$$
   
   Where $\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V \in \mathbb{R}^{512 \times 64}$ are learnable parameters.
   
   b. **Attention score matrix** for each head:
   $$\mathbf{S}_h = \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{64}} \in \mathbb{R}^{128 \times 128}$$
   
   c. **Apply softmax** to each row of the score matrix:
   $$\mathbf{A}_h = \text{softmax}(\mathbf{S}_h) \in \mathbb{R}^{128 \times 128}$$
   
   Where softmax is applied row-wise:
   $$\mathbf{A}_h[i, j] = \frac{e^{\mathbf{S}_h[i, j]}}{\sum_{k=1}^{128} e^{\mathbf{S}_h[i, k]}}$$
   
   d. **Compute head output**:
   $$\mathbf{H}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{128 \times 64}$$
   
   e. **Concatenate all heads**:
   $$\mathbf{H}_{\text{concat}} = [\mathbf{H}_1; \mathbf{H}_2; \ldots; \mathbf{H}_8] \in \mathbb{R}^{128 \times 512}$$
   
   Where $[;]$ represents concatenation along the last dimension.
   
   f. **Project concatenated output**:
   $$\mathbf{H}_{\text{proj}} = \mathbf{H}_{\text{concat}} \mathbf{W}^O \in \mathbb{R}^{128 \times 512}$$
   
   Where $\mathbf{W}^O \in \mathbb{R}^{512 \times 512}$ is a learnable parameter matrix.
   
   g. **Add & Norm**:
   $$\mathbf{X}_{\text{attn}} = \text{LayerNorm}(\mathbf{X}_{\text{drop}} + \mathbf{H}_{\text{proj}}) \in \mathbb{R}^{128 \times 512}$$

5. **Feed-Forward Network** (applied to each position independently):
   $$\mathbf{X}_{\text{ff}} = [\text{FFN}(\mathbf{X}_{\text{attn}}[1]); \ldots; \text{FFN}(\mathbf{X}_{\text{attn}}[128])] \in \mathbb{R}^{128 \times 512}$$
   
   Where for each position $i$:
   $$\text{FFN}(\mathbf{x}_i) = \max(0, \mathbf{x}_i\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 \in \mathbb{R}^{512}$$
   
   With $\mathbf{W}_1 \in \mathbb{R}^{512 \times 2048}$, $\mathbf{W}_2 \in \mathbb{R}^{2048 \times 512}$, $\mathbf{b}_1 \in \mathbb{R}^{2048}$, and $\mathbf{b}_2 \in \mathbb{R}^{512}$

6. **Final Layer Norm**:
   $$\mathbf{M} = \text{LayerNorm}(\mathbf{X}_{\text{attn}} + \mathbf{X}_{\text{ff}}) \in \mathbb{R}^{128 \times 512}$$

## Decoder

### Inputs

- Target token indices: $\mathbf{tgt} \in \mathbb{N}^{6}$ (6 integers between 0-23)
- Encoder memory: $\mathbf{M} \in \mathbb{R}^{128 \times 512}$ (128 encoded ball vectors)
- Over context: $\mathbf{c} \in \mathbb{R}^{20}$ (context vector)

### Output

Output logits: $\mathbf{L} \in \mathbb{R}^{6 \times 24}$

### Transformation Function

$$f_{\text{decoder}}(\mathbf{tgt}, \mathbf{M}, \mathbf{c}) = \mathbf{L}$$

Step-by-step transformations:

1. **Token embedding**: 
   $$\mathbf{E} = \text{Embedding}(\mathbf{tgt}) \cdot \sqrt{512} \in \mathbb{R}^{6 \times 512}$$
   
   Where $\text{Embedding} \in \mathbb{R}^{24 \times 512}$ and $\sqrt{512} = 22.63$

2. **Add positional encoding**: 
   $$\mathbf{E}_{\text{pos}} = \mathbf{E} + \text{PE}(6, 512) \in \mathbb{R}^{6 \times 512}$$
   
   Using the same PE function as the encoder

3. **Add context information** (to first position only): 
   $$\mathbf{c}' = \mathbf{c}\mathbf{W}_{\text{ctx}} \in \mathbb{R}^{512}$$
   
   Where $\mathbf{W}_{\text{ctx}} \in \mathbb{R}^{20 \times 512}$
   
   $$\mathbf{E}_{\text{pos}}[0] = \mathbf{E}_{\text{pos}}[0] + \mathbf{c}' \in \mathbb{R}^{512}$$

4. **Apply dropout**: 
   $$\mathbf{E}_{\text{drop}} = \text{Dropout}(\mathbf{E}_{\text{pos}}, p=0.1) \in \mathbb{R}^{6 \times 512}$$

5. **Causal Self-Attention**:
   
   a. **Create causal mask**: 
   $$\mathbf{mask}_{\text{causal}} \in \mathbb{R}^{6 \times 6}$$
   
   Where:
   $$\mathbf{mask}_{\text{causal}}[i, j] = \begin{cases}
   0 & \text{if } i \geq j \\
   -\infty & \text{if } i < j
   \end{cases}$$
   
   b. **Self-attention with causal mask** (similar to encoder self-attention but with causality):
   $$\mathbf{E}_{\text{self}} = \text{SelfAttention}(\mathbf{E}_{\text{drop}}, \mathbf{mask}_{\text{causal}}) \in \mathbb{R}^{6 \times 512}$$

6. **Cross-Attention** (attending to all 128 encoder outputs):
   
   a. **Linear projections** for each head $h \in \{1, 2, ..., 8\}$:
   $$\mathbf{Q}_h = \mathbf{E}_{\text{self}} \mathbf{W}_h^Q \in \mathbb{R}^{6 \times 64}$$
   $$\mathbf{K}_h = \mathbf{M} \mathbf{W}_h^K \in \mathbb{R}^{128 \times 64}$$
   $$\mathbf{V}_h = \mathbf{M} \mathbf{W}_h^V \in \mathbb{R}^{128 \times 64}$$
   
   With projection matrices $\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V \in \mathbb{R}^{512 \times 64}$
   
   b. **Attention score matrix** for each head:
   $$\mathbf{S}_h = \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{64}} \in \mathbb{R}^{6 \times 128}$$
   
   c. **Apply softmax** to each row:
   $$\mathbf{A}_h = \text{softmax}(\mathbf{S}_h) \in \mathbb{R}^{6 \times 128}$$
   
   d. **Compute head output**:
   $$\mathbf{H}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{6 \times 64}$$
   
   e. **Concatenate heads and project**:
   $$\mathbf{H}_{\text{cross}} = [\mathbf{H}_1; \ldots; \mathbf{H}_8]\mathbf{W}^O \in \mathbb{R}^{6 \times 512}$$
   
   f. **Add & Norm**:
   $$\mathbf{E}_{\text{cross}} = \text{LayerNorm}(\mathbf{E}_{\text{self}} + \mathbf{H}_{\text{cross}}) \in \mathbb{R}^{6 \times 512}$$

7. **Feed-Forward Network** (applied to each position):
   $$\mathbf{E}_{\text{ff}} = [\text{FFN}(\mathbf{E}_{\text{cross}}[1]); \ldots; \text{FFN}(\mathbf{E}_{\text{cross}}[6])] \in \mathbb{R}^{6 \times 512}$$
   
   Using the same FFN structure as in the encoder

8. **Final Layer Norm**:
   $$\mathbf{Y} = \text{LayerNorm}(\mathbf{E}_{\text{cross}} + \mathbf{E}_{\text{ff}}) \in \mathbb{R}^{6 \times 512}$$

9. **Project to vocabulary**:
   $$\mathbf{L} = \mathbf{Y}\mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}} \in \mathbb{R}^{6 \times 24}$$
   
   Where $\mathbf{W}_{\text{out}} \in \mathbb{R}^{512 \times 24}$ and $\mathbf{b}_{\text{out}} \in \mathbb{R}^{24}$

## Autoregressive Prediction Process

For generating cricket predictions during inference:

1. **Encode match history**:
   - Input: $\mathbf{X} \in \mathbb{R}^{128 \times 18}$ (128 ball vectors)
   - Output: $\mathbf{M} \in \mathbb{R}^{128 \times 512}$ (encoded memory)

2. **Initialize with start token**:
   - $\mathbf{tgt} = [1] \in \mathbb{N}^1$ (just the START token)
   - Initial logits: $\mathbf{l}_1 = f_{\text{decoder}}(\mathbf{tgt}, \mathbf{M}, \mathbf{c})[-1] \in \mathbb{R}^{24}$
   - First prediction: $\hat{y}_1 = \text{argmax}(\text{softmax}(\mathbf{l}_1))$

3. **Generate second token**:
   - $\mathbf{tgt} = [1, \hat{y}_1] \in \mathbb{N}^2$
   - New logits: $\mathbf{l}_2 = f_{\text{decoder}}(\mathbf{tgt}, \mathbf{M}, \mathbf{c})[-1] \in \mathbb{R}^{24}$
   - Second prediction: $\hat{y}_2 = \text{argmax}(\text{softmax}(\mathbf{l}_2))$

4. **Continue until end token or maximum length**:
   - $\mathbf{tgt} = [1, \hat{y}_1, \hat{y}_2, \ldots] \in \mathbb{N}^t$
   - Final predictions: $\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \ldots] \in \mathbb{N}^{t-1}$

This process demonstrates how the decoder attends to the complete match history (128 encoded vectors) while generating each token prediction sequentially.