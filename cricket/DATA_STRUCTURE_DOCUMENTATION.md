# Cricket Prediction Pipeline - Data Structure Documentation

## 🔄 Complete Data Flow: Raw Input → Processed Features → Tokenized Output

This document shows exactly what the data looks like at each stage of the pipeline, from raw cricket JSON to final model input.

---

## 📊 Stage 1: Raw Input Data

### Raw Cricket Match JSON (Cricsheet Format)
**Source**: [`data/ipl_json/*.json`](data/ipl_json/)

```json
{
  "info": {
    "teams": ["Mumbai Indians", "Chennai Super Kings"],
    "venue": "Wankhede Stadium",
    "season": "2023",
    "match_type": "T20",
    "dates": ["2023-04-01"]
  },
  "innings": [
    {
      "team": "Mumbai Indians",
      "overs": [
        {
          "over": 0,
          "deliveries": [
            {
              "batter": "Rohit Sharma",
              "bowler": "Deepak Chahar", 
              "non_striker": "Ishan Kishan",
              "runs": {
                "batter": 4,
                "extras": 0,
                "total": 4
              }
            },
            {
              "batter": "Rohit Sharma",
              "bowler": "Deepak Chahar",
              "non_striker": "Ishan Kishan", 
              "runs": {
                "batter": 0,
                "extras": 0,
                "total": 0
              }
            },
            {
              "batter": "Rohit Sharma",
              "bowler": "Deepak Chahar",
              "non_striker": "Ishan Kishan",
              "runs": {
                "batter": 1,
                "extras": 0,
                "total": 1
              }
            },
            {
              "batter": "Ishan Kishan",
              "bowler": "Deepak Chahar",
              "non_striker": "Rohit Sharma",
              "runs": {
                "batter": 0,
                "extras": 1,
                "total": 1
              },
              "extras": {
                "wides": 1
              }
            },
            {
              "batter": "Ishan Kishan",
              "bowler": "Deepak Chahar",
              "non_striker": "Rohit Sharma",
              "runs": {
                "batter": 6,
                "extras": 0,
                "total": 6
              }
            },
            {
              "batter": "Ishan Kishan",
              "bowler": "Deepak Chahar",
              "non_striker": "Rohit Sharma",
              "runs": {
                "batter": 0,
                "extras": 0,
                "total": 0
              },
              "wickets": [
                {
                  "kind": "caught",
                  "player_out": "Ishan Kishan",
                  "fielders": ["MS Dhoni"]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### Player Statistics CSV
**Source**: [`data/comprehensive_player_stats.csv`](data/comprehensive_player_stats.csv)

```csv
player_name,total_runs,total_wickets,batting_average,bowling_average,batting_strike_rate,bowling_strike_rate
Rohit Sharma,5611,15,31.17,47.33,130.61,36.0
MS Dhoni,4632,0,38.09,N/A,135.91,N/A
Deepak Chahar,154,69,11.84,27.42,128.33,19.8
Ishan Kishan,1534,0,30.27,N/A,135.69,N/A
```

---

## 📈 Stage 2: Parsed Ball Events

### BallEvent Data Structure
**File**: [`data_processor.py`](data_generation/data_processor.py:24)

Each ball gets parsed into a structured format:

```python
@dataclass
class BallEvent:
    over: int = 0                    # Over number (0-19 for T20)
    ball: int = 0                    # Ball in over (0-5, more for extras)
    batter: str = "Rohit Sharma"     # Batsman name
    bowler: str = "Deepak Chahar"    # Bowler name
    non_striker: str = "Ishan Kishan" # Non-striker name
    runs_batter: int = 4             # Runs scored by batter
    runs_extras: int = 0             # Extra runs (wides, no-balls, etc.)
    runs_total: int = 4              # Total runs on this ball
    extras_type: str = None          # Type: 'wide', 'noball', 'legbye', 'bye'
    wicket_type: str = None          # Type: 'caught', 'bowled', 'lbw', etc.
    wicket_player: str = None        # Player dismissed
    cumulative_score: int = 4        # Total team score after this ball
    cumulative_wickets: int = 0      # Total wickets after this ball
    balls_faced: int = 1             # Balls faced so far
```

**Example Parsed Sequence** (from the JSON above):
```python
# Ball 1: Rohit hits boundary
BallEvent(over=0, ball=0, batter="Rohit Sharma", bowler="Deepak Chahar", 
          runs_batter=4, runs_total=4, cumulative_score=4, cumulative_wickets=0)

# Ball 2: Rohit plays dot ball  
BallEvent(over=0, ball=1, batter="Rohit Sharma", bowler="Deepak Chahar",
          runs_batter=0, runs_total=0, cumulative_score=4, cumulative_wickets=0)

# Ball 3: Rohit takes single
BallEvent(over=0, ball=2, batter="Rohit Sharma", bowler="Deepak Chahar",
          runs_batter=1, runs_total=1, cumulative_score=5, cumulative_wickets=0)

# Ball 4: Wide ball
BallEvent(over=0, ball=3, batter="Ishan Kishan", bowler="Deepak Chahar",
          runs_batter=0, runs_extras=1, runs_total=1, extras_type="wide",
          cumulative_score=6, cumulative_wickets=0)

# Ball 5: Ishan hits six
BallEvent(over=0, ball=4, batter="Ishan Kishan", bowler="Deepak Chahar",
          runs_batter=6, runs_total=6, cumulative_score=12, cumulative_wickets=0)

# Ball 6: Ishan gets out
BallEvent(over=0, ball=5, batter="Ishan Kishan", bowler="Deepak Chahar",
          runs_batter=0, runs_total=0, wicket_type="caught", wicket_player="Ishan Kishan",
          cumulative_score=12, cumulative_wickets=1)
```

---

## 🎯 Stage 3: Feature Engineering

### Ball Vector (18 Dimensions)
**Function**: [`_create_ball_vector()`](data_generation/data_processor.py:202)

Each ball event becomes an 18-dimensional feature vector:

```python
# Example: Ball 1 (Rohit's boundary)
ball_vector = [
    # Ball identification
    0,      # over number
    0,      # ball in over
    
    # Runs information  
    4,      # runs by batter
    0,      # extra runs
    4,      # total runs
    
    # Match state
    0.04,   # cumulative score / 100 (normalized)
    0,      # cumulative wickets
    1,      # balls faced
    
    # Outcome flags
    0,      # is_wicket (0=no, 1=yes)
    0,      # is_wide
    0,      # is_noball  
    0,      # is_legbye
    0,      # is_bye
    
    # Batter stats (from CSV)
    31.17,  # batting average
    130.61, # strike rate
    5.611,  # career runs / 1000 (normalized)
    
    # Bowler stats (from CSV)
    27.42,  # bowling average
    19.8,   # bowling strike rate
    6.9,    # career wickets / 10 (normalized)
    
    # Ball pattern flags
    0,      # is_dot_ball (0=no, 1=yes)
    1,      # is_boundary (0=no, 1=yes)
    0       # is_six (0=no, 1=yes)
]
```

### Context Vector (20 Dimensions)
**Function**: [`_create_context_vector()`](data_generation/data_processor.py:318)

**IMPORTANT**: The context vector contains **CURRENT MATCH STATISTICS** calculated dynamically from the ongoing match state, not just historical player stats.

```python
# Example: Context before Over 1 (after Over 0 completed)
# Current match state: 12/1 after 1 over (12 runs, 1 wicket)
context_vector = [
    # CURRENT MATCH STATE (calculated from balls played so far)
    1,      # innings number (1st or 2nd innings)
    1,      # over number to predict (next over)
    0.12,   # CURRENT SCORE / 100 (12 runs scored so far / 100)
    1,      # CURRENT WICKETS (1 wicket fallen so far)
    11.4,   # BALLS REMAINING / 10 (114 balls left in innings / 10)
    7.2,    # CURRENT RUN RATE (12 runs in 1 over = 12.0, but calculated properly)
    0,      # REQUIRED RATE (0 for first innings, calculated for 2nd innings)
    
    # Venue & season context
    45,     # venue hash % 100 (Wankhede Stadium encoded)
    23,     # season year - 2000 (2023-2000=23)
    
    # CURRENT PLAYERS ON FIELD (who will bat/bowl next over)
    # Current striker stats (batsman on strike for next over)
    28.5,   # striker's career batting average
    142.3,  # striker's career strike rate
    2.1,    # striker's career runs / 1000 (normalized)
    
    # Current non-striker stats (batsman at other end)
    31.17,  # non-striker's career batting average
    130.61, # non-striker's career strike rate
    5.611,  # non-striker's career runs / 1000 (normalized)
    
    # Current bowler stats (who will bowl next over)
    29.8,   # bowler's career bowling average
    21.5,   # bowler's career bowling strike rate
    4.2,    # bowler's career wickets / 10 (normalized)
    
    # Head-to-head placeholders (future enhancement)
    0,      # Striker vs Bowler H2H average
    0       # Striker vs Bowler H2H strike rate
]
```

**Key Point**: The first 7 dimensions are **LIVE MATCH STATISTICS** calculated from the current match situation:
- **Current Score**: Actual runs scored in the match so far
- **Current Wickets**: Actual wickets fallen in the match so far
- **Current Run Rate**: Calculated from balls played and runs scored
- **Required Rate**: For 2nd innings, calculated as (target - current_score) / balls_remaining
- **Balls Remaining**: Actual balls left in the innings

---

## 🏷️ Stage 4: Tokenization

### Target Sequence (Cricket Vocabulary)
**Function**: [`_ball_to_token()`](data_generation/data_processor.py:177)

Each ball outcome becomes a token from the 24-token vocabulary:

**Vocabulary**: [`vocabulary.json`](data_generation/processed/vocabulary.json:1)
```json
{
  "0": 0,   "1": 1,   "2": 2,   "3": 3,   "4": 4,   "5": 5,   "6": 6,
  "W": 7,   "wd": 8,  "nb": 9,  "lb": 10, "b": 11,
  "wd1": 12, "wd2": 13, "wd3": 14, "wd4": 15,
  "nb1": 16, "nb2": 17, "nb3": 18, "nb4": 19, "nb6": 20,
  "<PAD>": 21, "<START>": 22, "<END>": 23
}
```

**Example Token Sequence** (from our over above):
```python
# Raw over: [4, 0, 1, wide, 6, wicket]
target_tokens = ["4", "0", "1", "wd", "6", "W"]

# Tokenized (converted to indices):
target_indices = [4, 0, 1, 8, 6, 7]

# With special tokens for training:
training_sequence = [22, 4, 0, 1, 8, 6, 7, 23]  # <START> + tokens + <END>
```

---

## 📦 Stage 5: Final Training Data Structure

### Complete Training Example
**Output**: [`processed/`](data_generation/processed/) files

Each training example consists of:

```python
training_example = {
    # INPUT: Match history (variable length sequence of ball vectors)
    'match_history': [
        # Ball 1 vector (18 dimensions)
        [0, 0, 4, 0, 4, 0.04, 0, 1, 0, 0, 0, 0, 0, 31.17, 130.61, 5.611, 27.42, 19.8, 6.9, 0, 1, 0],
        # Ball 2 vector (18 dimensions)  
        [0, 1, 0, 0, 0, 0.04, 0, 2, 0, 0, 0, 0, 0, 31.17, 130.61, 5.611, 27.42, 19.8, 6.9, 1, 0, 0],
        # ... more ball vectors (up to 128 balls max)
    ],
    
    # INPUT: Current context (20 dimensions)
    'context': [1, 1, 0.12, 1, 11.4, 7.2, 0, 45, 23, 28.5, 142.3, 2.1, 31.17, 130.61, 5.611, 29.8, 21.5, 4.2, 0, 0],
    
    # TARGET: Next over prediction (token sequence)
    'target_tokens': ["2", "1", "4", "0", "6", "1"],  # What actually happened
    'target_indices': [2, 1, 4, 0, 6, 1]              # Tokenized version
}
```

### Dataset Statistics
**From**: [`metadata.json`](data_generation/processed/metadata.json:1)

```json
{
  "num_sequences": 37397,        // Total training examples
  "ball_vector_dim": 18,         // Ball feature dimensions
  "context_vector_dim": 20,      // Context feature dimensions  
  "vocab_size": 24,              // Output vocabulary size
  "max_sequence_length": 128     // Maximum balls in history
}
```

---

## 🔄 Data Processing Pipeline Summary

```
Raw JSON Ball
     ↓
Parse into BallEvent
     ↓
Extract Features → 18D Ball Vector
     ↓
Accumulate History → Variable Length Sequence
     ↓
Create Context → 20D Context Vector
     ↓
Tokenize Target → Cricket Vocabulary Tokens
     ↓
Training Example: (History, Context, Target)
```

### Example Complete Flow

**Input**: Single ball from JSON
```json
{
  "batter": "Rohit Sharma",
  "bowler": "Deepak Chahar", 
  "runs": {"batter": 4, "total": 4}
}
```

**Step 1**: Parse to BallEvent
```python
BallEvent(batter="Rohit Sharma", bowler="Deepak Chahar", runs_batter=4, runs_total=4)
```

**Step 2**: Create 18D vector
```python
[0, 0, 4, 0, 4, 0.04, 0, 1, 0, 0, 0, 0, 0, 31.17, 130.61, 5.611, 27.42, 19.8, 6.9, 0, 1, 0]
```

**Step 3**: Tokenize outcome
```python
"4" → token_id: 4
```

**Step 4**: Create training example
```python
{
  'history': [ball_vector_1, ball_vector_2, ..., ball_vector_n],
  'context': [context_vector_20d],
  'target': [4, 0, 1, 8, 6, 7]  # Next over tokens
}
```

---

## 🎯 Model Input Format

### Transformer Input Structure

```python
# Encoder Input: Match History
encoder_input = torch.tensor([
    # Batch of variable-length sequences (padded to max_length)
    [[ball_vec_1], [ball_vec_2], ..., [ball_vec_n], [PAD], [PAD], ...],  # Sequence 1
    [[ball_vec_1], [ball_vec_2], ..., [ball_vec_m], [PAD], [PAD], ...],  # Sequence 2
    # ... more sequences in batch
])  # Shape: [batch_size, max_seq_len, 18]

# Context Input: Current Match State  
context_input = torch.tensor([
    [context_vec_1],  # 20D context for sequence 1
    [context_vec_2],  # 20D context for sequence 2
    # ... more contexts
])  # Shape: [batch_size, 20]

# Target Output: Next Over Tokens
target_output = torch.tensor([
    [22, 4, 0, 1, 8, 6, 7, 23],  # <START> + tokens + <END> for sequence 1
    [22, 2, 1, 4, 0, 6, 1, 23],  # <START> + tokens + <END> for sequence 2
    # ... more target sequences
])  # Shape: [batch_size, max_target_len]
```

This is exactly what gets fed into the transformer model for training and inference!

---

## 🏆 Key Insights

1. **Rich Feature Engineering**: Each ball becomes an 18D vector with match state, player stats, and outcome patterns
2. **Context-Aware**: 20D context vector captures current match situation for better predictions
3. **Sequence-to-Sequence**: Variable-length match history → Fixed-length over prediction
4. **Cricket-Specific Vocabulary**: 24 tokens cover all possible cricket outcomes
5. **Scalable Processing**: 37,397 training sequences from real match data

This data structure enables the transformer to learn complex cricket patterns and make accurate ball-by-ball predictions!

---

## 🧠 Stage 6: Model Data Flow

### Transformer Architecture Data Flow
**File**: [`cricket_transformer.py`](training/cricket_transformer.py:215)

Here's exactly how data flows through the neural network:

```
INPUT DATA
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CRICKET TRANSFORMER MODEL                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENCODER BRANCH                    DECODER BRANCH               │
│  ===============                  ===============               │
│                                                                 │
│  Match History                     THREE DECODER INPUTS:        │
│  [B, seq_len, 18]                 1. Target Tokens [B, tgt_len] │
│         ↓                         2. Context Vector [B, 20]     │
│  Input Projection                 3. Encoder Output [B,seq,512] │
│  [B, seq_len, 512]                         ↓                    │
│         ↓                         Token Embedding               │
│  Positional Encoding              [B, tgt_len, 512]             │
│  [B, seq_len, 512]                         ↓                    │
│         ↓                         Positional Encoding           │
│  Multi-Head Self-Attention        [B, tgt_len, 512]             │
│  (6 layers)                                ↓                    │
│         ↓                         Context Injection             │
│  Feed-Forward Networks            [B, tgt_len, 512]             │
│  [B, seq_len, 512]                         ↓                    │
│         ↓                         Masked Self-Attention         │
│  Layer Norm + Residual            [B, tgt_len, 512]             │
│  [B, seq_len, 512]                         ↓                    │
│         ↓                         Cross-Attention               │
│  ENCODER OUTPUT ────────────────→ (uses encoder output)         │
│  [B, seq_len, 512]                [B, tgt_len, 512]             │
│                                            ↓                    │
│                                   Feed-Forward Networks         │
│                                   [B, tgt_len, 512]             │
│                                            ↓                    │
│                                   Output Projection             │
│                                   [B, tgt_len, 24]              │
│                                            ↓                    │
│                                   CRICKET PREDICTIONS           │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Step-by-Step Data Flow

#### **Step 1: Input Processing**

```python
# ENCODER INPUT (1 input)
histories = torch.tensor([
    # Batch of match histories
    [[ball_vec_1], [ball_vec_2], ..., [ball_vec_n]],  # Sequence 1
    [[ball_vec_1], [ball_vec_2], ..., [ball_vec_m]],  # Sequence 2
])  # Shape: [batch_size, seq_len, 18]

# DECODER INPUTS (3 inputs)
# Input 1: Target token sequence
target_inputs = torch.tensor([
    [22, 4, 0, 1, 8, 6],  # <START> + target tokens for sequence 1
    [22, 2, 1, 4, 0, 6],  # <START> + target tokens for sequence 2
])  # Shape: [batch_size, tgt_len]

# Input 2: Current match context
contexts = torch.tensor([
    [context_vec_1],  # 20D context for sequence 1
    [context_vec_2],  # 20D context for sequence 2
])  # Shape: [batch_size, 20]

# Input 3: Encoder output (computed in Step 2)
# encoder_output = [batch_size, seq_len, 512] - comes from encoder
```

#### **Step 2: Encoder Processing**

```python
# ENCODER FORWARD PASS
class CricketTransformerEncoder:
    def forward(self, histories):
        # Step 2a: Project 18D ball vectors to 512D model space
        projected = self.input_projection(histories)
        # Shape: [batch_size, seq_len, 18] → [batch_size, seq_len, 512]
        
        # Step 2b: Add positional encoding
        with_position = self.pos_encoder(projected)
        # Shape: [batch_size, seq_len, 512] (same, but position-aware)
        
        # Step 2c: Multi-head self-attention (6 layers)
        for layer in self.transformer_encoder.layers:
            # Self-attention: each ball attends to all previous balls
            attention_output = layer.self_attn(with_position, with_position, with_position)
            # Feed-forward network
            ff_output = layer.feed_forward(attention_output)
            # Residual connection + layer norm
            with_position = layer.norm(ff_output + attention_output)
        
        encoder_output = with_position
        # Shape: [batch_size, seq_len, 512]
        return encoder_output
```

#### **Step 3: Decoder Processing**

```python
# DECODER FORWARD PASS - Uses ALL THREE INPUTS
class CricketTransformerDecoder:
    def forward(self, target_inputs, encoder_output, contexts):
        # DECODER INPUT 1: Target tokens [batch_size, tgt_len]
        # Step 3a: Token embedding
        token_emb = self.token_embedding(target_inputs)
        # Shape: [batch_size, tgt_len] → [batch_size, tgt_len, 512]
        
        # Step 3b: Add positional encoding
        with_position = self.pos_encoder(token_emb)
        # Shape: [batch_size, tgt_len, 512]
        
        # DECODER INPUT 2: Context vector [batch_size, 20]
        # Step 3c: Context injection (CRITICAL STEP!)
        context_emb = self.context_projection(contexts)  # [B, 20] → [B, 512]
        with_position[:, 0, :] += context_emb  # Add context to first token
        # Shape: [batch_size, tgt_len, 512] (first token now has match context)
        
        # Step 3d: Decoder layers (6 layers)
        for layer in self.transformer_decoder.layers:
            # Masked self-attention (uses target tokens)
            self_attn_output = layer.self_attn(with_position, with_position, with_position, causal_mask)
            
            # DECODER INPUT 3: Encoder output [batch_size, seq_len, 512]
            # Cross-attention to encoder output (attend to match history)
            cross_attn_output = layer.multihead_attn(
                query=self_attn_output,      # From decoder
                key=encoder_output,          # From encoder (match history)
                value=encoder_output         # From encoder (match history)
            )
            
            # Feed-forward network
            ff_output = layer.feed_forward(cross_attn_output)
            
            # Residual connections + layer norms
            with_position = layer.norm(ff_output + cross_attn_output)
        
        # Step 3e: Output projection to vocabulary
        logits = self.output_projection(with_position)
        # Shape: [batch_size, tgt_len, 512] → [batch_size, tgt_len, 24]
        
        return logits

# SUMMARY: Decoder uses THREE inputs:
# 1. target_inputs: What tokens to predict (teacher forcing during training)
# 2. contexts: Current match situation (20D vector)
# 3. encoder_output: Processed match history (from encoder)
```

#### **Step 4: Output Generation**

```python
# Final output logits
logits = model(histories, contexts, target_inputs)
# Shape: [batch_size, tgt_len, 24]

# Convert to probabilities
probabilities = F.softmax(logits, dim=-1)
# Shape: [batch_size, tgt_len, 24]

# Example output for one sequence:
sequence_probs = probabilities[0]  # [tgt_len, 24]
# sequence_probs[0] = probabilities for 1st ball: [0.1, 0.05, 0.02, 0.03, 0.7, ...]
#                                                  ['0', '1',  '2',  '3',  '4', ...]
# sequence_probs[1] = probabilities for 2nd ball: [0.8, 0.1, 0.05, 0.02, 0.03, ...]
# ...

# Predicted tokens (highest probability)
predicted_tokens = torch.argmax(probabilities, dim=-1)
# Shape: [batch_size, tgt_len]
# Example: [[4, 0, 1, 8, 6, 7], [2, 1, 4, 0, 6, 1]]
#          [['4','0','1','wd','6','W'], ['2','1','4','0','6','1']]
```

### Key Data Transformations

#### **Dimension Changes Through the Model**:

```python
# INPUT
histories:     [batch_size, seq_len, 18]     # Raw ball vectors
contexts:      [batch_size, 20]             # Current match context
target_inputs: [batch_size, tgt_len]        # Token indices

# ENCODER
projected:     [batch_size, seq_len, 512]   # After input projection
encoded:       [batch_size, seq_len, 512]   # After transformer layers

# DECODER
token_emb:     [batch_size, tgt_len, 512]   # After token embedding
context_proj:  [batch_size, 512]            # After context projection
decoded:       [batch_size, tgt_len, 512]   # After transformer layers

# OUTPUT
logits:        [batch_size, tgt_len, 24]    # Final predictions
probabilities: [batch_size, tgt_len, 24]    # After softmax
```

#### **Attention Mechanisms**:

1. **Encoder Self-Attention**: Each ball attends to all previous balls in match history
2. **Decoder Self-Attention**: Each predicted token attends to previous predicted tokens (with causal mask)
3. **Cross-Attention**: Each predicted token attends to the entire match history

#### **Context Integration**:

The 20D context vector gets projected to 512D and added to the first decoder token, allowing the model to condition all predictions on the current match state.

### Example Forward Pass

```python
# Real example with actual dimensions
batch_size = 2
seq_len = 50      # 50 balls of match history
tgt_len = 6       # Predicting 6-ball over

# Input
histories = torch.randn(2, 50, 18)    # 2 sequences, 50 balls each, 18D vectors
contexts = torch.randn(2, 20)         # 2 context vectors, 20D each
targets = torch.randint(0, 24, (2, 6)) # 2 target sequences, 6 tokens each

# Forward pass
logits = model(histories, contexts, targets)
# Output: [2, 6, 24] - 2 sequences, 6 predictions, 24 possible tokens each

# Interpretation
# logits[0, 0, :] = probabilities for 1st ball of 1st sequence
# logits[0, 1, :] = probabilities for 2nd ball of 1st sequence
# ...
# logits[1, 5, :] = probabilities for 6th ball of 2nd sequence
```

This architecture allows the model to:
1. **Understand match context** through the encoder processing match history
2. **Incorporate current situation** through context injection in the decoder
3. **Generate sequential predictions** through masked self-attention
4. **Maintain cricket logic** through cross-attention to match history