
# OverSight AI: A Deep Learning Cricket Forecaster

## Project Agenda

The goal of OverSight AI is to build a state-of-the-art deep learning model that predicts the ball-by-ball outcome of a cricket over. By leveraging detailed historical data and the current match context, the model aims to generate a probable sequence of events for the upcoming six deliveries.

### Core Idea

The project is framed as a **Sequence-to-Sequence (Seq2Seq)** problem. The model will learn to translate the story of a cricket match *up to a certain point* into the story of the *very next over*.

---

### I. The Input (`X`)

The model takes a complex input composed of two distinct parts for each prediction:

**1. High-Fidelity Match History (Encoder Input):**
   - A **sequence of vectors**, where each vector represents a single ball that has been bowled in the match so far.
   - Each `ball_vector` contains detailed information such as:
     - The players involved (batsman, bowler).
     - The specific outcome (run, wicket, extra).
     - The cumulative score and wickets at that moment.
   - This provides the model with a granular, ball-by-ball narrative of the game's momentum.

**2. Current Over Context (Decoder Context):**
   - A single, **rich feature vector** that captures a complete snapshot of the game state at the beginning of the over to be predicted.
   - This vector is a concatenation of:
     - **Striking Batsman Stats:** Career, seasonal, and current-match performance.
     - **Non-Striking Batsman Stats:** Career, seasonal, and current-match performance.
     - **Bowler Stats:** Career, seasonal, Head-to-Head (H2H), and current-match performance.
     - **Match State:** Innings, venue, score, run rate, required rate, wickets in hand, and balls remaining.

---

### II. The Model Architecture

A **Transformer-based Encoder-Decoder architecture** will be implemented to handle the long-range dependencies in the ball-by-ball match history.

```
[High-Fidelity Match History] ----> [TRANSFORMER ENCODER] ----> [Encoded Match Context]
                                                                        |
                                                                        |
                                         [TRANSFORMER DECODER] <---- [Current Over Context]
                                                 |
                                                 V
                                  [Predicted Ball-by-Ball Sequence]
```

The encoder processes the entire game history, and the decoder uses that summary, along with the detailed current context, to generate the outcome for the next over one ball at a time.

---

### III. The Output (`y`)

The model's output will be a **generated sequence of 6+ ball outcomes** that represents the most likely progression of the over.

- **Format:** A sequence of tokens from a predefined vocabulary.
- **Example Vocabulary:** `['0', '1', '2', '3', '4', '6', 'W', 'wd', 'nb', ...]`
- **Example Output Sequence:** `['1', '0', '4', 'W', '1', '0']`

This sequence provides a rich, interpretable forecast that goes far beyond simple run or wicket predictions.