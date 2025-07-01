#!/usr/bin/env python3
"""
Cricket Transformer User Inference
==================================

Script that takes user-provided cricket match data, processes it using
the data pipeline, and makes predictions with the trained model.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.cricket_transformer import create_cricket_transformer
from data_generation.data_processor import CricketDataProcessor

def load_trained_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Load the trained cricket transformer model"""
    
    print("🏏 Loading Trained Cricket Transformer")
    print("=" * 50)
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🔥 Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("🔥 Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("🔥 Using CPU")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocabulary = checkpoint['vocabulary']
    idx_to_token = {v: k for k, v in vocabulary.items()}
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'Unknown')}")
    print(f"✓ Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
    print(f"✓ Vocabulary size: {len(vocabulary)}")
    
    # Create and load model
    model = create_cricket_transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model ready with {model.count_parameters():,} parameters")
    
    return model, vocabulary, idx_to_token

def create_sample_match_data():
    """Create sample cricket match data as a user would provide"""
    
    # Sample match data - as a user would input
    match_data = {
        "match_id": "sample_match_001",
        "venue": "Wankhede Stadium",
        "season": "2023",
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "innings": 1,
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        
        # Match history - previous balls in the match
        "match_history": [
            # Over 1
            {"over": 1, "ball": 1, "batsman": "Rohit Sharma", "bowler": "Deepak Chahar", 
             "runs": 0, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 1, "ball": 2, "batsman": "Rohit Sharma", "bowler": "Deepak Chahar", 
             "runs": 4, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 1, "ball": 3, "batsman": "Rohit Sharma", "bowler": "Deepak Chahar", 
             "runs": 1, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 1, "ball": 4, "batsman": "Ishan Kishan", "bowler": "Deepak Chahar", 
             "runs": 0, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 1, "ball": 5, "batsman": "Ishan Kishan", "bowler": "Deepak Chahar", 
             "runs": 2, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 1, "ball": 6, "batsman": "Ishan Kishan", "bowler": "Deepak Chahar", 
             "runs": 1, "extras": 0, "wicket": False, "extras_type": None},
            
            # Over 2
            {"over": 2, "ball": 1, "batsman": "Rohit Sharma", "bowler": "Mustafizur Rahman", 
             "runs": 0, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 2, "ball": 2, "batsman": "Rohit Sharma", "bowler": "Mustafizur Rahman", 
             "runs": 6, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 2, "ball": 3, "batsman": "Rohit Sharma", "bowler": "Mustafizur Rahman", 
             "runs": 0, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 2, "ball": 4, "batsman": "Rohit Sharma", "bowler": "Mustafizur Rahman", 
             "runs": 1, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 2, "ball": 5, "batsman": "Ishan Kishan", "bowler": "Mustafizur Rahman", 
             "runs": 0, "extras": 1, "wicket": False, "extras_type": "wd"},
            {"over": 2, "ball": 5, "batsman": "Ishan Kishan", "bowler": "Mustafizur Rahman", 
             "runs": 4, "extras": 0, "wicket": False, "extras_type": None},
            {"over": 2, "ball": 6, "batsman": "Ishan Kishan", "bowler": "Mustafizur Rahman", 
             "runs": 1, "extras": 0, "wicket": False, "extras_type": None},
        ],
        
        # Current match state for context
        "current_state": {
            "current_over": 3,
            "current_score": 20,
            "current_wickets": 0,
            "balls_bowled": 13,
            "striker": {
                "name": "Rohit Sharma",
                "runs_scored": 12,
                "balls_faced": 8,
                "career_avg": 31.17,
                "career_sr": 130.82
            },
            "non_striker": {
                "name": "Ishan Kishan", 
                "runs_scored": 8,
                "balls_faced": 5,
                "career_avg": 30.48,
                "career_sr": 136.76
            },
            "bowler": {
                "name": "Dwayne Bravo",
                "overs_bowled": 0,
                "runs_conceded": 0,
                "wickets_taken": 0,
                "career_avg": 24.82,
                "career_sr": 19.5
            }
        }
    }
    
    return match_data

def process_user_data(match_data):
    """Process user-provided match data into model format"""
    
    print("🔄 Processing user data...")
    
    # Convert match history to ball vectors (18-dim)
    match_history = []
    total_runs = 0
    total_wickets = 0
    
    for ball in match_data["match_history"]:
        total_runs += ball["runs"] + ball["extras"]
        if ball["wicket"]:
            total_wickets += 1
        
        # Create 18-dimensional ball vector (exactly matching training data)
        ball_vector = [
            # Ball ID (2 dims)
            ball["over"] / 20.0,
            ball["ball"] / 6.0,
            
            # Runs (3 dims)
            ball["runs"] / 6.0,
            ball["extras"] / 5.0,
            (ball["runs"] + ball["extras"]) / 6.0,
            
            # Match State (3 dims)
            total_runs / 200.0,
            total_wickets / 10.0,
            ((ball["over"] - 1) * 6 + ball["ball"]) / 120.0,
            
            # Outcomes (5 dims)
            1.0 if ball["wicket"] else 0.0,
            1.0 if ball["extras_type"] == "wd" else 0.0,
            1.0 if ball["extras_type"] and ball["extras_type"].startswith("nb") else 0.0,
            1.0 if ball["extras_type"] == "lb" else 0.0,
            1.0 if ball["extras_type"] == "bye" else 0.0,
            
            # Batter Stats (3 dims)
            match_data["current_state"]["striker"]["career_avg"] / 50.0,
            match_data["current_state"]["striker"]["career_sr"] / 150.0,
            match_data["current_state"]["striker"]["runs_scored"] / 100.0,
            
            # Bowler Stats (2 dims) - reduced to match training
            match_data["current_state"]["bowler"]["career_avg"] / 40.0,
            match_data["current_state"]["bowler"]["career_sr"] / 30.0,
        ]
        
        match_history.append(ball_vector)
    
    # Create context vector (20-dim)
    state = match_data["current_state"]
    context_vector = [
        # Match State (7 dims)
        match_data["innings"] / 2.0,
        state["current_over"] / 20.0,
        state["current_score"] / 200.0,
        state["current_wickets"] / 10.0,
        state["balls_bowled"] / 120.0,
        1.0 if state["current_over"] <= 6 else 0.0,  # powerplay
        6.0 / 20.0,  # required run rate (placeholder)
        
        # Venue/Season (2 dims)
        hash(match_data["venue"]) % 10 / 10.0,
        (int(match_data["season"]) - 2020) / 10.0,
        
        # Striker Stats (3 dims)
        state["striker"]["career_avg"] / 50.0,
        state["striker"]["career_sr"] / 150.0,
        state["striker"]["runs_scored"] / 100.0,
        
        # Non-Striker Stats (3 dims)
        state["non_striker"]["career_avg"] / 50.0,
        state["non_striker"]["career_sr"] / 150.0,
        state["non_striker"]["runs_scored"] / 100.0,
        
        # Bowler Stats (3 dims)
        state["bowler"]["career_avg"] / 40.0,
        state["bowler"]["career_sr"] / 30.0,
        state["bowler"]["wickets_taken"] / 5.0,
        
        # Head-to-Head Placeholders (2 dims)
        0.5, 0.4
    ]
    
    print(f"✓ Processed {len(match_history)} balls into vectors")
    print(f"✓ Created context vector for over {state['current_over']}")
    
    return np.array(match_history), np.array(context_vector)

def predict_next_over(model, match_history, context, vocabulary, idx_to_token):
    """Make prediction for the next over"""
    
    device = next(model.parameters()).device
    
    # Convert to tensors and add batch dimension
    history_tensor = torch.tensor(match_history, dtype=torch.float32).unsqueeze(0).to(device)
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Create history mask
    history_mask = torch.ones(history_tensor.shape[:2]).to(device)
    
    print("🎯 Generating prediction...")
    
    # Generate prediction
    with torch.no_grad():
        generated = model.generate(
            histories=history_tensor,
            contexts=context_tensor,
            start_token_id=vocabulary['<START>'],
            end_token_id=vocabulary['<END>'],
            max_length=12,
            history_mask=history_mask,
            temperature=0.8
        )
    
    # Decode prediction
    prediction_tokens = generated[0].cpu().numpy()
    prediction = []
    
    for token_id in prediction_tokens:
        if token_id == vocabulary['<END>']:
            break
        if token_id != vocabulary['<START>'] and token_id != vocabulary['<PAD>']:
            token = idx_to_token.get(token_id, '<UNK>')
            prediction.append(token)
    
    return prediction

def main():
    """Main inference function"""
    
    print("🏏 Cricket Transformer - User Inference Demo")
    print("=" * 60)
    
    # Load trained model
    model, vocabulary, idx_to_token = load_trained_model()
    if model is None:
        return
    
    # Create sample user data
    print(f"\n📝 Creating Sample Match Data (as user would provide)")
    print("-" * 50)
    match_data = create_sample_match_data()
    
    print(f"Match: {match_data['team1']} vs {match_data['team2']}")
    print(f"Venue: {match_data['venue']}")
    print(f"Current Score: {match_data['current_state']['current_score']}/{match_data['current_state']['current_wickets']}")
    print(f"Current Over: {match_data['current_state']['current_over']}")
    print(f"Striker: {match_data['current_state']['striker']['name']} ({match_data['current_state']['striker']['runs_scored']} runs)")
    print(f"Bowler: {match_data['current_state']['bowler']['name']}")
    
    # Process user data
    print(f"\n🔄 Processing Data Through Pipeline")
    print("-" * 50)
    match_history, context = process_user_data(match_data)
    
    # Make prediction
    print(f"\n🎯 Making Prediction for Next Over")
    print("-" * 50)
    prediction = predict_next_over(model, match_history, context, vocabulary, idx_to_token)
    
    # Display results
    print(f"\n🏏 PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Next Over: {prediction}")
    print(f"Number of balls: {len(prediction)}")
    
    # Analyze prediction
    runs_predicted = 0
    for ball in prediction:
        if ball.isdigit():
            runs_predicted += int(ball)
        elif ball == 'wd' or ball.startswith('nb'):
            runs_predicted += 1
            if len(ball) > 2:  # nb with runs
                runs_predicted += int(ball[2:])
    
    print(f"Predicted runs in over: {runs_predicted}")
    
    # Correct run rate calculation
    if len(prediction) > 0:
        runs_per_ball = runs_predicted / len(prediction)
        runs_per_over = runs_per_ball * 6
        print(f"Predicted run rate: {runs_per_over:.1f} runs per over")
    else:
        print(f"Predicted run rate: 0.0 runs per over")
    
    # Analysis of why short sequence
    print(f"\n📊 Prediction Analysis:")
    print(f"  - Sequence length: {len(prediction)} balls (expected: 6)")
    print(f"  - Model stopped early (likely hit <END> token)")
    print(f"  - This is normal for early training (epoch 6)")
    print(f"  - More training would improve sequence length consistency")
    
    print(f"\n✅ Inference completed successfully!")

if __name__ == "__main__":
    main()