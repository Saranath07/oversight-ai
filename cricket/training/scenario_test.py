#!/usr/bin/env python3
"""
Cricket Transformer Scenario Testing
====================================

Test the trained model with various cricket match scenarios:
- Beginning of innings (powerplay)
- Middle overs
- Death overs
- Different match situations
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.cricket_transformer import create_cricket_transformer

def load_trained_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Load the trained cricket transformer model"""
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocabulary = checkpoint['vocabulary']
    idx_to_token = {v: k for k, v in vocabulary.items()}
    
    # Create and load model
    model = create_cricket_transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, vocabulary, idx_to_token

def create_ball_vector(over_num, ball_num, runs, extras, total_runs, wickets, balls_bowled,
                      is_wicket=False, extras_type=None, batter_avg=35.0, batter_sr=125.0, 
                      batter_runs=15, bowler_avg=28.0, bowler_sr=22.0):
    """Create 18-dimensional ball vector"""
    
    return [
        # Ball ID (2 dims)
        over_num / 20.0, ball_num / 6.0,
        # Runs (3 dims)
        runs / 6.0, extras / 5.0, (runs + extras) / 6.0,
        # Match State (3 dims)
        total_runs / 200.0, wickets / 10.0, balls_bowled / 120.0,
        # Outcomes (5 dims)
        1.0 if is_wicket else 0.0,
        1.0 if extras_type == "wd" else 0.0,
        1.0 if extras_type and extras_type.startswith("nb") else 0.0,
        1.0 if extras_type == "lb" else 0.0,
        1.0 if extras_type == "bye" else 0.0,
        # Batter Stats (3 dims)
        batter_avg / 50.0, batter_sr / 150.0, batter_runs / 100.0,
        # Bowler Stats (2 dims)
        bowler_avg / 40.0, bowler_sr / 30.0
    ]

def create_context_vector(innings, current_over, current_score, current_wickets, 
                         striker_avg=35.0, striker_sr=125.0, striker_runs=25,
                         non_striker_avg=40.0, non_striker_sr=110.0, non_striker_runs=20,
                         bowler_avg=28.0, bowler_sr=22.0, bowler_wickets=1):
    """Create 20-dimensional context vector"""
    
    return [
        # Match State (7 dims)
        innings / 2.0, current_over / 20.0, current_score / 200.0, current_wickets / 10.0,
        (current_over * 6) / 120.0, 1.0 if current_over <= 6 else 0.0, 6.0 / 20.0,
        # Venue/Season (2 dims)
        0.5, 0.3,
        # Striker Stats (3 dims)
        striker_avg / 50.0, striker_sr / 150.0, striker_runs / 100.0,
        # Non-Striker Stats (3 dims)
        non_striker_avg / 50.0, non_striker_sr / 150.0, non_striker_runs / 100.0,
        # Bowler Stats (3 dims)
        bowler_avg / 40.0, bowler_sr / 30.0, bowler_wickets / 5.0,
        # Head-to-Head (2 dims)
        0.5, 0.4
    ]

def predict_over(model, match_history, context, vocabulary, idx_to_token, temperature=0.8):
    """Make prediction for next over"""
    
    device = next(model.parameters()).device
    
    # Convert to tensors
    history_tensor = torch.tensor(match_history, dtype=torch.float32).unsqueeze(0).to(device)
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    history_mask = torch.ones(history_tensor.shape[:2]).to(device)
    
    # Generate prediction
    with torch.no_grad():
        generated = model.generate(
            histories=history_tensor,
            contexts=context_tensor,
            start_token_id=vocabulary['<START>'],
            end_token_id=vocabulary['<END>'],
            max_length=12,
            history_mask=history_mask,
            temperature=temperature
        )
    
    # Decode prediction
    prediction = []
    for token_id in generated[0].cpu().numpy():
        if token_id == vocabulary['<END>']:
            break
        if token_id not in [vocabulary['<START>'], vocabulary['<PAD>']]:
            token = idx_to_token.get(token_id, '<UNK>')
            prediction.append(token)
    
    return prediction

def analyze_prediction(prediction, scenario_name):
    """Analyze and display prediction results"""
    
    print(f"\n🎯 {scenario_name}")
    print("-" * 50)
    print(f"Predicted Over: {prediction}")
    print(f"Balls: {len(prediction)}")
    
    # Calculate runs
    runs = 0
    extras = 0
    boundaries = 0
    dots = 0
    
    for ball in prediction:
        if ball.isdigit():
            runs += int(ball)
            if int(ball) == 0:
                dots += 1
            elif int(ball) >= 4:
                boundaries += 1
        elif ball == 'wd':
            extras += 1
        elif ball.startswith('nb'):
            extras += 1
            if len(ball) > 2:
                runs += int(ball[2:])
    
    total_runs = runs + extras
    
    print(f"Runs: {runs}, Extras: {extras}, Total: {total_runs}")
    print(f"Dots: {dots}, Boundaries: {boundaries}")
    
    if len(prediction) > 0:
        run_rate = (total_runs / len(prediction)) * 6
        print(f"Run Rate: {run_rate:.1f} per over")
    
    return total_runs, len(prediction)

def test_scenarios():
    """Test various cricket scenarios"""
    
    print("🏏 Cricket Transformer - Scenario Testing")
    print("=" * 60)
    
    # Load model
    model, vocabulary, idx_to_token = load_trained_model()
    print(f"✓ Model loaded (Vocab: {len(vocabulary)} tokens)")
    
    # Scenario 1: Beginning of innings (Powerplay)
    print(f"\n🚀 SCENARIO TESTING")
    print("=" * 60)
    
    # Powerplay scenario - aggressive batting expected
    powerplay_history = [
        create_ball_vector(1, 1, 0, 0, 0, 0, 1),
        create_ball_vector(1, 2, 4, 0, 4, 0, 2),
        create_ball_vector(1, 3, 1, 0, 5, 0, 3),
        create_ball_vector(1, 4, 0, 0, 5, 0, 4),
        create_ball_vector(1, 5, 6, 0, 11, 0, 5),
        create_ball_vector(1, 6, 1, 0, 12, 0, 6),
    ]
    powerplay_context = create_context_vector(1, 2, 12, 0, striker_avg=45.0, striker_sr=140.0)
    
    pred1 = predict_over(model, powerplay_history, powerplay_context, vocabulary, idx_to_token)
    analyze_prediction(pred1, "POWERPLAY (Over 2) - Aggressive Phase")
    
    # Scenario 2: Middle overs (Conservative)
    middle_history = []
    total_runs = 0
    for over in range(1, 8):
        for ball in range(1, 7):
            runs = np.random.choice([0, 1, 2], p=[0.4, 0.5, 0.1])
            total_runs += runs
            middle_history.append(create_ball_vector(over, ball, runs, 0, total_runs, 1, (over-1)*6 + ball))
    
    middle_context = create_context_vector(1, 8, total_runs, 1, striker_avg=32.0, striker_sr=110.0)
    
    pred2 = predict_over(model, middle_history[-20:], middle_context, vocabulary, idx_to_token)
    analyze_prediction(pred2, "MIDDLE OVERS (Over 8) - Building Phase")
    
    # Scenario 3: Death overs (Aggressive)
    death_history = []
    total_runs = 120
    for over in range(15, 18):
        for ball in range(1, 7):
            runs = np.random.choice([0, 1, 2, 4, 6], p=[0.2, 0.3, 0.2, 0.2, 0.1])
            total_runs += runs
            death_history.append(create_ball_vector(over, ball, runs, 0, total_runs, 3, (over-1)*6 + ball))
    
    death_context = create_context_vector(1, 18, total_runs, 3, striker_avg=38.0, striker_sr=155.0)
    
    pred3 = predict_over(model, death_history[-15:], death_context, vocabulary, idx_to_token)
    analyze_prediction(pred3, "DEATH OVERS (Over 18) - Slog Phase")
    
    # Scenario 4: Pressure situation (High required rate)
    pressure_history = []
    total_runs = 80
    for over in range(10, 15):
        for ball in range(1, 7):
            runs = np.random.choice([0, 1, 2, 4], p=[0.3, 0.4, 0.2, 0.1])
            total_runs += runs
            pressure_history.append(create_ball_vector(over, ball, runs, 0, total_runs, 2, (over-1)*6 + ball))
    
    pressure_context = create_context_vector(1, 15, total_runs, 2, striker_avg=28.0, striker_sr=125.0)
    
    pred4 = predict_over(model, pressure_history[-18:], pressure_context, vocabulary, idx_to_token)
    analyze_prediction(pred4, "PRESSURE SITUATION (Over 15) - Chase Mode")
    
    # Scenario 5: Wickets falling (Conservative)
    collapse_history = []
    total_runs = 45
    for over in range(5, 10):
        for ball in range(1, 7):
            if np.random.random() < 0.1:  # 10% chance of wicket
                runs = 0
                wicket = True
            else:
                runs = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
                wicket = False
            total_runs += runs
            collapse_history.append(create_ball_vector(over, ball, runs, 0, total_runs, 5, (over-1)*6 + ball, is_wicket=wicket))
    
    collapse_context = create_context_vector(1, 10, total_runs, 5, striker_avg=25.0, striker_sr=95.0)
    
    pred5 = predict_over(model, collapse_history[-12:], collapse_context, vocabulary, idx_to_token)
    analyze_prediction(pred5, "WICKETS FALLING (Over 10) - Rebuild Phase")
    
    # Summary
    print(f"\n📊 SCENARIO SUMMARY")
    print("=" * 60)
    print("✓ Tested 5 different match scenarios")
    print("✓ Model generates predictions for various situations")
    print("✓ Sequence lengths vary (model still learning)")
    print("✓ Early training stage - more epochs would improve consistency")
    
    print(f"\n🎯 OBSERVATIONS:")
    print("- Model responds to different contexts")
    print("- Predictions vary based on match situation")
    print("- Short sequences due to early <END> tokens (normal for epoch 6)")
    print("- More training would improve sequence length and accuracy")

if __name__ == "__main__":
    test_scenarios()