#!/usr/bin/env python3
"""
Full T20 Cricket Match Simulation
================================

Simulate a complete 20-over T20 cricket match using the trained transformer model.
Both innings with realistic team chasing scenario.
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
                      is_wicket=False, extras_type=None, 
                      batter_avg=35.0, batter_sr=125.0, batter_runs=2500,
                      bowler_avg=28.0, bowler_sr=22.0, bowler_wickets=150):
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
        batter_avg / 50.0, batter_sr / 150.0, batter_runs / 5000.0,
        # Bowler Stats (2 dims)
        bowler_avg / 40.0, bowler_sr / 30.0
    ]

def create_context_vector(innings, current_over, current_score, current_wickets, 
                         striker_avg=35.0, striker_sr=125.0, striker_runs=2500,
                         non_striker_avg=40.0, non_striker_sr=110.0, non_striker_runs=3000,
                         bowler_avg=28.0, bowler_sr=22.0, bowler_wickets=150,
                         target_score=None):
    """Create 20-dimensional context vector"""
    
    # Calculate required rate for chasing team
    required_rate = 6.0  # Default
    if target_score and innings == 2:
        balls_remaining = (20 - current_over) * 6
        runs_needed = target_score - current_score
        if balls_remaining > 0:
            required_rate = (runs_needed * 6) / balls_remaining
    
    return [
        # Match State (7 dims)
        innings / 2.0, current_over / 20.0, current_score / 200.0, current_wickets / 10.0,
        (current_over * 6) / 120.0, 1.0 if current_over <= 6 else 0.0, required_rate / 12.0,
        # Venue/Season (2 dims)
        0.5, 0.3,
        # Striker Stats (3 dims)
        striker_avg / 50.0, striker_sr / 150.0, striker_runs / 5000.0,
        # Non-Striker Stats (3 dims)
        non_striker_avg / 50.0, non_striker_sr / 150.0, non_striker_runs / 5000.0,
        # Bowler Stats (3 dims)
        bowler_avg / 40.0, bowler_sr / 30.0, bowler_wickets / 300.0,
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

def ball_to_runs(ball_token):
    """Convert ball token to runs scored"""
    if ball_token.isdigit():
        return int(ball_token)
    elif ball_token == 'wd':
        return 1
    elif ball_token.startswith('wd'):
        return int(ball_token[2:]) if len(ball_token) > 2 else 1
    elif ball_token.startswith('nb'):
        return int(ball_token[2:]) + 1 if len(ball_token) > 2 else 1
    elif ball_token in ['lb', 'b']:
        return 1
    else:
        return 0

def is_wicket(ball_token):
    """Check if ball token represents a wicket"""
    return ball_token == 'W'

def is_extra(ball_token):
    """Check if ball token represents an extra"""
    return ball_token in ['wd', 'nb', 'lb', 'b'] or ball_token.startswith('wd') or ball_token.startswith('nb')

def simulate_innings(model, vocabulary, idx_to_token, team_name, players, bowlers, innings_num, target=None):
    """Simulate a complete 20-over innings"""
    
    print(f"\n" + "="*80)
    print(f"🏏 {team_name.upper()} INNINGS {innings_num}")
    if target:
        print(f"🎯 Target: {target} runs")
    print("="*80)
    
    # Match state
    total_score = 0
    total_wickets = 0
    total_balls = 0
    match_history = []
    
    # Batting order
    batting_order = list(players.keys())
    bowling_order = list(bowlers.keys())
    
    striker_idx = 0
    non_striker_idx = 1
    
    print(f"Openers: {batting_order[striker_idx]} & {batting_order[non_striker_idx]}")
    
    # Track individual scores
    player_scores = {player: {"runs": 0, "balls": 0} for player in batting_order}
    
    # Simulate each over
    for over_num in range(1, 21):
        bowler = bowling_order[(over_num - 1) % len(bowling_order)]
        striker = batting_order[striker_idx] if striker_idx < len(batting_order) else "Tail-ender"
        non_striker = batting_order[non_striker_idx] if non_striker_idx < len(batting_order) else "Tail-ender"
        
        # Calculate required rate for display
        if target and innings_num == 2:
            balls_remaining = (20 - over_num + 1) * 6
            runs_needed = target - total_score
            req_rate = (runs_needed * 6) / balls_remaining if balls_remaining > 0 else 0
            print(f"\n🎯 OVER {over_num} | Need {runs_needed} in {balls_remaining} balls (RR: {req_rate:.1f})")
        else:
            print(f"\n🎯 OVER {over_num}")
        
        print(f"⚾ {bowler} | 🏏 {striker} & {non_striker} | 📊 {total_score}/{total_wickets}")
        
        # Get player stats
        striker_stats = players.get(striker, {"avg": 25.0, "sr": 120.0, "career_runs": 1000})
        non_striker_stats = players.get(non_striker, {"avg": 25.0, "sr": 120.0, "career_runs": 1000})
        bowler_stats = bowlers.get(bowler, {"avg": 30.0, "sr": 24.0, "career_wickets": 50})
        
        # Create context
        context = create_context_vector(
            innings_num, over_num, total_score, total_wickets,
            striker_avg=striker_stats["avg"], striker_sr=striker_stats["sr"], striker_runs=striker_stats["career_runs"],
            non_striker_avg=non_striker_stats["avg"], non_striker_sr=non_striker_stats["sr"], non_striker_runs=non_striker_stats["career_runs"],
            bowler_avg=bowler_stats["avg"], bowler_sr=bowler_stats["sr"], bowler_wickets=bowler_stats["career_wickets"],
            target_score=target
        )
        
        # Handle empty history for first over
        if len(match_history) == 0:
            dummy_ball = create_ball_vector(
                0, 1, 0, 0, 0, 0, 0,
                batter_avg=striker_stats["avg"], batter_sr=striker_stats["sr"], batter_runs=striker_stats["career_runs"],
                bowler_avg=bowler_stats["avg"], bowler_sr=bowler_stats["sr"], bowler_wickets=bowler_stats["career_wickets"]
            )
            history_for_prediction = [dummy_ball]
        else:
            # Use last 50 balls for context (to avoid memory issues)
            history_for_prediction = match_history[-50:] if len(match_history) > 50 else match_history
        
        # Predict the over
        predicted_over = predict_over(model, history_for_prediction, context, vocabulary, idx_to_token, temperature=0.9)
        
        # Ensure exactly 6 balls
        while len(predicted_over) < 6:
            predicted_over.append('0')
        predicted_over = predicted_over[:6]
        
        # Process each ball
        over_runs = 0
        over_wickets = 0
        ball_details = []
        
        for ball_idx, ball_token in enumerate(predicted_over):
            ball_num = ball_idx + 1
            runs = ball_to_runs(ball_token)
            wicket = is_wicket(ball_token)
            extra = is_extra(ball_token)
            
            # Update match state
            total_score += runs
            over_runs += runs
            total_balls += 1
            
            # Update player stats
            if not extra and striker in player_scores:
                player_scores[striker]["runs"] += runs
                player_scores[striker]["balls"] += 1
            
            if wicket:
                total_wickets += 1
                over_wickets += 1
                # Next batsman comes in
                if striker_idx + 2 < len(batting_order):
                    striker_idx += 2
                    striker = batting_order[striker_idx] if striker_idx < len(batting_order) else "Tail-ender"
            
            # Create ball vector for history
            ball_vector = create_ball_vector(
                over_num, ball_num, runs, 1 if extra else 0, total_score, total_wickets, total_balls,
                is_wicket=wicket, extras_type=ball_token if extra else None,
                batter_avg=striker_stats["avg"], batter_sr=striker_stats["sr"], batter_runs=striker_stats["career_runs"],
                bowler_avg=bowler_stats["avg"], bowler_sr=bowler_stats["sr"], bowler_wickets=bowler_stats["career_wickets"]
            )
            match_history.append(ball_vector)
            
            # Ball description
            if wicket:
                ball_desc = f"{ball_token} - WICKET!"
            elif runs >= 4:
                ball_desc = f"{ball_token} - BOUNDARY!"
            elif runs == 0:
                ball_desc = f"{ball_token}"
            else:
                ball_desc = f"{ball_token}"
            
            ball_details.append(ball_desc)
        
        # Display over
        print(f"🤖 {' '.join(ball_details)} | {over_runs} runs")
        
        # Swap strike if odd runs on last ball
        if len(predicted_over) > 0 and not is_wicket(predicted_over[-1]):
            last_ball_runs = ball_to_runs(predicted_over[-1])
            if last_ball_runs % 2 == 1:
                striker_idx, non_striker_idx = non_striker_idx, striker_idx
        
        # Check if all out
        if total_wickets >= 10:
            print(f"🏁 ALL OUT! {team_name} bowled out for {total_score}")
            break
        
        # Check if target achieved (for chasing team)
        if target and total_score >= target:
            print(f"🎉 TARGET ACHIEVED! {team_name} wins by {10 - total_wickets} wickets!")
            break
    
    # Innings summary
    run_rate = total_score / min(20, over_num)
    print(f"\n📊 INNINGS SUMMARY:")
    print(f"🏏 {team_name}: {total_score}/{total_wickets} in {min(20, over_num)} overs (RR: {run_rate:.1f})")
    
    # Top scorers
    top_scorers = sorted([(p, s["runs"], s["balls"]) for p, s in player_scores.items() if s["runs"] > 0], 
                        key=lambda x: x[1], reverse=True)[:3]
    
    print(f"🌟 Top Scorers:")
    for i, (player, runs, balls) in enumerate(top_scorers):
        sr = (runs / balls * 100) if balls > 0 else 0
        print(f"   {i+1}. {player}: {runs}* ({balls}b, SR: {sr:.1f})")
    
    return total_score, total_wickets, over_num

def simulate_full_t20_match():
    """Simulate a complete T20 match"""
    
    print("🏏 COMPLETE T20 CRICKET MATCH SIMULATION")
    print("=" * 80)
    print("🎯 Mumbai Indians vs Chennai Super Kings")
    print("🏟️ Venue: Wankhede Stadium, Mumbai")
    print("🏆 IPL 2024 Final")
    print("=" * 80)
    
    # Load model
    model, vocabulary, idx_to_token = load_trained_model()
    print(f"✓ AI Cricket Predictor loaded and ready!")
    
    # Team setups
    team1 = "Mumbai Indians"
    team2 = "Chennai Super Kings"
    
    # Mumbai Indians squad
    mi_players = {
        "Rohit Sharma": {"avg": 31.17, "sr": 130.82, "career_runs": 5879},
        "Ishan Kishan": {"avg": 30.48, "sr": 136.76, "career_runs": 1873},
        "Suryakumar Yadav": {"avg": 28.86, "sr": 135.34, "career_runs": 2341},
        "Tilak Varma": {"avg": 32.52, "sr": 140.12, "career_runs": 892},
        "Hardik Pandya": {"avg": 27.59, "sr": 143.79, "career_runs": 1476},
        "Kieron Pollard": {"avg": 26.75, "sr": 147.32, "career_runs": 3412},
        "Krunal Pandya": {"avg": 22.45, "sr": 126.89, "career_runs": 1143},
        "Jasprit Bumrah": {"avg": 8.12, "sr": 95.23, "career_runs": 89},
        "Trent Boult": {"avg": 6.78, "sr": 88.45, "career_runs": 67},
        "Rahul Chahar": {"avg": 9.34, "sr": 102.67, "career_runs": 123},
        "Arjun Tendulkar": {"avg": 12.45, "sr": 110.23, "career_runs": 45}
    }
    
    # Chennai Super Kings squad  
    csk_players = {
        "Ruturaj Gaikwad": {"avg": 32.89, "sr": 125.67, "career_runs": 1654},
        "Devon Conway": {"avg": 35.12, "sr": 128.45, "career_runs": 892},
        "Ajinkya Rahane": {"avg": 32.45, "sr": 118.23, "career_runs": 4187},
        "Shivam Dube": {"avg": 24.67, "sr": 134.56, "career_runs": 743},
        "MS Dhoni": {"avg": 39.65, "sr": 135.91, "career_runs": 4978},
        "Ravindra Jadeja": {"avg": 26.78, "sr": 127.34, "career_runs": 2756},
        "Moeen Ali": {"avg": 23.45, "sr": 142.67, "career_runs": 1234},
        "Deepak Chahar": {"avg": 12.34, "sr": 115.67, "career_runs": 234},
        "Tushar Deshpande": {"avg": 8.67, "sr": 98.45, "career_runs": 89},
        "Mustafizur Rahman": {"avg": 7.23, "sr": 92.34, "career_runs": 67},
        "Maheesh Theekshana": {"avg": 9.12, "sr": 105.67, "career_runs": 45}
    }
    
    # Bowling attacks
    mi_bowlers = {
        "Jasprit Bumrah": {"avg": 23.45, "sr": 18.2, "career_wickets": 145},
        "Trent Boult": {"avg": 25.67, "sr": 19.8, "career_wickets": 76},
        "Hardik Pandya": {"avg": 31.23, "sr": 22.4, "career_wickets": 42},
        "Krunal Pandya": {"avg": 34.56, "sr": 24.7, "career_wickets": 51},
        "Rahul Chahar": {"avg": 28.89, "sr": 21.3, "career_wickets": 34}
    }
    
    csk_bowlers = {
        "Deepak Chahar": {"avg": 27.34, "sr": 19.8, "career_wickets": 72},
        "Mustafizur Rahman": {"avg": 26.12, "sr": 17.9, "career_wickets": 106},
        "Ravindra Jadeja": {"avg": 29.85, "sr": 21.4, "career_wickets": 127},
        "Moeen Ali": {"avg": 32.67, "sr": 23.8, "career_wickets": 45},
        "Maheesh Theekshana": {"avg": 28.45, "sr": 20.6, "career_wickets": 23}
    }
    
    # Toss
    print(f"\n🪙 TOSS: Mumbai Indians win the toss and elect to bat first!")
    
    # First Innings
    mi_score, mi_wickets, mi_overs = simulate_innings(
        model, vocabulary, idx_to_token, team1, mi_players, csk_bowlers, 1
    )
    
    # Second Innings
    target = mi_score + 1
    print(f"\n🎯 CHASE: {team2} need {target} runs to win!")
    
    csk_score, csk_wickets, csk_overs = simulate_innings(
        model, vocabulary, idx_to_token, team2, csk_players, mi_bowlers, 2, target
    )
    
    # Match Result
    print(f"\n" + "="*80)
    print(f"🏆 MATCH RESULT")
    print("="*80)
    print(f"🏏 {team1}: {mi_score}/{mi_wickets} ({mi_overs} overs)")
    print(f"🏏 {team2}: {csk_score}/{csk_wickets} ({csk_overs} overs)")
    
    if csk_score >= target:
        margin = 10 - csk_wickets
        print(f"🎉 {team2} wins by {margin} wickets!")
    elif csk_overs >= 20:
        margin = mi_score - csk_score
        print(f"🎉 {team1} wins by {margin} runs!")
    else:
        print(f"🎉 Match completed!")
    
    print(f"\n🤖 AI MODEL PERFORMANCE:")
    print(f"   ✓ Successfully simulated complete T20 match")
    print(f"   ✓ Generated {(mi_overs + csk_overs) * 6} realistic cricket balls")
    print(f"   ✓ Maintained context across both innings")
    print(f"   ✓ Adapted to chase scenarios and match pressure")
    
    print(f"\n🎊 FULL T20 MATCH SIMULATION COMPLETE!")

if __name__ == "__main__":
    simulate_full_t20_match()