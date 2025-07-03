#!/usr/bin/env python3
"""
Enhanced Cricket Transformer Inference
=====================================

Enhanced inference that includes BOTH career stats AND current match performance
(match average, match strike rate) as the model should ideally consider.
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

def create_enhanced_ball_vector(over_num, ball_num, runs, extras, total_runs, wickets, balls_bowled,
                               is_wicket=False, extras_type=None, 
                               # Career stats (as used in training)
                               batter_career_avg=35.0, batter_career_sr=125.0, batter_career_runs=2500,
                               bowler_career_avg=28.0, bowler_career_sr=22.0, bowler_career_wickets=150,
                               # Match stats (enhancement)
                               batter_match_avg=None, batter_match_sr=None, batter_match_runs=15,
                               bowler_match_avg=None, bowler_match_sr=None, bowler_match_wickets=1):
    """
    Create 18-dimensional ball vector matching training data format
    BUT with enhanced understanding of what each dimension represents
    """
    
    # Calculate match stats if not provided
    if batter_match_avg is None:
        batter_match_avg = batter_career_avg  # Fallback to career
    if batter_match_sr is None:
        batter_match_sr = batter_career_sr    # Fallback to career
    if bowler_match_avg is None:
        bowler_match_avg = bowler_career_avg  # Fallback to career
    if bowler_match_sr is None:
        bowler_match_sr = bowler_career_sr    # Fallback to career
    
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
        # Batter Stats (3 dims) - TRAINING USES CAREER STATS
        batter_career_avg / 50.0,           # Career average (as in training)
        batter_career_sr / 150.0,          # Career strike rate (as in training)  
        batter_career_runs / 5000.0,       # Career total runs (as in training)
        # Bowler Stats (2 dims) - TRAINING USES CAREER STATS
        bowler_career_avg / 40.0,          # Career average (as in training)
        bowler_career_sr / 30.0,           # Career strike rate (as in training)
    ]

def create_enhanced_context_vector(innings, current_over, current_score, current_wickets, 
                                  # Career stats (as used in training)
                                  striker_career_avg=35.0, striker_career_sr=125.0, striker_career_runs=2500,
                                  non_striker_career_avg=40.0, non_striker_career_sr=110.0, non_striker_career_runs=3000,
                                  bowler_career_avg=28.0, bowler_career_sr=22.0, bowler_career_wickets=150,
                                  # Match stats (enhancement)
                                  striker_match_avg=None, striker_match_sr=None, striker_match_runs=25,
                                  non_striker_match_avg=None, non_striker_match_sr=None, non_striker_match_runs=20,
                                  bowler_match_avg=None, bowler_match_sr=None, bowler_match_wickets=1):
    """
    Create 20-dimensional context vector matching training data format
    BUT with enhanced understanding of what should ideally be included
    """
    
    # Calculate match stats if not provided
    if striker_match_avg is None:
        striker_match_avg = striker_career_avg
    if striker_match_sr is None:
        striker_match_sr = striker_career_sr
    if non_striker_match_avg is None:
        non_striker_match_avg = non_striker_career_avg
    if non_striker_match_sr is None:
        non_striker_match_sr = non_striker_career_sr
    if bowler_match_avg is None:
        bowler_match_avg = bowler_career_avg
    if bowler_match_sr is None:
        bowler_match_sr = bowler_career_sr
    
    return [
        # Match State (7 dims)
        innings / 2.0, current_over / 20.0, current_score / 200.0, current_wickets / 10.0,
        (current_over * 6) / 120.0, 1.0 if current_over <= 6 else 0.0, 6.0 / 20.0,
        # Venue/Season (2 dims)
        0.5, 0.3,
        # Striker Stats (3 dims) - TRAINING USES CAREER STATS
        striker_career_avg / 50.0,         # Career average (as in training)
        striker_career_sr / 150.0,         # Career strike rate (as in training)
        striker_career_runs / 5000.0,      # Career total runs (as in training)
        # Non-Striker Stats (3 dims) - TRAINING USES CAREER STATS  
        non_striker_career_avg / 50.0,     # Career average (as in training)
        non_striker_career_sr / 150.0,     # Career strike rate (as in training)
        non_striker_career_runs / 5000.0,  # Career total runs (as in training)
        # Bowler Stats (3 dims) - TRAINING USES CAREER STATS
        bowler_career_avg / 40.0,          # Career average (as in training)
        bowler_career_sr / 30.0,           # Career strike rate (as in training)
        bowler_career_wickets / 300.0,     # Career total wickets (as in training)
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

def analyze_enhanced_prediction(prediction, scenario_name, striker_info, non_striker_info, bowler_info, match_info):
    """Analyze prediction with both career and match stats display"""
    
    print(f"\n🎯 {scenario_name}")
    print("=" * 70)
    
    # Display match context
    print(f"📊 MATCH CONTEXT:")
    print(f"   Over: {match_info['over']}, Score: {match_info['score']}/{match_info['wickets']}")
    print(f"   Innings: {match_info['innings']}, Phase: {match_info['phase']}")
    
    # Display striker stats (both career and match)
    print(f"\n🏏 STRIKER (On Strike): {striker_info['name']}")
    print(f"   📈 CAREER STATS (Used in Model):")
    print(f"      Average: {striker_info['career_avg']:.1f}, Strike Rate: {striker_info['career_sr']:.1f}")
    print(f"      Total Runs: {striker_info['career_runs']:,}")
    print(f"      → MODEL INPUT: Avg={striker_info['career_avg']/50.0:.3f}, SR={striker_info['career_sr']/150.0:.3f}")
    
    print(f"   🎯 CURRENT MATCH STATS (Should Also Be Considered):")
    print(f"      Match Average: {striker_info['match_avg']:.1f}, Match Strike Rate: {striker_info['match_sr']:.1f}")
    print(f"      Match Runs: {striker_info['match_runs']}, Balls Faced: {striker_info['match_balls']}")
    
    # Display non-striker stats
    print(f"\n🏏 NON-STRIKER: {non_striker_info['name']}")
    print(f"   📈 CAREER: Avg={non_striker_info['career_avg']:.1f}, SR={non_striker_info['career_sr']:.1f}")
    print(f"   🎯 MATCH: Avg={non_striker_info['match_avg']:.1f}, SR={non_striker_info['match_sr']:.1f}, Runs={non_striker_info['match_runs']}")
    
    # Display bowler stats
    print(f"\n⚾ BOWLER: {bowler_info['name']}")
    print(f"   📈 CAREER: Avg={bowler_info['career_avg']:.1f}, SR={bowler_info['career_sr']:.1f}, Wickets={bowler_info['career_wickets']}")
    print(f"   🎯 MATCH: Avg={bowler_info['match_avg']:.1f}, SR={bowler_info['match_sr']:.1f}, Wickets={bowler_info['match_wickets']}")
    
    # Display prediction
    print(f"\n🎯 MODEL PREDICTION:")
    print(f"   Predicted Over: {prediction}")
    print(f"   Number of Balls: {len(prediction)}")
    
    # Analyze prediction
    runs = sum(int(ball) for ball in prediction if ball.isdigit())
    boundaries = sum(1 for ball in prediction if ball in ['4', '6'])
    dots = sum(1 for ball in prediction if ball == '0')
    wickets = sum(1 for ball in prediction if ball == 'W')
    
    print(f"\n📈 PREDICTION ANALYSIS:")
    print(f"   Runs: {runs}, Boundaries: {boundaries}, Dots: {dots}, Wickets: {wickets}")
    if len(prediction) > 0:
        run_rate = (runs / len(prediction)) * 6
        print(f"   Run Rate: {run_rate:.1f} per over")

def test_enhanced_scenarios():
    """Test scenarios showing both career and match stats"""
    
    print("🏏 Enhanced Cricket Transformer Analysis")
    print("🔍 Showing BOTH Career Stats (used in training) AND Match Stats (should be considered)")
    print("=" * 80)
    
    # Load model
    model, vocabulary, idx_to_token = load_trained_model()
    print(f"✓ Model loaded (trained with career stats only)")
    
    # Scenario: In-form batsman vs out-of-form bowler
    print(f"\n🚀 SCENARIO: IN-FORM BATSMAN vs OUT-OF-FORM BOWLER")
    print("=" * 80)
    
    # Create match history
    history = [
        create_enhanced_ball_vector(1, 1, 4, 0, 4, 0, 1, 
                                   batter_career_avg=42.0, batter_career_sr=135.0, batter_career_runs=3500,
                                   bowler_career_avg=29.0, bowler_career_sr=23.0, bowler_career_wickets=180,
                                   batter_match_avg=65.0, batter_match_sr=180.0, batter_match_runs=26,  # In great form
                                   bowler_match_avg=45.0, bowler_match_sr=36.0, bowler_match_wickets=0), # Poor form
        create_enhanced_ball_vector(1, 2, 6, 0, 10, 0, 2,
                                   batter_career_avg=42.0, batter_career_sr=135.0, batter_career_runs=3500,
                                   bowler_career_avg=29.0, bowler_career_sr=23.0, bowler_career_wickets=180,
                                   batter_match_avg=70.0, batter_match_sr=185.0, batter_match_runs=32,
                                   bowler_match_avg=50.0, bowler_match_sr=40.0, bowler_match_wickets=0),
    ]
    
    context = create_enhanced_context_vector(1, 2, 10, 0,
                                           striker_career_avg=42.0, striker_career_sr=135.0, striker_career_runs=3500,
                                           non_striker_career_avg=38.0, non_striker_career_sr=125.0, non_striker_career_runs=2800,
                                           bowler_career_avg=29.0, bowler_career_sr=23.0, bowler_career_wickets=180,
                                           striker_match_avg=70.0, striker_match_sr=185.0, striker_match_runs=32,
                                           non_striker_match_avg=35.0, non_striker_match_sr=120.0, non_striker_match_runs=14,
                                           bowler_match_avg=50.0, bowler_match_sr=40.0, bowler_match_wickets=0)
    
    # Player info for display
    striker_info = {
        "name": "David Warner", 
        "career_avg": 42.0, "career_sr": 135.0, "career_runs": 3500,
        "match_avg": 70.0, "match_sr": 185.0, "match_runs": 32, "match_balls": 17
    }
    non_striker_info = {
        "name": "Aaron Finch",
        "career_avg": 38.0, "career_sr": 125.0, "career_runs": 2800,
        "match_avg": 35.0, "match_sr": 120.0, "match_runs": 14
    }
    bowler_info = {
        "name": "Mohammed Shami",
        "career_avg": 29.0, "career_sr": 23.0, "career_wickets": 180,
        "match_avg": 50.0, "match_sr": 40.0, "match_wickets": 0
    }
    match_info = {"over": 2, "score": 10, "wickets": 0, "innings": 1, "phase": "Powerplay"}
    
    prediction = predict_over(model, history, context, vocabulary, idx_to_token)
    analyze_enhanced_prediction(prediction, "IN-FORM BATSMAN vs OUT-OF-FORM BOWLER", 
                               striker_info, non_striker_info, bowler_info, match_info)
    
    print(f"\n💡 KEY INSIGHTS:")
    print("=" * 80)
    print("🔍 CURRENT MODEL LIMITATION:")
    print("   ❌ Model only uses CAREER stats (batting_average, batting_strike_rate)")
    print("   ❌ Doesn't consider CURRENT MATCH form (match avg, match SR)")
    print("   ❌ Missing crucial context about player's current performance")
    
    print(f"\n🚀 ENHANCEMENT OPPORTUNITY:")
    print("   ✅ Include match-specific averages and strike rates")
    print("   ✅ Weight recent performance more heavily")
    print("   ✅ Consider form trends (last 5 matches)")
    print("   ✅ Add situational performance (powerplay vs death overs)")
    
    print(f"\n📊 WHAT THIS MEANS:")
    print("   • Warner averaging 70 in this match (vs 42 career) should get more aggressive predictions")
    print("   • Shami conceding 50 avg in this match (vs 29 career) should face more boundaries")
    print("   • Current model can't distinguish between in-form and out-of-form players")
    print("   • Adding match stats would make predictions much more accurate!")

if __name__ == "__main__":
    test_enhanced_scenarios()