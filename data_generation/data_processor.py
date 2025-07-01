"""
OverSight AI Data Processor
===========================

This module processes cricket match data to create training sequences for the 
ball-by-ball prediction transformer model.

The data is structured as a Sequence-to-Sequence problem:
- Input (X): High-fidelity match history + Current over context
- Output (Y): Next over's ball-by-ball sequence
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BallEvent:
    """Represents a single ball event with all relevant information"""
    over: int
    ball: int
    batter: str
    bowler: str
    non_striker: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    extras_type: Optional[str] = None
    wicket_type: Optional[str] = None
    wicket_player: Optional[str] = None
    cumulative_score: int = 0
    cumulative_wickets: int = 0
    balls_faced: int = 0

@dataclass
class OverContext:
    """Rich context for the current over to be predicted"""
    # Match state
    innings: int
    over_number: int
    score: int
    wickets: int
    balls_remaining: int
    run_rate: float
    required_rate: Optional[float]
    
    # Players
    striker: str
    non_striker: str
    bowler: str
    
    # Venue and match info
    venue: str
    season: str
    match_type: str
    
    # Player stats (will be populated from comprehensive_player_stats.csv)
    striker_stats: Dict
    non_striker_stats: Dict
    bowler_stats: Dict

class CricketDataProcessor:
    """Main class for processing cricket data into training sequences"""
    
    def __init__(self, json_dir: str, player_stats_file: str):
        self.json_dir = Path(json_dir)
        self.player_stats_file = Path(player_stats_file)
        self.player_stats = self._load_player_stats()
        self.vocabulary = self._create_vocabulary()
        
    def _load_player_stats(self) -> pd.DataFrame:
        """Load comprehensive player statistics"""
        return pd.read_csv(self.player_stats_file)
    
    def _create_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary for ball outcomes"""
        vocab = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            'W': 7,  # Wicket
            'wd': 8,  # Wide
            'nb': 9,  # No ball
            'lb': 10, # Leg bye
            'b': 11,  # Bye
            'wd1': 12, 'wd2': 13, 'wd3': 14, 'wd4': 15,  # Wide + runs
            'nb1': 16, 'nb2': 17, 'nb3': 18, 'nb4': 19, 'nb6': 20,  # No ball + runs
            '<PAD>': 21, '<START>': 22, '<END>': 23
        }
        return vocab
    
    def _get_player_stats(self, player_name: str) -> Dict:
        """Get comprehensive stats for a player"""
        player_row = self.player_stats[self.player_stats['player_name'] == player_name]
        
        if player_row.empty:
            # Return default stats for unknown players
            return {
                'total_runs': 0, 'total_wickets': 0, 'batting_average': 20.0,
                'bowling_average': 30.0, 'batting_strike_rate': 120.0,
                'bowling_strike_rate': 24.0
            }
        
        stats = player_row.iloc[0].to_dict()
        # Handle NaN values
        for key, value in stats.items():
            if pd.isna(value) or value == 'N/A':
                if 'average' in key:
                    stats[key] = 25.0 if 'batting' in key else 30.0
                elif 'strike_rate' in key:
                    stats[key] = 120.0 if 'batting' in key else 24.0
                else:
                    stats[key] = 0
        
        return stats
    
    def _parse_ball_event(self, delivery: Dict, over_num: int, ball_num: int, 
                         cumulative_score: int, cumulative_wickets: int) -> BallEvent:
        """Parse a single delivery into a BallEvent"""
        
        # Basic info
        batter = delivery['batter']
        bowler = delivery['bowler']
        non_striker = delivery['non_striker']
        runs = delivery['runs']
        
        # Runs breakdown
        runs_batter = runs['batter']
        runs_extras = runs['extras']
        runs_total = runs['total']
        
        # Extras
        extras_type = None
        if 'extras' in delivery:
            extras = delivery['extras']
            if 'wides' in extras:
                extras_type = 'wide'
            elif 'noballs' in extras:
                extras_type = 'noball'
            elif 'legbyes' in extras:
                extras_type = 'legbye'
            elif 'byes' in extras:
                extras_type = 'bye'
        
        # Wickets
        wicket_type = None
        wicket_player = None
        if 'wickets' in delivery:
            wicket = delivery['wickets'][0]
            wicket_type = wicket['kind']
            wicket_player = wicket['player_out']
            cumulative_wickets += 1
        
        cumulative_score += runs_total
        
        return BallEvent(
            over=over_num,
            ball=ball_num,
            batter=batter,
            bowler=bowler,
            non_striker=non_striker,
            runs_batter=runs_batter,
            runs_extras=runs_extras,
            runs_total=runs_total,
            extras_type=extras_type,
            wicket_type=wicket_type,
            wicket_player=wicket_player,
            cumulative_score=cumulative_score,
            cumulative_wickets=cumulative_wickets
        )
    
    def _ball_to_token(self, ball_event: BallEvent) -> str:
        """Convert a ball event to a vocabulary token"""
        
        # Handle wickets
        if ball_event.wicket_type:
            return 'W'
        
        # Handle extras with runs
        if ball_event.extras_type:
            if ball_event.extras_type == 'wide':
                if ball_event.runs_total > 1:
                    return f'wd{ball_event.runs_total}'
                return 'wd'
            elif ball_event.extras_type == 'noball':
                if ball_event.runs_total > 1:
                    return f'nb{ball_event.runs_total}'
                return 'nb'
            elif ball_event.extras_type == 'legbye':
                return 'lb'
            elif ball_event.extras_type == 'bye':
                return 'b'
        
        # Regular runs
        return str(ball_event.runs_batter)
    
    def _create_ball_vector(self, ball_event: BallEvent) -> np.ndarray:
        """Create a comprehensive vector representation of a ball with player stats (no IDs)"""
        
        # Get player stats - representing players purely through statistics
        batter_stats = self._get_player_stats(ball_event.batter)
        bowler_stats = self._get_player_stats(ball_event.bowler)
        
        vector = [
            # Ball identification
            ball_event.over,
            ball_event.ball,
            
            # Runs information
            ball_event.runs_batter,
            ball_event.runs_extras,
            ball_event.runs_total,
            
            # Match state at this ball
            ball_event.cumulative_score / 100,  # Normalized score
            ball_event.cumulative_wickets,
            ball_event.balls_faced,
            
            # Outcome encoding (what happened on this ball)
            1 if ball_event.wicket_type else 0,
            1 if ball_event.extras_type == 'wide' else 0,
            1 if ball_event.extras_type == 'noball' else 0,
            1 if ball_event.extras_type == 'legbye' else 0,
            1 if ball_event.extras_type == 'bye' else 0,
            
            # Batter representation (through stats only - no unique IDs)
            batter_stats['batting_average'],
            batter_stats['batting_strike_rate'],
            batter_stats['total_runs'] / 1000,  # Normalized career runs
            
            # Bowler representation (through stats only - no unique IDs)
            bowler_stats['bowling_average'],
            bowler_stats['bowling_strike_rate'],
            bowler_stats['total_wickets'] / 10,  # Normalized career wickets
            
            # Additional ball outcome patterns
            1 if ball_event.runs_batter == 0 else 0,  # Dot ball
            1 if ball_event.runs_batter == 4 else 0,  # Boundary
            1 if ball_event.runs_batter == 6 else 0,  # Six
        ]
        
        return np.array(vector, dtype=np.float32)
    
    def _create_over_context(self, match_data: Dict, innings_idx: int, over_idx: int) -> OverContext:
        """Create rich context for an over"""
        
        innings = match_data['innings'][innings_idx]
        over_data = innings['overs'][over_idx]
        
        # Calculate current match state
        score = 0
        wickets = 0
        balls_played = 0
        
        # Calculate cumulative stats up to this over
        for prev_over_idx in range(over_idx):
            prev_over = innings['overs'][prev_over_idx]
            for delivery in prev_over['deliveries']:
                score += delivery['runs']['total']
                balls_played += 1
                if 'wickets' in delivery:
                    wickets += len(delivery['wickets'])
        
        # Get first delivery to identify players
        first_delivery = over_data['deliveries'][0]
        striker = first_delivery['batter']
        non_striker = first_delivery['non_striker']
        bowler = first_delivery['bowler']
        
        # Calculate rates
        overs_completed = balls_played / 6.0
        run_rate = score / overs_completed if overs_completed > 0 else 0
        
        # Required rate (for second innings)
        required_rate = None
        if innings_idx == 1 and 'target' in innings:
            target = innings['target']['runs']
            balls_remaining = (20 * 6) - balls_played
            runs_needed = target - score
            required_rate = (runs_needed * 6) / balls_remaining if balls_remaining > 0 else 0
        
        return OverContext(
            innings=innings_idx + 1,
            over_number=over_idx,
            score=score,
            wickets=wickets,
            balls_remaining=(20 * 6) - balls_played,
            run_rate=run_rate,
            required_rate=required_rate,
            striker=striker,
            non_striker=non_striker,
            bowler=bowler,
            venue=match_data['info']['venue'],
            season=match_data['info']['season'],
            match_type=match_data['info']['match_type'],
            striker_stats=self._get_player_stats(striker),
            non_striker_stats=self._get_player_stats(non_striker),
            bowler_stats=self._get_player_stats(bowler)
        )
    
    def _extract_season_year(self, season) -> int:
        """Extract year from season, handling both string and integer formats"""
        if isinstance(season, int):
            return season - 2000  # Direct integer year like 2014
        elif isinstance(season, str):
            if '/' in season:
                return int(season.split('/')[0]) - 2000  # Format like "2007/08"
            else:
                return int(season) - 2000  # String year like "2014"
        else:
            return 14  # Default to 2014 if unknown format
    
    def _create_context_vector(self, context: OverContext) -> np.ndarray:
        """Create a comprehensive context vector for the current over"""
        
        vector = [
            # Match state
            context.innings,
            context.over_number,
            context.score / 100,  # Normalized
            context.wickets,
            context.balls_remaining / 10,  # Normalized
            context.run_rate,
            context.required_rate if context.required_rate else 0,
            
            # Venue encoding (simplified - could be expanded with one-hot)
            hash(context.venue) % 100,  # Simple venue encoding
            
            # Season encoding
            self._extract_season_year(context.season),  # Year normalization
            
            # Striker stats
            context.striker_stats['batting_average'],
            context.striker_stats['batting_strike_rate'],
            context.striker_stats['total_runs'] / 1000,  # Normalized
            
            # Non-striker stats
            context.non_striker_stats['batting_average'],
            context.non_striker_stats['batting_strike_rate'],
            context.non_striker_stats['total_runs'] / 1000,  # Normalized
            
            # Bowler stats
            context.bowler_stats['bowling_average'],
            context.bowler_stats['bowling_strike_rate'],
            context.bowler_stats['total_wickets'] / 10,  # Normalized
            
            # Head-to-head placeholder (could be expanded)
            0,  # Striker vs Bowler H2H average
            0,  # Striker vs Bowler H2H strike rate
        ]
        
        return np.array(vector, dtype=np.float32)
    
    def process_match(self, match_file: Path) -> List[Tuple[List[np.ndarray], np.ndarray, List[str]]]:
        """Process a single match file and extract training sequences"""
        
        with open(match_file, 'r') as f:
            match_data = json.load(f)
        
        sequences = []
        
        # Process each innings
        for innings_idx, innings in enumerate(match_data['innings']):
            if len(innings['overs']) < 2:  # Need at least 2 overs for sequence
                continue
            
            # Process each over (except the last one, as we need next over for target)
            for over_idx in range(len(innings['overs']) - 1):
                try:
                    # Create match history up to current over
                    match_history = []
                    cumulative_score = 0
                    cumulative_wickets = 0
                    
                    # Process all balls up to current over
                    for hist_over_idx in range(over_idx + 1):
                        over_data = innings['overs'][hist_over_idx]
                        for ball_idx, delivery in enumerate(over_data['deliveries']):
                            ball_event = self._parse_ball_event(
                                delivery, hist_over_idx, ball_idx,
                                cumulative_score, cumulative_wickets
                            )
                            ball_vector = self._create_ball_vector(ball_event)
                            match_history.append(ball_vector)
                            
                            cumulative_score = ball_event.cumulative_score
                            cumulative_wickets = ball_event.cumulative_wickets
                    
                    # Create context for the next over
                    next_over_context = self._create_over_context(match_data, innings_idx, over_idx + 1)
                    context_vector = self._create_context_vector(next_over_context)
                    
                    # Create target sequence (next over)
                    next_over = innings['overs'][over_idx + 1]
                    target_tokens = []
                    
                    for ball_idx, delivery in enumerate(next_over['deliveries']):
                        ball_event = self._parse_ball_event(
                            delivery, over_idx + 1, ball_idx,
                            cumulative_score, cumulative_wickets
                        )
                        token = self._ball_to_token(ball_event)
                        target_tokens.append(token)
                        
                        cumulative_score = ball_event.cumulative_score
                        cumulative_wickets = ball_event.cumulative_wickets
                    
                    sequences.append((match_history, context_vector, target_tokens))
                    
                except Exception as e:
                    print(f"Error processing over {over_idx} in {match_file}: {e}")
                    continue
        
        return sequences
    
    def process_all_matches(self) -> Tuple[List, List, List]:
        """Process all match files and create training dataset"""
        
        all_match_histories = []
        all_contexts = []
        all_targets = []
        
        json_files = list(self.json_dir.glob('*.json'))
        print(f"Processing {len(json_files)} match files...")
        
        for i, match_file in enumerate(json_files):
            if i % 10 == 0:
                print(f"Processed {i}/{len(json_files)} files...")
            
            try:
                sequences = self.process_match(match_file)
                for match_history, context, target in sequences:
                    all_match_histories.append(match_history)
                    all_contexts.append(context)
                    all_targets.append(target)
            except Exception as e:
                print(f"Error processing {match_file}: {e}")
                continue
        
        print(f"Generated {len(all_match_histories)} training sequences")
        return all_match_histories, all_contexts, all_targets
    
    def save_processed_data(self, output_dir: str):
        """Process and save all data"""
        
        match_histories, contexts, targets = self.process_all_matches()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save the processed data
        with open(output_path / 'match_histories.pkl', 'wb') as f:
            pickle.dump(match_histories, f)
        
        with open(output_path / 'contexts.pkl', 'wb') as f:
            pickle.dump(contexts, f)
        
        with open(output_path / 'targets.pkl', 'wb') as f:
            pickle.dump(targets, f)
        
        # Save vocabulary
        with open(output_path / 'vocabulary.json', 'w') as f:
            json.dump(self.vocabulary, f, indent=2)
        
        # Save metadata
        metadata = {
            'num_sequences': len(match_histories),
            'ball_vector_dim': len(match_histories[0][0]) if match_histories else 0,
            'context_vector_dim': len(contexts[0]) if contexts else 0,
            'vocab_size': len(self.vocabulary),
            'max_sequence_length': max(len(hist) for hist in match_histories) if match_histories else 0
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved processed data to {output_path}")
        print(f"Dataset statistics:")
        print(f"  - Number of sequences: {metadata['num_sequences']}")
        print(f"  - Ball vector dimension: {metadata['ball_vector_dim']}")
        print(f"  - Context vector dimension: {metadata['context_vector_dim']}")
        print(f"  - Vocabulary size: {metadata['vocab_size']}")
        print(f"  - Max sequence length: {metadata['max_sequence_length']}")

if __name__ == "__main__":
    # Initialize processor
    processor = CricketDataProcessor(
        json_dir="data/ipl_json",
        player_stats_file="data/comprehensive_player_stats.csv"
    )
    
    # Process and save data
    processor.save_processed_data("data_generation/processed")