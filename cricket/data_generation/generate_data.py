"""
OverSight AI Data Generation Pipeline
====================================

Main script to orchestrate the complete data generation process for the 
cricket ball-by-ball prediction transformer model.
"""

import argparse
import sys
from pathlib import Path
import time

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_processor import CricketDataProcessor
from data_loader import CricketDataLoader, print_sample_data
from data_analyzer import CricketDataAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Generate training data for OverSight AI')
    parser.add_argument('--json_dir', default='data/ipl_json', 
                       help='Directory containing match JSON files')
    parser.add_argument('--player_stats', default='data/comprehensive_player_stats.csv',
                       help='Path to player statistics CSV file')
    parser.add_argument('--output_dir', default='data_generation/processed',
                       help='Output directory for processed data')
    parser.add_argument('--analysis_dir', default='data_generation/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--skip_processing', action='store_true',
                       help='Skip data processing (use existing processed data)')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip data analysis')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loader testing')
    parser.add_argument('--max_history_length', type=int, default=500,
                       help='Maximum length of match history sequences')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OVERSIGHT AI - DATA GENERATION PIPELINE")
    print("=" * 60)
    print(f"JSON Directory: {args.json_dir}")
    print(f"Player Stats: {args.player_stats}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Analysis Directory: {args.analysis_dir}")
    print("=" * 60)
    
    # Validate input paths
    json_dir = Path(args.json_dir)
    player_stats_file = Path(args.player_stats)
    
    if not json_dir.exists():
        print(f"❌ Error: JSON directory {json_dir} does not exist!")
        return 1
    
    if not player_stats_file.exists():
        print(f"❌ Error: Player stats file {player_stats_file} does not exist!")
        return 1
    
    json_files = list(json_dir.glob('*.json'))
    if not json_files:
        print(f"❌ Error: No JSON files found in {json_dir}!")
        return 1
    
    print(f"✓ Found {len(json_files)} match files")
    print(f"✓ Player stats file exists")
    
    # Step 1: Data Processing
    if not args.skip_processing:
        print("\n" + "=" * 60)
        print("STEP 1: DATA PROCESSING")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            processor = CricketDataProcessor(
                json_dir=str(json_dir),
                player_stats_file=str(player_stats_file)
            )
            
            processor.save_processed_data(args.output_dir)
            
            processing_time = time.time() - start_time
            print(f"✓ Data processing completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error during data processing: {e}")
            return 1
    else:
        print("\n⏭️  Skipping data processing (using existing data)")
        
        # Verify processed data exists
        output_path = Path(args.output_dir)
        required_files = ['match_histories.pkl', 'contexts.pkl', 'targets.pkl', 
                         'vocabulary.json', 'metadata.json']
        
        for file in required_files:
            if not (output_path / file).exists():
                print(f"❌ Error: Required file {file} not found in {output_path}")
                return 1
        
        print("✓ All required processed files found")
    
    # Step 2: Data Loader Testing
    print("\n" + "=" * 60)
    print("STEP 2: DATA LOADER TESTING")
    print("=" * 60)
    
    try:
        # Test data loading
        print("Testing data loader...")
        data_loader = CricketDataLoader(
            data_dir=args.output_dir,
            batch_size=args.batch_size,
            max_history_length=args.max_history_length
        )
        
        # Test a batch
        print("\nTesting batch creation...")
        for i, batch in enumerate(data_loader.train_loader):
            print(f"✓ Batch {i+1} shape validation:")
            print(f"  - Histories: {batch['histories'].shape}")
            print(f"  - Contexts: {batch['contexts'].shape}")
            print(f"  - Target inputs: {batch['target_inputs'].shape}")
            print(f"  - Target outputs: {batch['target_outputs'].shape}")
            
            if i >= 2:  # Test first 3 batches
                break
        
        print("✓ Data loader testing completed successfully")
        
        # Print sample data
        print("\nSample data inspection:")
        print_sample_data(args.output_dir, num_samples=2)
        
    except Exception as e:
        print(f"❌ Error during data loader testing: {e}")
        return 1
    
    # Step 3: Data Analysis
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("STEP 3: DATA ANALYSIS")
        print("=" * 60)
        
        try:
            analyzer = CricketDataAnalyzer(
                json_dir=str(json_dir),
                player_stats_file=str(player_stats_file),
                processed_dir=args.output_dir
            )
            
            # Create analysis directory
            Path(args.analysis_dir).mkdir(exist_ok=True)
            
            # Generate comprehensive report
            report = analyzer.generate_comprehensive_report()
            
            print("✓ Data analysis completed successfully")
            
        except Exception as e:
            print(f"❌ Error during data analysis: {e}")
            print("⚠️  Continuing without analysis...")
    else:
        print("\n⏭️  Skipping data analysis")
    
    # Step 4: Final Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETION SUMMARY")
    print("=" * 60)
    
    output_path = Path(args.output_dir)
    
    # Check output files
    files_created = []
    file_sizes = {}
    
    expected_files = [
        'match_histories.pkl',
        'contexts.pkl', 
        'targets.pkl',
        'vocabulary.json',
        'metadata.json'
    ]
    
    for file in expected_files:
        file_path = output_path / file
        if file_path.exists():
            files_created.append(file)
            file_sizes[file] = file_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"✓ Output directory: {output_path}")
    print(f"✓ Files created: {len(files_created)}/{len(expected_files)}")
    
    for file in files_created:
        print(f"  - {file}: {file_sizes[file]:.2f} MB")
    
    # Load and display final metadata
    try:
        import json
        with open(output_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"  - Training sequences: {metadata['num_sequences']:,}")
        print(f"  - Ball vector dimension: {metadata['ball_vector_dim']}")
        print(f"  - Context vector dimension: {metadata['context_vector_dim']}")
        print(f"  - Vocabulary size: {metadata['vocab_size']}")
        print(f"  - Max sequence length: {metadata['max_sequence_length']}")
        
    except Exception as e:
        print(f"⚠️  Could not load metadata: {e}")
    
    print(f"\n🎉 DATA GENERATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📁 Processed data available at: {output_path}")
    
    if not args.skip_analysis:
        print(f"📊 Analysis results available at: {args.analysis_dir}")
    
    print(f"\n🚀 Ready for transformer model training!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)