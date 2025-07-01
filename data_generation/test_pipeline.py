"""
OverSight AI Data Generation Pipeline Test
==========================================

Test script to validate the data generation pipeline with a small subset of data.
"""

import sys
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_processor import CricketDataProcessor
from data_loader import CricketDataLoader, create_sample_batch
from data_analyzer import CricketDataAnalyzer

def test_data_processor():
    """Test the data processor with a single match"""
    print("=== TESTING DATA PROCESSOR ===")
    
    try:
        processor = CricketDataProcessor(
            json_dir="data/ipl_json",
            player_stats_file="data/comprehensive_player_stats.csv"
        )
        
        # Test vocabulary creation
        assert len(processor.vocabulary) > 20, "Vocabulary too small"
        assert 'W' in processor.vocabulary, "Wicket token missing"
        assert '4' in processor.vocabulary, "Four runs token missing"
        print("✓ Vocabulary creation successful")
        
        # Test player stats loading
        assert len(processor.player_stats) > 100, "Too few players loaded"
        print("✓ Player stats loading successful")
        
        # Test single match processing
        json_files = list(Path("data/ipl_json").glob("*.json"))
        if json_files:
            test_file = json_files[0]
            sequences = processor.process_match(test_file)
            
            assert len(sequences) > 0, "No sequences generated"
            
            # Validate sequence structure
            match_history, context, target = sequences[0]
            assert len(match_history) > 0, "Empty match history"
            assert len(context) == 19, f"Context vector wrong size: {len(context)}"
            assert len(target) > 0, "Empty target sequence"
            
            print(f"✓ Single match processing successful ({len(sequences)} sequences)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def test_data_loader():
    """Test the data loader functionality"""
    print("\n=== TESTING DATA LOADER ===")
    
    # Check if processed data exists
    processed_dir = Path("data_generation/processed")
    if not processed_dir.exists():
        print("⚠️  Processed data not found, skipping data loader test")
        return True
    
    try:
        # Test data loading
        data_loader = CricketDataLoader(
            data_dir=str(processed_dir),
            batch_size=4,
            max_history_length=100
        )
        
        # Test batch creation
        batch_count = 0
        for batch in data_loader.train_loader:
            # Validate batch structure
            assert 'histories' in batch, "Missing histories in batch"
            assert 'contexts' in batch, "Missing contexts in batch"
            assert 'target_inputs' in batch, "Missing target_inputs in batch"
            assert 'target_outputs' in batch, "Missing target_outputs in batch"
            
            # Validate shapes
            batch_size = batch['histories'].shape[0]
            assert batch_size <= 4, "Batch size too large"
            assert batch['contexts'].shape[0] == batch_size, "Context batch size mismatch"
            
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        print(f"✓ Data loader test successful ({batch_count} batches tested)")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_data_analyzer():
    """Test the data analyzer functionality"""
    print("\n=== TESTING DATA ANALYZER ===")
    
    try:
        analyzer = CricketDataAnalyzer(
            json_dir="data/ipl_json",
            player_stats_file="data/comprehensive_player_stats.csv",
            processed_dir="data_generation/processed"
        )
        
        # Test raw data analysis
        raw_stats = analyzer.analyze_raw_data()
        assert raw_stats['total_matches'] > 0, "No matches analyzed"
        assert raw_stats['total_balls'] > 0, "No balls analyzed"
        print("✓ Raw data analysis successful")
        
        # Test ball outcome analysis
        outcome_counts = analyzer.analyze_ball_outcomes()
        assert len(outcome_counts) > 5, "Too few outcome types"
        print("✓ Ball outcome analysis successful")
        
        # Test player stats analysis
        player_analysis = analyzer.analyze_player_stats()
        assert player_analysis['total_players'] > 100, "Too few players"
        print("✓ Player stats analysis successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data analyzer test failed: {e}")
        return False

def test_vocabulary_consistency():
    """Test vocabulary consistency across components"""
    print("\n=== TESTING VOCABULARY CONSISTENCY ===")
    
    try:
        # Test processor vocabulary
        processor = CricketDataProcessor(
            json_dir="data/ipl_json",
            player_stats_file="data/comprehensive_player_stats.csv"
        )
        
        vocab1 = processor.vocabulary
        
        # Test if processed data exists and has consistent vocabulary
        vocab_file = Path("data_generation/processed/vocabulary.json")
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                vocab2 = json.load(f)
            
            # Check consistency
            assert vocab1.keys() == vocab2.keys(), "Vocabulary keys mismatch"
            assert vocab1 == vocab2, "Vocabulary values mismatch"
            print("✓ Vocabulary consistency check passed")
        else:
            print("⚠️  No processed vocabulary found, skipping consistency check")
        
        return True
        
    except Exception as e:
        print(f"❌ Vocabulary consistency test failed: {e}")
        return False

def test_data_integrity():
    """Test data integrity and format validation"""
    print("\n=== TESTING DATA INTEGRITY ===")
    
    try:
        # Check input data
        json_dir = Path("data/ipl_json")
        player_stats_file = Path("data/comprehensive_player_stats.csv")
        
        # Validate JSON files
        json_files = list(json_dir.glob("*.json"))
        assert len(json_files) > 0, "No JSON files found"
        
        # Test a few JSON files for format
        valid_files = 0
        for i, json_file in enumerate(json_files[:5]):  # Test first 5 files
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Basic structure validation
                assert 'info' in data, f"Missing 'info' in {json_file}"
                assert 'innings' in data, f"Missing 'innings' in {json_file}"
                assert len(data['innings']) > 0, f"No innings in {json_file}"
                
                valid_files += 1
                
            except Exception as e:
                print(f"⚠️  Invalid JSON file {json_file}: {e}")
        
        assert valid_files > 0, "No valid JSON files found"
        print(f"✓ JSON file validation successful ({valid_files}/{min(5, len(json_files))} files valid)")
        
        # Validate player stats CSV
        import pandas as pd
        player_stats = pd.read_csv(player_stats_file)
        
        required_columns = ['player_name', 'total_runs', 'total_wickets', 
                           'batting_average', 'batting_strike_rate']
        for col in required_columns:
            assert col in player_stats.columns, f"Missing column: {col}"
        
        assert len(player_stats) > 100, "Too few players in stats file"
        print("✓ Player stats CSV validation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data integrity test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("=" * 60)
    print("OVERSIGHT AI - COMPREHENSIVE PIPELINE TEST")
    print("=" * 60)
    
    tests = [
        ("Data Integrity", test_data_integrity),
        ("Data Processor", test_data_processor),
        ("Vocabulary Consistency", test_vocabulary_consistency),
        ("Data Loader", test_data_loader),
        ("Data Analyzer", test_data_analyzer),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pipeline is ready!")
        return True
    else:
        print("⚠️  Some tests failed - Please check the issues above")
        return False

def quick_test():
    """Run a quick test with minimal data"""
    print("=== QUICK PIPELINE TEST ===")
    
    try:
        # Test basic imports
        from data_processor import CricketDataProcessor, BallEvent, OverContext
        from data_loader import CricketDataset, CricketDataLoader
        from data_analyzer import CricketDataAnalyzer
        print("✓ All imports successful")
        
        # Test basic functionality
        json_files = list(Path("data/ipl_json").glob("*.json"))
        if len(json_files) > 0:
            print(f"✓ Found {len(json_files)} JSON files")
        else:
            print("⚠️  No JSON files found")
            return False
        
        # Test player stats loading
        import pandas as pd
        stats = pd.read_csv("data/comprehensive_player_stats.csv")
        print(f"✓ Loaded {len(stats)} player records")
        
        print("🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test OverSight AI data pipeline')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
    elif args.comprehensive:
        success = run_comprehensive_test()
    else:
        # Default: run quick test
        success = quick_test()
    
    sys.exit(0 if success else 1)