#!/usr/bin/env python3
"""
Complete evaluation script - Run both ESIndex and SelfIndex evaluations
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 COMPLETE INDEXING AND RETRIEVAL EVALUATION")
    print("="*80)
    
    # Check if all files exist
    required_files = [
        'index_base.py',
        'self_index.py', 
        'selfindex_evaluator.py'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return
    
    print("✅ All required files found")
    
    # Import and run evaluations
    try:
        from selfindex_evaluator import SelfIndexEvaluator
        
        print("\n📊 Running SelfIndex Evaluation...")
        evaluator = SelfIndexEvaluator()
        results = evaluator.run_evaluation(max_configs=108, sample_size=50000)
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"   📁 Results directory: selfindex_results/")
        print(f"   📊 Plots generated: Plot.A, Plot.AB, Plot.AC, Plot.C")
        print(f"   📋 Report saved: comprehensive_evaluation_report.json")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()