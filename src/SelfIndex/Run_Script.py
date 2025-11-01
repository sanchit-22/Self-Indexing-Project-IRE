#!/usr/bin/env python3
"""
Easy-to-run script for SelfIndex evaluation with different modes
"""

from optimized_selfindex_evaluator import OptimizedSelfIndexEvaluator

def run_quick_test():
    """Quick test: 10 configs, 1000 docs (~10 minutes)"""
    print("🧪 RUNNING QUICK TEST")
    print("="*40)
    
    evaluator = OptimizedSelfIndexEvaluator()
    results = evaluator.run_comprehensive_evaluation(max_configs=72, sample_size=100)
    print(f"✅ Quick test complete: {len(results)} configurations tested")

def run_medium_test():
    """Medium test: 50 configs, 10000 docs (~1 hour)"""
    print("🏃 RUNNING MEDIUM TEST")
    print("="*40)
    
    evaluator = OptimizedSelfIndexEvaluator()
    results = evaluator.run_comprehensive_evaluation(max_configs=50, sample_size=10000)
    print(f"✅ Medium test complete: {len(results)} configurations tested")

def run_full_evaluation():
    """Full evaluation: ALL 108 configs, 50000 docs (~3-4 hours)"""
    print("🚀 RUNNING FULL EVALUATION - ALL 108 CONFIGURATIONS")
    print("="*60)
    print("⚠️  This will take 3-4 hours with 50,000 documents!")
    print("   Data will be loaded ONCE and reused for all 108 configurations")
    
    evaluator = OptimizedSelfIndexEvaluator()
    results = evaluator.run_comprehensive_evaluation(max_configs=72, sample_size=50000)
    print(f"🎉 FULL evaluation complete: {len(results)} configurations tested")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "quick":
            run_quick_test()
        elif mode == "medium":
            run_medium_test() 
        elif mode == "full":
            run_full_evaluation()
        else:
            print("Usage: python run_selfindex_evaluation.py [quick|medium|full]")
            print("")
            print("Options:")
            print("  quick  - 72 configs, 200 docs (~10 min)")
            print("  medium - 50 configs, 10000 docs (~1 hour)")
            print("  full   - 108 configs, 50000 docs (~3-4 hours)")
    else:
        # Default to quick test
        run_quick_test()