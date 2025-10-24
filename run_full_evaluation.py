#!/usr/bin/env python3
"""
Easy script to run the full SelfIndex evaluation with proper commands
"""

from optimized_selfindex_evaluator import OptimizedSelfIndexEvaluator

def run_test_evaluation():
    """Run a quick test with 10 configurations and 1000 documents"""
    print("🧪 RUNNING TEST EVALUATION")
    print("="*50)
    
    evaluator = OptimizedSelfIndexEvaluator()
    results = evaluator.run_comprehensive_evaluation(max_configs=10, sample_size=1000)
    print(f"✅ Test evaluation complete with {len(results)} configurations")

def run_medium_evaluation():
    """Run medium evaluation with 50 configurations and 10000 documents"""
    print("🏃 RUNNING MEDIUM EVALUATION")
    print("="*50)
    
    evaluator = OptimizedSelfIndexEvaluator()
    results = evaluator.run_comprehensive_evaluation(max_configs=50, sample_size=10000)
    print(f"✅ Medium evaluation complete with {len(results)} configurations")

def run_full_evaluation():
    """Run FULL evaluation with all 108 configurations and 50000 documents"""
    print("🚀 RUNNING FULL EVALUATION - ALL 108 CONFIGURATIONS")
    print("="*60)
    print("⚠️  This will take 2-3 hours with 50,000 documents!")
    
    evaluator = OptimizedSelfIndexEvaluator()
    results = evaluator.run_comprehensive_evaluation(max_configs=108, sample_size=50000)
    print(f"🎉 FULL evaluation complete with {len(results)} configurations")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "test":
            run_test_evaluation()
        elif mode == "medium":
            run_medium_evaluation() 
        elif mode == "full":
            run_full_evaluation()
        else:
            print("Usage: python run_full_evaluation.py [test|medium|full]")
    else:
        # Default to test
        run_test_evaluation()