"""
Test Runner for Advanced RAG MVP
Runs all test suites and generates comprehensive reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
import subprocess

def run_test_suite(test_file):
    """Run a specific test suite"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(test_file))
        
        return {
            "test_file": test_file,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except Exception as e:
        return {
            "test_file": test_file,
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

def generate_comprehensive_report():
    """Generate a comprehensive test report"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    # Collect all test result files
    test_results_dir = "../test-results"
    
    if not os.path.exists(test_results_dir):
        print("No test results directory found!")
        return
    
    # Read all summary files
    summary_files = [f for f in os.listdir(test_results_dir) if f.endswith("_summary.json")]
    
    all_summaries = []
    total_tests = 0
    total_successful = 0
    
    for summary_file in summary_files:
        try:
            with open(os.path.join(test_results_dir, summary_file), 'r') as f:
                summary = json.load(f)
                all_summaries.append(summary)
                total_tests += summary.get("total_tests", 0)
                total_successful += summary.get("successful_tests", 0)
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")
    
    # Generate comprehensive report
    comprehensive_report = {
        "report_title": "Advanced RAG MVP - Comprehensive Test Report",
        "generated_at": datetime.now().isoformat(),
        "overall_statistics": {
            "total_test_suites": len(all_summaries),
            "total_tests": total_tests,
            "total_successful": total_successful,
            "overall_success_rate": (total_successful / total_tests * 100) if total_tests > 0 else 0
        },
        "test_suite_summaries": all_summaries,
        "test_files_generated": []
    }
    
    # List all generated files
    for file in os.listdir(test_results_dir):
        comprehensive_report["test_files_generated"].append(file)
    
    # Save comprehensive report
    with open(os.path.join(test_results_dir, "comprehensive_test_report.json"), 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for summary in all_summaries:
        summary_data.append({
            "test_suite": summary.get("test_suite", "Unknown"),
            "total_tests": summary.get("total_tests", 0),
            "successful_tests": summary.get("successful_tests", 0),
            "success_rate": summary.get("success_rate", 0),
            "timestamp": summary.get("timestamp", "")
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(test_results_dir, "all_tests_summary.csv"), index=False)
    
    return comprehensive_report

def main():
    """Main test runner function"""
    print("ADVANCED RAG MVP - TEST RUNNER")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Ensure test-results directory exists
    os.makedirs("../test-results", exist_ok=True)
    
    # List of test files to run
    test_files = [
        "test_indexing_optimization.py",
        "test_vector_databases.py"
    ]
    
    # Run each test suite
    test_results = []
    for test_file in test_files:
        if os.path.exists(test_file):
            result = run_test_suite(test_file)
            test_results.append(result)
            
            if result["success"]:
                print(f"✅ {test_file} - PASSED")
            else:
                print(f"❌ {test_file} - FAILED")
                if result["stderr"]:
                    print(f"Error: {result['stderr']}")
        else:
            print(f"⚠️  {test_file} - NOT FOUND")
    
    # Generate comprehensive report
    comprehensive_report = generate_comprehensive_report()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    
    if comprehensive_report:
        stats = comprehensive_report["overall_statistics"]
        print(f"Total Test Suites: {stats['total_test_suites']}")
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Successful Tests: {stats['total_successful']}")
        print(f"Overall Success Rate: {stats['overall_success_rate']:.1f}%")
        
        print("\nGenerated Files:")
        for file in comprehensive_report["test_files_generated"]:
            print(f"  - {file}")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    # Return success if all tests passed
    return all(result["success"] for result in test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

