#!/usr/bin/env python3
"""
Convert comprehensive SelfIndex evaluation results from JSON to CSV format.

This script takes the detailed JSON results from comprehensive SelfIndex evaluation
and converts them to a clean, readable CSV format suitable for analysis and reporting.

CSV COLUMNS INCLUDED:
===================

Configuration Details:
- Config_ID: Unique configuration identifier (1-72)
- Config_Name: Full configuration name
- Index_Type: BOOLEAN, WORDCOUNT, or TFIDF
- Datastore: CUSTOM or DB1
- Compression: NONE, CODE, or CLIB
- Query_Processing: TERMatat or DOCatat
- Optimization: Null or Skipping

Performance Metrics:
- Index_Time_Seconds: Time taken to build the index

Latency Metrics (all in milliseconds):
- Latency_Mean_ms: Average query latency
- Latency_P50_ms: 50th percentile (median)
- Latency_P95_ms: 95th percentile
- Latency_P99_ms: 99th percentile
- Latency_Min_ms: Minimum latency
- Latency_Max_ms: Maximum latency

Throughput Metrics:
- Throughput_Single_Thread_QPS: Queries per second (single thread)
- Throughput_Multi_Thread_QPS: Queries per second (multi-threaded)
- Throughput_Speedup_Factor: Multi-thread speedup vs single-thread

Memory Metrics:
- Memory_Index_Size_MB: Size of the index on disk
- Memory_Docs_Per_MB: Documents indexed per MB of memory
- Memory_Growth_MB: Memory growth during indexing

Functional Metrics:
- Functional_MAP: Mean Average Precision
- Functional_Recall: Mean Recall
- Functional_F1_Score: Mean F1 Score
- Functional_NDCG: Mean NDCG
- Functional_Coverage_Rate: Query coverage rate

Query Category Latency Breakdown:
- Latency_[Category]_Mean_ms: Mean latency for specific query types
- Latency_[Category]_P95_ms: 95th percentile for specific query types

USAGE:
    python jsonToCSV.py [input_json_file] [output_csv_file]

    If no arguments provided, uses default files:
    - Input: comprehensive_results_20251103_165428.json
    - Output: comprehensive_selfindex_results.csv

EXAMPLE OUTPUT:
    The CSV contains 72 rows (one per configuration) with 41 columns
    of comprehensive performance metrics for easy analysis in Excel,
    Google Sheets, or any data analysis tool.
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, Any

def load_json_results(json_file_path: str) -> Dict[str, Any]:
    """Load the comprehensive results JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

def extract_metrics_for_csv(results: Dict[str, Any]) -> list:
    """Extract and flatten metrics for CSV output."""
    rows = []

    for config_name, config_data in results.items():
        row = {}

        # Configuration details
        config = config_data.get('config', {})
        row['Config_ID'] = config.get('config_id', '')
        row['Config_Name'] = config.get('name', config_name)
        row['Index_Type'] = config.get('index_type', '')
        row['Datastore'] = config.get('datastore', '')
        row['Compression'] = config.get('compression', '')
        row['Query_Processing'] = config.get('query_proc', '')
        row['Optimization'] = config.get('optimization', '')

        # Index time
        row['Index_Time_Seconds'] = round(config_data.get('index_time', 0), 2)

        # Latency metrics
        latency = config_data.get('metrics', {}).get('latency', {})
        row['Latency_Mean_ms'] = round(latency.get('mean', 0), 2)
        row['Latency_P50_ms'] = round(latency.get('p50', 0), 2)
        row['Latency_P95_ms'] = round(latency.get('p95', 0), 2)
        row['Latency_P99_ms'] = round(latency.get('p99', 0), 2)
        row['Latency_Min_ms'] = round(latency.get('min', 0), 2)
        row['Latency_Max_ms'] = round(latency.get('max', 0), 2)

        # Throughput metrics
        throughput = config_data.get('metrics', {}).get('throughput', {})
        row['Throughput_Single_Thread_QPS'] = round(throughput.get('single_thread_qps', 0), 2)
        row['Throughput_Multi_Thread_QPS'] = round(throughput.get('multi_thread_qps', 0), 2)
        row['Throughput_Speedup_Factor'] = round(throughput.get('speedup_factor', 0), 3)

        # Memory metrics
        memory = config_data.get('metrics', {}).get('memory', {})
        row['Memory_Index_Size_MB'] = round(memory.get('index_size_mb', 0), 2)
        row['Memory_Docs_Per_MB'] = round(memory.get('memory_efficiency_docs_per_mb', 0), 2)
        row['Memory_Growth_MB'] = round(memory.get('memory_growth_mb', 0), 2)

        # Functional metrics
        functional = config_data.get('metrics', {}).get('functional', {})
        row['Functional_MAP'] = round(functional.get('mean_average_precision', 0), 4)
        row['Functional_Recall'] = round(functional.get('mean_recall', 0), 4)
        row['Functional_F1_Score'] = round(functional.get('mean_f1_score', 0), 4)
        row['Functional_NDCG'] = round(functional.get('mean_ndcg', 0), 4)
        row['Functional_Coverage_Rate'] = round(functional.get('coverage_rate', 0), 4)

        # Query category latency breakdown (optional - can be commented out if too many columns)
        category_breakdown = latency.get('category_breakdown', {})
        for category, cat_metrics in category_breakdown.items():
            row[f'Latency_{category.replace("_", "_").title()}_Mean_ms'] = round(cat_metrics.get('mean', 0), 2)
            row[f'Latency_{category.replace("_", "_").title()}_P95_ms'] = round(cat_metrics.get('p95', 0), 2)

        rows.append(row)

    return rows

def write_csv(rows: list, output_path: str):
    """Write the extracted data to CSV file."""
    if not rows:
        print("Error: No data to write to CSV")
        return

    # Get all possible column names
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())

    # Sort columns for consistent ordering
    column_order = [
        'Config_ID', 'Config_Name', 'Index_Type', 'Datastore', 'Compression',
        'Query_Processing', 'Optimization', 'Index_Time_Seconds',
        'Latency_Mean_ms', 'Latency_P50_ms', 'Latency_P95_ms', 'Latency_P99_ms',
        'Latency_Min_ms', 'Latency_Max_ms',
        'Throughput_Single_Thread_QPS', 'Throughput_Multi_Thread_QPS', 'Throughput_Speedup_Factor',
        'Memory_Index_Size_MB', 'Memory_Docs_Per_MB', 'Memory_Growth_MB',
        'Functional_MAP', 'Functional_Recall', 'Functional_F1_Score', 'Functional_NDCG', 'Functional_Coverage_Rate'
    ]

    # Add any remaining columns (like category breakdowns)
    remaining_columns = sorted(all_columns - set(column_order))
    column_order.extend(remaining_columns)

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_order)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ CSV file created successfully: {output_path}")
        print(f"📊 Total configurations: {len(rows)}")
        print(f"📋 Total columns: {len(column_order)}")

    except Exception as e:
        print(f"Error writing CSV file: {e}")
        sys.exit(1)

def main():
    """Main function to convert JSON to CSV."""
    print("🔄 Converting SelfIndex comprehensive results JSON to CSV...")

    # Default paths
    json_file = "comprehensive_results_20251103_165428.json"
    csv_file = "comprehensive_selfindex_results.csv"

    # Check if custom paths provided via command line
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        csv_file = sys.argv[2]

    # Load JSON data
    print(f"📁 Loading JSON file: {json_file}")
    results = load_json_results(json_file)

    # Extract metrics for CSV
    print("📊 Extracting metrics for CSV format...")
    rows = extract_metrics_for_csv(results)

    # Sort rows by config ID for better readability
    rows.sort(key=lambda x: x.get('Config_ID', 999))

    # Write CSV
    print(f"💾 Writing CSV file: {csv_file}")
    write_csv(rows, csv_file)

    print("\n🎉 Conversion complete!")
    print(f"📈 Ready for analysis in: {csv_file}")

if __name__ == "__main__":
    main()
