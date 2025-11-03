# Self-Indexing Project - Information Retrieval (IRE)

## 📋 Project Overview

This is a comprehensive Information Retrieval assignment implementing **72 different configurations** of a custom-built search indexing system. The project evaluates how different design choices (index types, compression, storage backends, query processing strategies, and optimizations) affect **latency (Plot A), throughput (Plot B), and memory footprint (Plot C)**.

### 🎯 Key Objectives

- **Build a Production-Ready Index System** supporting multiple index types and optimizations
- **Comprehensive Evaluation** of 72 index configurations with formal performance metrics
- **Formal Academic Report** with detailed analysis, visualizations (Plots A, B, C), and production recommendations
- **Performance Analysis** measuring latency percentiles (P50, P95, P99), throughput (QPS), and memory usage

---

## 📁 Repository Structure

```
Self-Indexing-Project-IRE/
├── README.md                              # This file - Complete project guide
├── comprehensive_enhanced_report.pdf      # FINAL REPORT (38 pages)
├── comprehensive_enhanced_report.tex      # LaTeX source for the report
├── PLOT_BC_COMPLIANCE_SUMMARY.md          # Detailed Plot B/C implementation summary
│
├── src/                                   # Main source code directory
│   ├── requirements.txt                   # Python dependencies
│   ├── Dataset/                           # Data storage
│   │   ├── PreProcessedData/
│   │   │   └── preprocessed_dataset.csv   # 50K Wikipedia documents (processed)
│   │   └── Plots/                         # Frequency distribution plots
│   │
│   ├── SelfIndex/                         # ⭐ MAIN PROJECT CODE
│   │   ├── index_base.py                  # Abstract base class for all index types
│   │   ├── self_index.py                  # SelfIndex implementation (1247 lines)
│   │   ├── optimized_selfindex_evaluator.py # Evaluation framework (992 lines)
│   │   ├── Run_Script.py                  # Easy-to-run evaluation script
│   │   ├── manual_test_index.py           # Manual testing utilities
│   │   ├── indexes/                       # Stored index files (generated)
│   │   └── comprehensive_selfindex_results/  # Evaluation results (generated)
│   │
│   ├── ElasticSearchIndex/                # Alternative: Elasticsearch implementation
│   │   ├── ElasticSearch.ipynb            # Jupyter notebook for ES setup
│   │   └── install_docker.sh              # Docker installation script
│   │
│   ├── local_wikipedia_data/              # Wikipedia dataset cache
│   │   └── wikimedia___wikipedia/20231101.en/ # Raw data files
│   │
│   └── DumpFiles/                         # Reference/legacy files
│       ├── CodebaseExplanation.md         # Detailed architecture documentation
│       ├── IMPLEMENTATION_SUMMARY.md      # Implementation notes
│       ├── complete_evaluation.py         # Reference evaluation script
│       ├── compression.py                 # Compression utilities
│       ├── db_backends.py                 # Database backend implementations
│       ├── query_parser.py                # Query parsing utilities
│       ├── test_self_index.py             # Unit tests
│       └── *.ipynb                        # Reference Jupyter notebooks
│
├── report_plots/                          # Generated visualization plots
│   ├── plot_a_*.pdf                       # Latency plots (P50, P95, P99)
│   ├── plot_b_*.pdf                       # Throughput plots (QPS)
│   └── plot_c_*.pdf                       # Memory footprint plots (MB)
│
└── myenv/                                 # Python virtual environment
```

---

## 📦 Download Indexes (Google Drive)

All pre-built index files for the 72 configurations are available for download:

**[Download Indexes from Google Drive](https://drive.google.com/drive/folders/1XC0iZD9QzgRVhwnAUEHEryZcoE0LxL0s?usp=sharing)**

- Place downloaded files in `src/SelfIndex/indexes/`
- This allows you to run queries and evaluations without rebuilding indexes from scratch

---

## 📦 Download Preprocessed Dataset (Google Drive)

The preprocessed Wikipedia dataset (50K documents) is available for download:

**[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1ZL6LU-0D8rhT4SeVSgLvpH6Vq2_Pg4zH?usp=sharing)**

- Place downloaded files in `src/Dataset/PreProcessedData/`
- This allows you to skip the data preprocessing step and directly run evaluations

---

## 🔧 Setup & Installation

### Prerequisites

- **Python 3.8+** (3.12 recommended)
- **pip** or **conda** for package management
- **Git** (for version control)
- **~5 GB disk space** (for preprocessed dataset + indexes)

### Step 1: Clone the Repository

```bash
cd /path/to/project
git clone <repository-url>
cd Self-Indexing-Project-IRE
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv myenv
source myenv/bin/activate

# OR using conda
conda create -n self-index python=3.12
conda activate self-index
```

### Step 3: Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

**Required Packages:**

- `datasets==4.1.1` - Wikipedia dataset loading
- `nltk==3.9.1` - Natural language processing (tokenization, stemming)
- `matplotlib==3.10.6` - Visualization
- `scipy==1.16.2` - Scientific computing
- `seaborn==0.13.2` - Statistical visualization
- `psutil==7.1.1` - System memory/performance monitoring
- `numpy==2.3.1` - Numerical arrays
- `pandas==2.3.3` - Data manipulation
- `elasticsearch==8.11.0` - Elasticsearch client (optional)

### Step 4: Download NLTK Data (One-time Setup)

```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 🚀 How to Run the Evaluation

### Quick Reference: The 72 Index Configurations

The system tests all combinations of:

| Dimension                | Variable | Values                                     | Count  |
| ------------------------ | -------- | ------------------------------------------ | ------ |
| **Index Type**           | x        | Boolean (1), WordCount (2), TF-IDF (3)     | 3      |
| **Storage**              | y        | Custom (1), DB1 SQLite (2)                 | 2      |
| **Compression**          | z        | None (1), Dictionary (2), zlib (3)         | 3      |
| **Query Processing**     | q        | Document-at-a-time (D), Term-at-a-time (T) | 2      |
| **Skip Pointers**        | i        | Disabled (0), Enabled (1)                  | 2      |
| **Total Configurations** | -        | 3 × 2 × 3 × 2 × 2 =                        | **72** |

### Running the Evaluation

#### 🟢 **Quick Test** (~10 minutes)

```bash
cd src/SelfIndex
python3 Run_Script.py quick
```

- Tests 3 configurations
- Uses 100 sample documents
- Good for verifying setup works

#### 🟡 **Medium Test** (~1 hour)

```bash
cd src/SelfIndex
python3 Run_Script.py medium
```

- Tests 50 selected configurations
- Uses 10,000 documents
- Balances accuracy and speed

#### 🔴 **Full Evaluation** (~3-4 hours)

```bash
cd src/SelfIndex
python3 Run_Script.py full
```

- Tests all 72 configurations
- Uses full 50,000 document dataset
- Provides complete performance profile
- Results saved to `comprehensive_selfindex_results/` directory

### Output Files

After evaluation, results are saved as:

- `comprehensive_results_YYYYMMDD_HHMMSS.json` - Complete metrics for all configurations
- Includes latency (P50, P95, P99), throughput (QPS), memory (MB), and quality metrics

---

## 📊 Project Dimensions Explained

### 1️⃣ **Index Type (x)** - Information Stored Per Term

| Type          | Code | What It Stores             | Memory   | Latency  | Quality                  |
| ------------- | ---- | -------------------------- | -------- | -------- | ------------------------ |
| **Boolean**   | x=1  | Just document IDs          | Minimal  | Fastest  | Basic (no ranking)       |
| **WordCount** | x=2  | Doc IDs + term frequencies | Moderate | Moderate | Better (frequency-based) |
| **TF-IDF**    | x=3  | Doc IDs + TF-IDF scores    | Maximum  | Slower   | Best (relevance ranking) |

### 2️⃣ **Storage Backend (y)** - Where Data Persists

| Backend    | Code | Technology                | Pros                 | Cons           |
| ---------- | ---- | ------------------------- | -------------------- | -------------- |
| **Custom** | y=1  | In-memory + Python pickle | Fast, simple         | Not persistent |
| **DB1**    | y=2  | SQLite (indexed design)   | Reliable persistence | Slower I/O     |

### 3️⃣ **Compression (z)** - Index Size Reduction

| Method         | Code | How It Works             | Space Saving | Throughput Impact |
| -------------- | ---- | ------------------------ | ------------ | ----------------- |
| **None**       | z=1  | Raw index data           | 0%           | Fastest           |
| **Dictionary** | z=2  | Variable-length encoding | ~20-30%      | -20-40% QPS       |
| **zlib**       | z=3  | DEFLATE compression      | ~40-60%      | -40-60% QPS       |

### 4️⃣ **Query Processing (q)** - How Queries Execute

| Strategy               | Code | Approach                                         | Best For         | Trade-off             |
| ---------------------- | ---- | ------------------------------------------------ | ---------------- | --------------------- |
| **Document-at-a-time** | q=D  | Process one document, collect all matching terms | Cache locality   | More operations       |
| **Term-at-a-time**     | q=T  | Process one term, collect all matching docs      | Working set size | Scatter memory access |

### 5️⃣ **Skip Pointers (i)** - Query Optimization

| Mode         | Code | What It Does                 | Benefit                                  |
| ------------ | ---- | ---------------------------- | ---------------------------------------- |
| **Disabled** | i=0  | Linear traversal of postings | Simpler                                  |
| **Enabled**  | i=1  | Jump to promising postings   | 10-30% faster for high-threshold queries |

---

## 💻 Manual Testing & Interactive Usage

### Interactive Python Session

```bash
cd src/SelfIndex
python3
```

```python
from self_index import SelfIndex

# Create an index
idx = SelfIndex(
    index_type='TFIDF',      # or 'BOOLEAN', 'WORDCOUNT'
    compression='NONE',      # or 'CODE', 'CLIB' (zlib)
    query_processing='DOCatat',  # or 'TERMatat'
    skip_pointers=True       # Enable skip pointers
)

# Create index with sample documents
documents = [
    ('doc1', 'artificial intelligence machine learning'),
    ('doc2', 'deep learning neural networks'),
    ('doc3', 'information retrieval search engines'),
]
idx.create_index('my_index', documents)

# Run a query
results = idx.query('machine learning')
print(results)
```

### Manual Testing Script

```bash
cd src/SelfIndex
python3 manual_test_index.py
```

This script:

- Creates indexes with different configurations
- Runs sample queries
- Displays timing and memory information
- Useful for debugging specific configurations

---

## 📈 Performance Metrics Explained

### Plot A: Latency Performance (seconds)

**What it measures**: Time to execute a single query

- **P50 (Median)**: 50% of queries faster than this
- **P95 (95th Percentile)**: 95% of queries faster than this (tail latency)
- **P99 (99th Percentile)**: Worst-case query performance

**Trade-offs by configuration**:

- Boolean index fastest (minimal postings processing)
- TF-IDF slowest (score computation overhead)
- Compression adds decompression latency
- Skip pointers reduce latency for selective queries

### Plot B: Throughput Performance (Queries/Second)

**What it measures**: Maximum query capacity of the system

- **Single-thread QPS**: Single processor capability
- **Multi-thread QPS**: Parallel query processing (when possible)
- **Speedup Factor**: Efficiency of parallelization

**Trade-offs by configuration**:

- No compression: Maximum throughput (~100-500 QPS)
- Dictionary compression: -20-40% throughput
- zlib compression: -40-60% throughput (CPU-bound)
- Python GIL limits multi-threading for CPU-bound operations

### Plot C: Memory Footprint (MB)

**What it measures**: RAM consumed by the index structure

- **Index Storage**: Size of inverted index (dominant)
- **Process Memory**: Total Python process memory
- **Peak Memory**: Maximum memory during operations

**Key findings**:

- Boolean index: 100-200 MB (minimal overhead)
- WordCount index: 200-400 MB (frequency storage)
- TF-IDF index: 300-800 MB (float scores)
- Compression reduces storage by 40-60%
- Query processing strategy: NO impact (uses same index)

---

## 🔍 Understanding the Code

### Main Files You'll Work With

#### `src/SelfIndex/index_base.py`

- **Purpose**: Abstract interface defining all index systems
- **Key Classes**: `IndexInfo`, `DataStore`, `Compression`, `QueryProc`, `Optimizations`, `IndexBase`
- **When to modify**: Adding new index types or storage backends

#### `src/SelfIndex/self_index.py` (⭐ Core)

- **Purpose**: Full SelfIndex implementation
- **Key Classes**: `InvertedListPointer`, `SelfIndex`
- **Main Methods**:
  - `create_index()` - Build index from documents
  - `query()` - Execute search query
  - `update_index()` - Add/remove documents
  - `save_index()` / `load_index()` - Persistence
- **Features**:
  - All 72 configuration combinations
  - Skip pointer implementation
  - Compression support (none, dictionary, zlib)
  - Multiple query processing strategies

#### `src/SelfIndex/optimized_selfindex_evaluator.py` (⭐ Evaluation)

- **Purpose**: Comprehensive benchmark framework
- **Key Classes**: `OptimizedSelfIndexEvaluator`
- **Main Methods**:
  - `run_comprehensive_evaluation()` - Test all configs
  - `measure_latency()` - Query timing (P50, P95, P99)
  - `measure_throughput()` - Queries per second
  - `measure_memory()` - Memory footprint
  - `measure_quality()` - Mean Average Precision (MAP)
- **Features**:
  - Single data load for all 72 configs
  - Efficient configuration testing
  - JSON results export
  - Matplotlib visualization generation

### Data Flow

```
Wikipedia Dataset (50K docs)
    ↓
CSV Preprocessing (tokenization, stemming)
    ↓
Cached Document Array
    ↓
SelfIndex Create/Query Operations
    ├─ Index Type (x): Boolean/WordCount/TF-IDF
    ├─ Storage (y): Custom/DB1
    ├─ Compression (z): None/Dictionary/zlib
    ├─ Query Processing (q): DOCatat/TERMatat
    └─ Optimizations (i): Skip Pointers on/off
    ↓
Performance Measurements
├─ Latency (P50, P95, P99)
├─ Throughput (Single/Multi-thread QPS)
├─ Memory (Index size, process memory)
└─ Quality (Mean Average Precision)
    ↓
JSON Results → Visualizations (Plots A, B, C)
    ↓
LaTeX Report (38 pages)
```

---

## 📊 Generating Plots & Report

### Generate Plots from Existing Results

```bash
cd src
python3 create_missing_plots.py
```

Generates:

- **Plot A**: Latency with P50/P95/P99 percentiles
- **Plot B**: Throughput in queries/second (QPS)
- **Plot C**: Memory footprint in MB

Plots are saved to `report_plots/` directory.

### Generate LaTeX Report

```bash
cd /path/to/project/root
pdflatex -interaction=nonstopmode comprehensive_enhanced_report.tex
```

Output:

- `comprehensive_enhanced_report.pdf` (38 pages)
- Contains all plots, analysis, and recommendations

### Report Sections

1. **Abstract** - Executive summary
2. **Introduction** - Project background and objectives
3. **System Design**
   - Architecture overview
   - Index types explanation
   - Storage backends and compression
   - Query processing strategies
4. **Index Type Analysis** (Plot A: Latency, Plot C: Memory)
5. **Storage Backend Analysis** (Plot A: Latency)
6. **Compression Algorithm Analysis** (Plot A: Latency, Plot B: Throughput)
7. **Skip Pointers Analysis** (Plot A: Latency)
8. **Query Processing Strategy** (Plot A: Latency, Plot C: Memory)
9. **Production Deployment Guide**
   - Memory-constrained systems
   - High-throughput systems
   - Balanced deployments
10. **Conclusion & Future Work**

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'datasets'"

**Solution**: Install requirements

```bash
pip install -r src/requirements.txt
```

### Issue: "CSV file not found"

**Solution**: Verify file exists

```bash
ls -lh src/Dataset/PreProcessedData/preprocessed_dataset.csv
```

If missing, download Wikipedia dataset:

```bash
cd src
python3 -c "from datasets import load_dataset; load_dataset('wikipedia', '20231101.en')"
```

### Issue: "NLTK punkt tokenizer not found"

**Solution**: Download NLTK data

```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: Memory error with 50K documents

**Solution**: Use smaller dataset

```python
# In Run_Script.py, change:
evaluator.run_comprehensive_evaluation(max_configs=72, sample_size=1000)  # Smaller sample
```

### Issue: Slow performance

**Solution**: Check system resources

```bash
# Monitor memory during evaluation
watch -n 1 'ps aux | grep python'

# Check disk space
df -h src/
```

---

## 📝 Key Formulas & Metrics

### Latency

```
Latency = Time to execute single query (milliseconds)
P50 = Median latency (50th percentile)
P95 = 95th percentile latency (tail behavior)
P99 = 99th percentile latency (worst case)
```

### Throughput

```
Throughput (QPS) = 1 / Average Latency (seconds)
Speedup = Multi-thread QPS / Single-thread QPS
Efficiency = Speedup / Number of Threads
```

### Memory

```
Index Storage = Size of inverted index file (MB)
Process Memory = Total RAM used by Python process (MB)
Peak Memory = Maximum memory during operations (MB)
Compression Ratio = Original Size / Compressed Size
```

### Quality (Mean Average Precision)

```
MAP = Mean of Average Precision scores across queries
Average Precision = Sum of (Precision@k × Relevance@k) / Relevant Documents
```

---

## 🎯 Common Tasks

### Run Quick Evaluation

```bash
cd src/SelfIndex
python3 Run_Script.py quick
```

### Analyze Specific Configuration

```bash
cd src/SelfIndex
python3
```

```python
from self_index import SelfIndex
idx = SelfIndex(index_type='TFIDF', compression='NONE', query_processing='DOCatat', skip_pointers=True)
# ... test your configuration
```

### Compare Two Configurations

```python
# See results in comprehensive_selfindex_results/comprehensive_results_*.json
import json
with open('comprehensive_selfindex_results/comprehensive_results_YYYYMMDD_HHMMSS.json') as f:
    results = json.load(f)

# Find specific configs
for config in results:
    if config['index_type'] == 'TFIDF' and config['compression'] == 'NONE':
        print(config['metrics'])
```

### Generate Custom Plots

```bash
cd src
python3 create_missing_plots.py
```

---

## 📚 Related Documentation

- **`src/DumpFiles/CodebaseExplanation.md`** - Detailed architecture breakdown
- **`src/DumpFiles/IMPLEMENTATION_SUMMARY.md`** - Implementation details
- **`PLOT_BC_COMPLIANCE_SUMMARY.md`** - Plot B/C methodology
- **`comprehensive_enhanced_report.tex`** - Full LaTeX source

---

## ✅ Assignment Compliance

### Plot A: Latency with Percentiles (P50, P95, P99)

- ✅ Index Types: Boolean, WordCount, TF-IDF
- ✅ Storage Backends: Custom, DB1
- ✅ Compression: None, Dictionary, zlib
- ✅ Skip Pointers: Enabled, Disabled
- ✅ Query Processing: DOCatat, TERMatat

### Plot B: Throughput (Queries/Second)

- ✅ Compression Algorithm: Comprehensive QPS analysis

### Plot C: Memory Footprint (MB)

- ✅ Index Types: Memory progression
- ✅ Query Processing: Memory invariance analysis

---

## 🤝 Contributing

To modify or extend the project:

1. **Add new index type**: Modify `index_base.py`, implement in `self_index.py`
2. **New storage backend**: Extend `DataStore` enum, implement in `self_index.py`
3. **New compression**: Add to `Compression` enum, implement compression logic
4. **Custom evaluations**: Extend `OptimizedSelfIndexEvaluator` class

---

## 📞 Support

For issues or questions:

1. Check `DumpFiles/CodebaseExplanation.md` for architecture details
2. Review `PLOT_BC_COMPLIANCE_SUMMARY.md` for metrics explanation
3. See comprehensive report PDF for formal analysis
4. Check test files in `DumpFiles/` for usage examples

---

## 📄 Project Information

- **Course**: Information Retrieval & Evaluation (IRE)
- **Institution**: IIIT Hyderabad, Semester 3
- **Assignment Type**: Comprehensive System Design & Evaluation
- **Report**: 38 pages with 12+ visualizations
- **Total Configurations**: 72 index variants
- **Dataset**: 50,000 Wikipedia documents
- **Evaluation Time**: 3-4 hours (full run)

---

## 📋 Quick Reference Card

```bash
# Setup
cd src && pip install -r requirements.txt

# Quick test (10 min)
cd SelfIndex && python3 Run_Script.py quick

# Medium test (1 hour)
cd SelfIndex && python3 Run_Script.py medium

# Full evaluation (3-4 hours)
cd SelfIndex && python3 Run_Script.py full

# Generate plots
cd .. && python3 create_missing_plots.py

# Build report
cd /path/to/root && pdflatex comprehensive_enhanced_report.tex

# Manual testing
cd SelfIndex && python3 manual_test_index.py
```

---

**Last Updated**: November 4, 2025  
**Report Version**: 38 pages  
**Project Status**: Complete with all 72 configurations evaluated
