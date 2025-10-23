# 🚩 Remaining Tasks for Full Marks - Indexing & Retrieval Assignment

This document summarizes **what is already completed** and **what is missing** based on your codebase and assignment requirements. It is structured so you can hand it directly to a developer or agent to finish the assignment for full marks.

---

## ✅ What is Already Done

1. **Data Loading & Preprocessing**
   - Wikipedia data loaded from Huggingface (`wikimedia/wikipedia`, split `20231101.en`)
   - Preprocessing: Lowercasing, punctuation removal, tokenization, stopword removal, stemming
   - Word frequency plots (before & after preprocessing)

2. **Elasticsearch Indexing (ESIndex-v1.0)**
   - Bulk indexing into ES with optimized settings
   - Querying, stats retrieval, and index management
   - Metrics A (latency), B (throughput), C (memory), D (functional) measured and plotted
   - Diverse query set with justification

3. **SelfIndex - Boilerplate Present**
   - Abstract base class (`index_base.py`) and a template for custom index (`self_index.py`)
   - Versioning system for SelfIndex-v1.xyziq is present and partially configurable

---

## ❗ What is **NOT** Done or Needs Improvement

### 1. **SelfIndex - Full Implementation**

- [ ] **Boolean Index (x=1)**
  - Should support: Document IDs and position IDs (done/partial in boilerplate, but check for full query support)

- [ ] **WordCount Index (x=2)**
  - Should support: Ranking with term frequencies/word counts
  - Verify: Are results ranked using word counts? Implement if missing.

- [ ] **TF-IDF Index (x=3)**
  - Should support: Ranking with precomputed TF-IDF
  - Check: Are TF-IDF scores calculated and used for ranking in queries?

### 2. **Boolean Query Support (CRITICAL)**
- [ ] **Support for Boolean Queries:**  
  - Operators: AND, OR, NOT, PHRASE (with parentheses and correct precedence)
  - Query grammar as specified in assignment
- [ ] **Operator Precedence:**  
  - Highest: PHRASE, then NOT, then AND, then OR
- [ ] **PHRASE Query Support:**  
  - E.g., `"political philosophy"`
- [ ] **Parentheses Support:**  
  - E.g., `"Apple" AND ("Banana" OR "Orange")`
- [ ] **Comprehensive tests for Boolean queries**

### 3. **Multiple Datastore Backends (y=1,2)**
- [ ] **Custom (y=1):**  
  - Already implemented with pickle/json.
- [ ] **Off-the-shelf DBs (y=2, y=3):**  
  - Implement at least **two**: e.g., **PostgreSQL GIN**, **RocksDB**, **Redis**
  - At minimum:  
    - Implement index persistence and loading with each DB (even if simple)
    - Add code to select backend via versioning (SelfIndex-v1.xyziq)
    - Document very briefly the pros/cons of each DB

### 4. **Compression on Postings List (z=1,2,3)**
- [ ] **z=1:** No compression (already present)
- [ ] **z=2:** Simple custom code (e.g., gap encoding, variable byte encoding)
- [ ] **z=3:** Library compression (e.g., zlib/gzip)
- [ ] **Expose compression choice in versioning/config & plot impact on metrics**

### 5. **Index Optimizations (i=0/1)**
- [ ] **i=0:** Null (no optimization)
- [ ] **i=sp/th/es:** Implement at least one:
  - Skipping with pointers (`sp`), thresholding (`th`), or early stopping (`es`)
- [ ] **Expose optimization in versioning/config & compare latency/throughput**

### 6. **Query Processing Strategies (q=Tn/Dn)**
- [ ] **q=Tn:** Term-at-a-time query processing (already default)
- [ ] **q=Dn:** Document-at-a-time processing
- [ ] **Allow selecting in config/versioning**
- [ ] **Plot/compare impact on performance**

### 7. **Comparison Plots & Reporting**
- [ ] **For each variant (x, y, z, i, q):**
  - Plot and compare metrics with ESIndex for the same query set
  - Make sure plots clearly label which variant/config is being compared
- [ ] **Functional metrics (precision, recall, ranking) for SelfIndex**
- [ ] **A/B/C/D metrics for all variants**

### 8. **Persistence & Auto-Loading**
- [ ] **Auto-load all indices from disk at server/notebook start** (for SelfIndex, not just ES)
- [ ] **List and manage existing indices from disk**

### 9. **Documentation & Instructions**
- [ ] **Update README/Notebook**:
  - How to run SelfIndex for all variants
  - How to run all metrics and comparison plots
  - How to add new data/query sets

---

## 📝 **How to Finish the Assignment**

1. **Finish/extend `self_index.py`** to ensure all items above are implemented for SelfIndex-v1.xyziq.
2. **Add code/tests for Boolean/PHRASE/parentheses queries.**
3. **Implement DB1 and DB2 backends** .
4. **Implement/combine compression methods** for posting lists and allow switching.
5. **Add at least one optimization (skipping/thresholding/early stopping).**
6. **Allow query processing strategy selection (term-at-a-time, doc-at-a-time).**
7. **Add/extend notebook code to run and plot all metrics for every variant**
8. **Document the full process for a grader.**

---

## 📣 **NOTE**
**Please make sure to update the indexing dropdown in Interactive_SelfIndex_Version2.ipynb file alongwith the required changes needed.**

---