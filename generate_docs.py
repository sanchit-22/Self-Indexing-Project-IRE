#!/usr/bin/env python3
"""
Script to generate comprehensive documentation for Self-Indexing Project
"""

import os
from pathlib import Path

def create_technical_guide():
    """Create the comprehensive technical guide"""
    
    content = """# COMPREHENSIVE TECHNICAL GUIDE
# Self-Indexing Project for Information Retrieval and Evaluation

**Version:** 1.0  
**Last Updated:** November 17, 2025  
**Course:** Information Retrieval & Evaluation (IRE)  
**Institution:** IIIT Hyderabad, Semester 3  
**Project Type:** Comprehensive System Design & Implementation  
**Total Configurations Evaluated:** 72 unique index variants  
**Dataset Size:** 50,000 Wikipedia documents

---

## EXECUTIVE SUMMARY

This document provides a comprehensive, in-depth technical guide to the Self-Indexing Project—a production-grade information retrieval system implementing 72 unique configurations. The guide progresses from fundamental concepts to advanced implementation details, covering architecture, algorithms, code structure, performance analysis, and operational guidance.

**Key Highlights:**
- **Scope:** 2,771 lines of Python code implementing a full-featured search engine
- **Index Types:** Boolean, WordCount, TF-IDF with complete ranking algorithms
- **Storage:** Custom in-memory, SQLite database, JSON-based backends
- **Compression:** None, Dictionary encoding, zlib DEFLATE algorithm
- **Query Processing:** Term-at-a-time and Document-at-a-time strategies
- **Optimizations:** Skip pointers, decompression caching, heap-based selection
- **Evaluation:** Comprehensive performance metrics (latency, throughput, memory, quality)

---

## TABLE OF CONTENTS

### PART I: FOUNDATIONS
1. [Introduction and Project Overview](#1-introduction-and-project-overview)
2. [Fundamental Concepts of Information Retrieval](#2-fundamental-concepts-of-information-retrieval)
3. [Text Processing and Linguistic Foundations](#3-text-processing-and-linguistic-foundations)
4. [Inverted Index Architecture](#4-inverted-index-architecture)

### PART II: SYSTEM DESIGN
5. [System Architecture and Design Principles](#5-system-architecture-and-design-principles)
6. [Index Types: Boolean, WordCount, and TF-IDF](#6-index-types-boolean-wordcount-and-tfidf)
7. [Storage Backends and Persistence](#7-storage-backends-and-persistence)
8. [Compression Algorithms and Techniques](#8-compression-algorithms-and-techniques)

### PART III: ALGORITHMS
9. [Query Processing Strategies](#9-query-processing-strategies)
10. [Skip Pointers and Query Optimization](#10-skip-pointers-and-query-optimization)
11. [Boolean Query Processing](#11-boolean-query-processing)
12. [Ranking and Scoring Algorithms](#12-ranking-and-scoring-algorithms)

### PART IV: IMPLEMENTATION
13. [Detailed Code Structure](#13-detailed-code-structure)
14. [Module-by-Module Implementation Analysis](#14-module-by-module-implementation-analysis)
15. [Data Structures and Memory Management](#15-data-structures-and-memory-management)
16. [Performance Optimization Techniques](#16-performance-optimization-techniques)

### PART V: EVALUATION
17. [Evaluation Framework and Methodology](#17-evaluation-framework-and-methodology)
18. [Performance Metrics and Measurements](#18-performance-metrics-and-measurements)
19. [Results Analysis: 72 Configuration Comparison](#19-results-analysis-72-configuration-comparison)
20. [Trade-offs and Design Decisions](#20-trade-offs-and-design-decisions)

### PART VI: PRACTICAL GUIDE
21. [Complete Setup and Installation Guide](#21-complete-setup-and-installation-guide)
22. [Usage Examples and Tutorials](#22-usage-examples-and-tutorials)
23. [Troubleshooting and Debugging](#23-troubleshooting-and-debugging)
24. [Production Deployment Guide](#24-production-deployment-guide)

### PART VII: ADVANCED TOPICS
25. [Scalability and Performance Tuning](#25-scalability-and-performance-tuning)
26. [Extending the System](#26-extending-the-system)
27. [Future Enhancements and Research Directions](#27-future-enhancements-and-research-directions)
28. [References and Further Reading](#28-references-and-further-reading)

---

# PART I: FOUNDATIONS

## 1. Introduction and Project Overview

### 1.1 Project Context and Motivation

Information Retrieval (IR) is a foundational discipline in computer science that addresses the challenge of finding relevant information from large document collections. In an era where data grows exponentially, efficient search systems are critical infrastructure for digital society.

This Self-Indexing Project implements a **complete search engine from scratch** to understand the fundamental algorithms, data structures, and trade-offs that power modern search systems like Google, Elasticsearch, and Apache Solr.

### 1.2 Project Objectives

**Primary Objectives:**

1. **Implement Core IR Algorithms**
   - Build inverted indexes with multiple scoring schemes
   - Implement Boolean query processing
   - Develop ranking algorithms (Boolean, TF, TF-IDF)

2. **Explore Design Space**
   - Evaluate 72 unique configurations systematically
   - Understand trade-offs between latency, throughput, memory, and quality
   - Compare storage backends and compression strategies

3. **Measure Real-World Performance**
   - Use 50,000 Wikipedia documents as realistic corpus
   - Measure latency percentiles (P50, P95, P99)
   - Benchmark throughput (queries per second)
   - Profile memory usage and index sizes

4. **Document Comprehensively**
   - Provide end-to-end technical guide
   - Explain every design decision
   - Enable reproducibility and extensibility

**Secondary Objectives:**

- Understand production system requirements
- Learn performance engineering principles
- Practice scientific evaluation methodology
- Develop software engineering best practices

### 1.3 System Capabilities

The Self-Indexing system supports:

**Index Types:**
- **Boolean Index**: Binary presence/absence indexing
- **WordCount Index**: Frequency-based indexing with term counts
- **TF-IDF Index**: Relevance-based indexing with TF-IDF scores

**Storage Backends:**
- **Custom Storage**: In-memory with Python pickle serialization
- **SQLite (DB1)**: SQL database with relational tables and B-tree indexes
- **JSON Database (DB2)**: Lightweight JSON-based storage

**Compression Methods:**
- **No Compression (NONE)**: Raw data storage for maximum speed
- **Dictionary Encoding (CODE)**: Delta compression with variable-length encoding
- **zlib Compression (CLIB)**: DEFLATE-based compression for maximum space savings

**Query Processing:**
- **Term-at-a-time (TERMatat)**: Process all documents for each term sequentially
- **Document-at-a-time (DOCatat)**: Process all terms for each document sequentially

**Optimizations:**
- **Skip Pointers**: Jump ahead in posting lists to skip irrelevant documents
- **Decompression Caching**: Cache decompressed postings for repeated access
- **Query Preprocessing Cache**: Cache tokenized queries
- **Heap-based Top-K Selection**: Efficient retrieval of top results

**Query Features:**
- Simple multi-term queries
- Boolean operators (AND, OR, NOT)
- Phrase queries (exact sequence matching)
- Ranked retrieval with scoring

### 1.4 Technology Stack and Dependencies

**Core Technologies:**

- **Python 3.8+**: Primary programming language
  - Chosen for: Rapid development, rich libraries, readability
  - Version requirement: 3.8+ for dictionary ordering guarantees

- **NLTK (Natural Language Toolkit)**
  - Purpose: Text preprocessing, tokenization, stemming
  - Components used:
    - `word_tokenize`: Word boundary detection
    - `PorterStemmer`: Morphological analysis and stemming
    - `stopwords`: Common word filtering

- **SQLite 3**
  - Purpose: Relational database backend
  - Features used:
    - BLOB storage for serialized postings
    - B-tree indexes for fast lookups
    - ACID transactions

**Data Processing:**

- **Pandas**: CSV processing and data manipulation
- **NumPy**: Numerical array operations
- **HuggingFace Datasets**: Wikipedia corpus access

**Evaluation and Visualization:**

- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization
- **psutil**: System monitoring and memory profiling

**Development Tools:**

- **Git**: Version control
- **pytest**: Testing framework
- **LaTeX**: Report generation

### 1.5 Dataset: 50,000 Wikipedia Documents

**Source:** HuggingFace `datasets` library  
**Collection:** Wikipedia dump from November 2023  
**Language:** English  
**Size:** 50,000 articles

**Dataset Characteristics:**

- **Average document length:** ~500 tokens (after preprocessing)
- **Total vocabulary:** ~150,000 unique terms
- **Total tokens:** ~25 million
- **Topics:** Diverse coverage across all Wikipedia categories
- **Quality:** High-quality, well-structured text

**Preprocessing Pipeline:**

1. **Raw Text Extraction**: UTF-8 encoded Wikipedia markup
2. **Tokenization**: Word boundary detection using NLTK
3. **Lowercasing**: Case normalization for case-insensitive search
4. **Punctuation Removal**: Strip non-alphanumeric characters
5. **Stopword Filtering**: Remove common words (the, is, and, etc.)
6. **Stemming**: Reduce words to root form using Porter Stemmer

**Storage Format:**

Preprocessed dataset stored as CSV:
```
id,original_text,processed_tokens,title,token_count
12,\"Wikipedia article text...\",\"token1 token2 token3...\",\"Article Title\",450
```

### 1.6 Configuration Space: 72 Unique Variants

The system evaluates **72 configurations** formed by all combinations of:

| Dimension | Variable | Options | Count |
|-----------|----------|---------|-------|
| **Index Type** | x | Boolean (1), WordCount (2), TF-IDF (3) | 3 |
| **Storage Backend** | y | Custom (1), SQLite/DB1 (2) | 2 |
| **Compression** | z | None (1), Dictionary/CODE (2), zlib/CLIB (3) | 3 |
| **Query Processing** | q | Document-at-a-time (D), Term-at-a-time (T) | 2 |
| **Optimization** | i | Skip Pointers Off (0), On (1) | 2 |

**Total:** 3 × 2 × 3 × 2 × 2 = **72 configurations**

**Configuration Identifier Format:**
```
SelfIndex_i{x}d{y}c{z}q{q}o{i}

Example: SelfIndex_i3d1c2qDo1
  - i3: TF-IDF index
  - d1: Custom storage
  - c2: Dictionary compression
  - qD: Document-at-a-time query processing
  - o1: Skip pointers enabled
```

### 1.7 Performance Metrics

**Latency Metrics:**
- **P50 (Median)**: 50% of queries complete faster
- **P95**: 95% of queries complete faster (tail latency)
- **P99**: 99% of queries complete faster (worst-case)
- **Mean**: Average query execution time

**Throughput Metrics:**
- **Single-thread QPS**: Queries per second (sequential processing)
- **Multi-thread QPS**: Queries per second (parallel processing)
- **Speedup Factor**: Multi-thread / Single-thread performance ratio

**Memory Metrics:**
- **Index Size (MB)**: Disk space consumed by inverted index
- **Process Memory (MB)**: Total RAM used by Python process
- **Peak Memory (MB)**: Maximum memory during operations
- **Compression Ratio**: Original size / Compressed size

**Quality Metrics:**
- **Mean Average Precision (MAP)**: Ranking quality
- **Coverage Rate**: Fraction of queries returning results
- **Result Count**: Average number of matches per query

---

## 2. Fundamental Concepts of Information Retrieval

### 2.1 What is Information Retrieval?

**Definition:** Information Retrieval (IR) is the science of searching for information in documents, searching for documents themselves, searching for metadata about documents, or searching within databases.

**Key Components:**

1. **Document Collection**: Set of textual documents to search
2. **Queries**: User information needs expressed as text
3. **Retrieval Model**: Algorithm to match queries with documents
4. **Ranking Function**: Method to order results by relevance
5. **User Interface**: Presentation and interaction layer

**IR vs. Database Queries:**

| Aspect | Database Query | Information Retrieval |
|--------|---------------|----------------------|
| **Match Type** | Exact | Approximate |
| **Query Language** | Structured (SQL) | Natural language |
| **Results** | All matching records | Ranked by relevance |
| **Schema** | Fixed, predefined | Unstructured text |
| **Semantics** | Precise | Ambiguous |

**Example:**

**Database Query (SQL):**
```sql
SELECT * FROM articles WHERE title = 'Machine Learning' AND year > 2020;
```
- Returns: Exact matches only
- Semantics: Precise, unambiguous
- Ranking: None (or by single column)

**Information Retrieval Query:**
```
machine learning algorithms
```
- Returns: Documents containing any/all terms
- Semantics: Fuzzy, context-dependent
- Ranking: By relevance score (TF-IDF, BM25, etc.)

### 2.2 The Information Retrieval Process

**Step-by-Step Workflow:**

1. **Document Collection**
   - Gather documents from various sources
   - Clean and normalize text
   - Extract structured information

2. **Document Processing**
   - Tokenization: Split text into terms
   - Linguistic analysis: Stemming, lemmatization
   - Feature extraction: Terms, phrases, entities

3. **Index Construction**
   - Build inverted index: term → documents
   - Store positional information
   - Calculate statistical features (TF, IDF)

4. **Query Processing**
   - Parse user query
   - Apply same preprocessing as documents
   - Identify query terms and operators

5. **Document Matching**
   - Retrieve posting lists for query terms
   - Compute relevance scores
   - Apply Boolean logic (if applicable)

6. **Ranking**
   - Sort documents by relevance score
   - Apply re-ranking models (if available)
   - Select top-k results

7. **Result Presentation**
   - Format results for display
   - Generate snippets/summaries
   - Provide relevance feedback mechanism

### 2.3 Retrieval Models

#### 2.3.1 Boolean Model

**Concept:** Documents either match or don't match the query (binary decision)

**Query Example:**
```
(machine AND learning) OR (artificial AND intelligence)
```

**Advantages:**
- Simple to understand and implement
- Fast query processing
- Precise control over results

**Disadvantages:**
- No ranking (all results equally relevant)
- Too restrictive (may return no results)
- Too permissive (may return too many results)

**Implementation in Self-Indexing:**
- Boolean index type (x=1)
- Supports AND, OR, NOT operators
- Score = 1 if match, 0 otherwise

#### 2.3.2 Vector Space Model

**Concept:** Represent documents and queries as vectors in high-dimensional term space

**Document Vector:**
```
Document = [w1, w2, w3, ..., wn]
  where wi = weight of term i (TF-IDF, TF, or binary)
```

**Query Vector:**
```
Query = [q1, q2, q3, ..., qn]
  where qi = weight of term i in query
```

**Similarity Measure:** Cosine similarity
```
similarity(D, Q) = (D · Q) / (|D| × |Q|)
  = Σ(wi × qi) / √(Σwi²) × √(Σqi²)
```

**Advantages:**
- Partial matching (documents can partially satisfy query)
- Ranked results (continuous scores)
- Handles term weights naturally

**Disadvantages:**
- Assumes term independence
- Ignores term order and proximity
- No semantic understanding

**Implementation in Self-Indexing:**
- WordCount index (x=2): weights = term frequencies
- TF-IDF index (x=3): weights = TF-IDF scores
- Ranking by sum of term weights

#### 2.3.3 Probabilistic Model (BM25)

**Concept:** Estimate probability that document is relevant given query

**BM25 Formula:**
```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))

where:
  - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
  - f(qi, D) = frequency of term qi in document D
  - |D| = length of document D
  - avgdl = average document length
  - k1, b = tuning parameters (typically k1=1.2, b=0.75)
```

**Advantages:**
- State-of-the-art ranking quality
- Robust across different domains
- Theoretically grounded (probabilistic)

**Disadvantages:**
- More complex to implement
- Requires parameter tuning
- Slower than simpler models

**Note:** BM25 is not directly implemented in this project but TF-IDF is a related approach

### 2.4 Evaluation Metrics

#### 2.4.1 Precision and Recall

**Precision:** Fraction of retrieved documents that are relevant
```
Precision = |Relevant ∩ Retrieved| / |Retrieved|
```

**Recall:** Fraction of relevant documents that are retrieved
```
Recall = |Relevant ∩ Retrieved| / |Relevant|
```

**Example:**
```
Total documents: 1000
Relevant documents: 20
Retrieved documents: 15
Relevant AND Retrieved: 10

Precision = 10/15 = 0.667 (66.7%)
Recall = 10/20 = 0.500 (50.0%)
```

**Trade-off:**
- High precision: Conservative retrieval (few false positives)
- High recall: Liberal retrieval (few false negatives)
- Usually can't maximize both simultaneously

#### 2.4.2 F-Measure (F1 Score)

**Definition:** Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Properties:**
- Balanced measure (equal weight to precision and recall)
- Range: [0, 1], higher is better
- Only high if both precision and recall are high

#### 2.4.3 Mean Average Precision (MAP)

**Definition:** Mean of average precision scores across all queries

**Average Precision for Single Query:**
```
AP = (Σ (Precision@k × rel(k))) / Number of relevant documents

where:
  - Precision@k = Precision at rank k
  - rel(k) = 1 if document at rank k is relevant, 0 otherwise
```

**Mean Average Precision:**
```
MAP = (Σ AP(q)) / Number of queries
```

**Example:**
```
Query results: [Relevant, Not, Relevant, Relevant, Not]
Relevant documents at ranks: 1, 3, 4

Precision@1 = 1/1 = 1.000
Precision@3 = 2/3 = 0.667
Precision@4 = 3/4 = 0.750

AP = (1.000 + 0.667 + 0.750) / 3 = 0.806
```

**Interpretation:**
- MAP = 1.0: Perfect ranking (all relevant docs at top)
- MAP = 0.5: Moderate ranking quality
- MAP = 0.1: Poor ranking quality

#### 2.4.4 Normalized Discounted Cumulative Gain (NDCG)

**Concept:** Reward systems that rank highly relevant documents higher

**Discounted Cumulative Gain (DCG):**
```
DCG@k = Σ (rel_i / log2(i + 1))  for i = 1 to k

where rel_i = relevance grade of document at position i
```

**Normalized DCG:**
```
NDCG@k = DCG@k / IDCG@k

where IDCG@k = DCG@k for ideal ranking
```

**Properties:**
- Accounts for position (higher ranks weighted more)
- Supports graded relevance (not just binary)
- Normalized to [0, 1] range

---

## 3. Text Processing and Linguistic Foundations

### 3.1 Text Preprocessing Pipeline

Text preprocessing is critical for effective information retrieval. Raw text must be transformed into a normalized representation suitable for indexing and retrieval.

**Complete Pipeline:**

```
Raw Text
  ↓
Tokenization
  ↓
Lowercasing
  ↓
Punctuation Removal
  ↓
Stopword Removal
  ↓
Stemming/Lemmatization
  ↓
Normalized Tokens
```

#### 3.1.1 Tokenization

**Purpose:** Split continuous text into discrete units (tokens)

**Challenges:**
- Word boundaries: "don't" → ["don", "t"] or ["don't"]?
- Hyphenated words: "state-of-the-art" → ["state", "of", "the", "art"]?
- Abbreviations: "U.S.A." → ["U", "S", "A"] or ["USA"]?
- Special characters: Emails, URLs, hashtags

**NLTK word_tokenize:**
- Based on Penn Treebank tokenization
- Handles common English patterns
- Splits on whitespace and punctuation

**Example:**
```python
import nltk
from nltk.tokenize import word_tokenize

text = "Machine learning is fascinating! It's the future."
tokens = word_tokenize(text)
# Result: ['Machine', 'learning', 'is', 'fascinating', '!', 'It', \"'s\", 'the', 'future', '.']
```

**Alternative Tokenizers:**
- **Whitespace**: `text.split()` - simple but crude
- **Regex**: `re.findall(r'\\w+', text)` - customizable patterns
- **Subword**: BPE, WordPiece - for neural models

#### 3.1.2 Lowercasing

**Purpose:** Normalize case for case-insensitive matching

**Advantages:**
- "Machine" and "machine" treated identically
- Reduces vocabulary size (~30-40%)
- Improves recall

**Disadvantages:**
- Loses information (proper nouns, acronyms)
- "US" (country) vs "us" (pronoun) - ambiguity

**Implementation:**
```python
tokens = [token.lower() for token in tokens]
# ['machine', 'learning', 'is', 'fascinating', '!', 'it', \"'s\", 'the', 'future', '.']
```

**Alternatives:**
- **Truecasing**: Preserve case for proper nouns
- **Mixed**: Lowercase all but specific terms

#### 3.1.3 Punctuation Removal

**Purpose:** Remove non-alphanumeric characters

**Rationale:**
- Punctuation rarely affects meaning in IR
- Reduces index size
- Simplifies matching

**Implementation:**
```python
import string
punct_table = str.maketrans('', '', string.punctuation)
tokens = [token.translate(punct_table) for token in tokens if token.isalpha()]
# ['machine', 'learning', 'is', 'fascinating', 'it', 's', 'the', 'future']
```

**Considerations:**
- Keep punctuation for: emails, URLs, code
- Remove for: general text search
- Context-dependent decision

#### 3.1.4 Stopword Removal

**Purpose:** Remove high-frequency, low-information words

**Common Stopwords:**
- Articles: the, a, an
- Prepositions: in, on, at, to, from
- Pronouns: I, you, he, she, it
- Auxiliary verbs: is, are, was, were
- Conjunctions: and, or, but

**NLTK Stopword List:**
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# 179 English stopwords
```

**Implementation:**
```python
tokens = [token for token in tokens if token not in stop_words]
# ['machine', 'learning', 'fascinating', 'future']
```

**Trade-offs:**

| Aspect | With Stopword Removal | Without |
|--------|----------------------|---------|
| **Index Size** | 40-50% smaller | Larger |
| **Query Speed** | Faster | Slower |
| **Phrase Queries** | Broken | Preserved |
| **Recall** | Lower (missing stopwords) | Higher |

**Best Practices:**
- Remove for general search
- Keep for phrase queries ("to be or not to be")
- Domain-specific stopword lists

#### 3.1.5 Stemming

**Purpose:** Reduce words to their root form (stem)

**Rationale:**
- "running", "runs", "ran" → "run"
- Increases recall (matches morphological variants)
- Reduces vocabulary size

**Porter Stemmer Algorithm:**

**Step 1: Remove plurals and verb suffixes**
- SSES → SS (caresses → caress)
- IES → I (ponies → poni)
- S → ε (cats → cat)

**Step 2: Remove verb endings**
- ED → ε (agreed → agre)
- ING → ε (hoping → hop)

**Step 3-5: Remove derivational suffixes**
- ATIONAL → ATE (relational → relate)
- IZER → IZE (digitizer → digitize)
- FUL → ε (hopeful → hope)

**Implementation:**
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

tokens = ['machine', 'learning', 'fascinating', 'future']
stemmed = [stemmer.stem(token) for token in tokens]
# ['machin', 'learn', 'fascin', 'futur']
```

**Limitations:**
- Over-stemming: "organization", "organ" → "organ"
- Under-stemming: "alumnus", "alumni" → different stems
- Non-semantic: "universe", "university" → "univers"

**Alternatives:**
- **Lemmatization**: Use dictionary to find base form
  - "better" → "good" (not "better")
  - More accurate but slower
  - Requires part-of-speech tagging
  
- **No Stemming**: Keep original forms
  - Highest precision
  - Lower recall
  - Larger vocabulary

### 3.2 Example: Complete Preprocessing

**Input Text:**
```
\"The Machine Learning algorithms are revolutionizing AI! 
They're becoming increasingly sophisticated and powerful.\"
```

**Step 1: Tokenization**
```python
['The', 'Machine', 'Learning', 'algorithms', 'are', 'revolutionizing', 
 'AI', '!', 'They', \"'re\", 'becoming', 'increasingly', 'sophisticated', 
 'and', 'powerful', '.']
```

**Step 2: Lowercasing**
```python
['the', 'machine', 'learning', 'algorithms', 'are', 'revolutionizing', 
 'ai', '!', 'they', \"'re\", 'becoming', 'increasingly', 'sophisticated', 
 'and', 'powerful', '.']
```

**Step 3: Punctuation Removal + Alpha Filter**
```python
['the', 'machine', 'learning', 'algorithms', 'are', 'revolutionizing', 
 'ai', 'they', 're', 'becoming', 'increasingly', 'sophisticated', 
 'and', 'powerful']
```

**Step 4: Stopword Removal**
```python
['machine', 'learning', 'algorithms', 'revolutionizing', 'ai', 
 'becoming', 'increasingly', 'sophisticated', 'powerful']
```

**Step 5: Stemming**
```python
['machin', 'learn', 'algorithm', 'revolution', 'ai', 
 'becom', 'increasingli', 'sophist', 'power']
```

**Final Tokens for Indexing:**
```python
['machin', 'learn', 'algorithm', 'revolution', 'ai', 
 'becom', 'increasingli', 'sophist', 'power']
```

**Impact on Search:**
- Query "machine learning" matches "Machine Learning", "MACHINE LEARNING", etc.
- Query "algorithms" matches "algorithm", "algorithms"  
- Vocabulary reduced from ~20 unique words to ~9 stems

### 3.3 Linguistic Challenges in IR

#### 3.3.1 Synonymy

**Problem:** Different words, same meaning  
**Examples:**
- "car" vs "automobile"  
- "purchase" vs "buy"  
- "big" vs "large"

**Impact:** Lower recall (miss relevant documents with synonyms)

**Solutions:**
- Query expansion: Add synonyms to query
- Thesaurus: WordNet, domain-specific dictionaries
- Semantic embeddings: Word2Vec, GloVe, BERT

#### 3.3.2 Polysemy

**Problem:** Same word, different meanings  
**Examples:**
- "bank" (financial institution vs river bank)  
- "bat" (animal vs sports equipment)  
- "python" (snake vs programming language)

**Impact:** Lower precision (retrieve irrelevant documents)

**Solutions:**
- Context analysis: Use surrounding words
- Word sense disambiguation: Identify correct meaning
- User feedback: Clarify intent

#### 3.3.3 Mor"
