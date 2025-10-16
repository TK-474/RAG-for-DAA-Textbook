Retrieval Augmented Generation for Algorithm Textbook

Problem Statement
While studying complex courses like Design and Analysis of Algorithms, we found that answers to difficult questions could be found in textbooks, but LLMs alone were not effective at explaining concepts in the way professors teach in class. This project explores whether Retrieval Augmented Generation (RAG) over course textbooks can bridge this gap.
We implemented RAG using the textbook "Introduction to the Design and Analysis of Algorithms" by Anany Levitin.
Platform Details
Environment: Google Colab
GPU: T4-GPU
Dataset
Source: "Introduction to the Design and Analysis of Algorithms" by Anany Levitin
Processing: Split by page, treating each page as a document
Corpus Size: 593 documents
Methodology
Retrieval Methods
We experimented with three retrieval strategies:
1. Keyword-based Retrieval (BM25)
Used rank_bm25 for traditional keyword-based indexing
Computes TF-IDF scores for documents relative to queries
Effective for exact term matching but less robust for semantic similarity
2. Semantic Retrieval (Dense Embeddings)
Embedding Model: BAAI/bge-small-en via HuggingFaceEmbeddings
Vector Database: FAISS for fast similarity search using cosine similarity
Excels at semantic matching even when exact terms are absent
3. Hybrid Retrieval (Reciprocal Rank Fusion)
Combined BM25 and dense retrieval using Reciprocal Rank Fusion (RRF)
Fused rankings from both methods to compute unified relevance scores
Justification: Balanced lexical and semantic similarity, consistently improving retrieval quality. BM25 performed better on formula-heavy content while dense retrieval captured semantic intent.
Language Models
We tested the following instruction-tuned models from Hugging Face:
Qwen/Qwen2.5-3B-Instruct
microsoft/phi-2
meta-llama/Llama-3.2-3B-Instruct
Justification: Small instruction-tuned models were chosen to balance quality with inference efficiency on limited GPU memory. Qwen and Llama-3.2 produced the most coherent and factual answers.
Chunking Strategy
Method: RecursiveCharacterTextSplitter from LangChain
Parameters:
chunk_size: 500 and 1000
chunk_overlap: 50
Justification: Recursive splitting preserves semantic continuity around section boundaries
Prompt Engineering
Retrieved context was structured with:
Clear enumeration (1., 2., 3.)
Explicit instructions for the LLM to merge, compare, or reject based on relevance
Instructions to provide coherent answers, mention contradictions, and output "no info found" for irrelevant retrieval
This approach reduced hallucination and increased factual alignment.
Evaluation Framework
We implemented a comprehensive LLM-based evaluation with three components:
Part 1: Relevance of Retrieval
Assessed whether each retrieved chunk was relevant to the question
Provided justification for each assessment
Part 2: Faithfulness of Retrieval
Broke down generated answers into individual factual claims
Verified if each claim was supported by retrieved content
Identified supporting chunks
Faithfulness Score = (Number of Supported Claims) / (Total Number of Claims)
Part 3: LLM Response Evaluation
Rated answers on a scale of 1-5 across five dimensions:
Correctness: Accuracy compared to ground truth
Relevance: Focus on core aspects of the question
Coherence: Clarity and logical flow
Completeness: Coverage of expected scope
Faithfulness: Grounding in retrieved content without hallucination
Results
Best performing model: Qwen/Qwen2.5-3B-Instruct with chunk size 500
| Question | Faithfulness Score | Correctness | Relevance | Coherence | Completeness | Faithfulness |
|----------|-------------------|-------------|-----------|-----------|--------------|--------------|
| What is the Master Theorem? | 75.0% (6/8) | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 |
| Explain Divide and Conquer algorithms | 100% (6/6) | 5/5 | 4/5 | 4/5 | 3/5 | 4/5 |
| What is the Knapsack Problem? | 85.71% (6/7) | 5/5 | 4/5 | 5/5 | 4/5 | 4/5 |
Key Findings
Hybrid retrieval (RRF) significantly improved answer quality by combining lexical and semantic matching
Small instruction-tuned models can effectively perform RAG tasks with proper prompt engineering
Structured context presentation and explicit instructions reduced hallucination
Qwen/Qwen2.5-3B-Instruct demonstrated the best performance across evaluation metrics
Repository Structure
├── notebook.ipynb          # Main implementation notebook
├── README.md              # This file
└── data/                  # Textbook data (not included)
Usage
See the notebook for detailed implementation and experimentation code.
License
This project is for educational purposes as part of the Introduction to Text Analytics course.
