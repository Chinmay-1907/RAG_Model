# 🧮 Intelligent Financial RAG & Transcript Intelligence Engine

This repository contains a comprehensive **Retrieval-Augmented Generation (RAG)** and **transcript analysis** pipeline that automates the extraction of structured financial metrics from 10-K filings and earnings-call transcripts and generates AI-driven insights and Excel financial models.

---

## 📁 Data Sources (Latest)
- **SEC 10-K Filings** (PDF)  
- **Earnings-Call Transcripts** (PDF)
---

## 🛠 Tools and Frameworks
- **Programming Language**: Python   
- **Core Libraries**:  
  - `sentence-transformers` (MiniLM)  
  - `FAISS` for vector indexing  
  - `LangChain` for text splitting and document handling  
  - `groq` (LLaMA 3.3) for LLM-driven JSON extraction and analysis  
  - `PyMuPDF` / `pdfplumber` for PDF text extraction  
  - `openpyxl` for Excel report generation  
  - `pandas`, `NumPy` for data manipulation  
  - `tiktoken`, `Pydantic`, `regex` for validation and token counting

---

## 🔍 Pipeline Workflow

### 1. **PDF Ingestion & Cleaning**
- Extract raw text from 10-K and transcript PDFs  
- Clean headers, footers, and OCR artifacts  

### 2. **Intelligent Chunking & Embedding**
- Split cleaned text by financial statement headings and logical separators  
- Embed chunks using MiniLM and build a FAISS vector store  

### 3. **Hybrid Retrieval & Metric Extraction**
- Perform similarity search with section-based boosting  
- Generate targeted RAG prompts and call LLaMA to extract metrics in strict JSON  

### 4. **Excel Model Generation**
- Parse JSON metrics into Income, Balance Sheet, and Cash Flow DataFrames  
- Add retained earnings and revenue forecasts  
- Export polished three-statement models to Excel with professional formatting  

### 5. **Earnings-Call Transcript Analysis**
- Convert transcripts to text, clean and segment into sections  
- Detect speakers and Q&A structure  
- Use LLaMA to produce:  
  - Concise summaries  
  - Financial highlights  
  - Sentiment scores  
  - Key quote extraction  
  - Risks & opportunities report  

---

## 📊 Results & Impact
- **90%+ reduction** in manual data preparation and model-building time  
- **Structured, JSON-driven** metric extraction ensures consistency and auditability  
- **AI-driven summaries** empower rapid decision-making for analysts and executives  

---
