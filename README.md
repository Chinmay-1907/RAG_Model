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
  - `PyMuPDF`, `pdf2image`, `pytesseract`, `Pillow` for OCR  
  - `openpyxl` for Excel model generation  
  - `pandas`, `NumPy` for data manipulation  
  - `tiktoken`, `Pydantic`, `regex` for validation and token counting

---


## 🔍 End-to-End Pipeline

### 1. **PDF/Text Ingestion & Cleaning**
- Supports both PDF (with OCR fallback) and `.txt` text files  
- Cleans headers, footers, page numbers, and OCR noise  

### 2. **Smart Chunking & Embedding**
- Splits by financial statement headers and logical markers  
- Embeds sections using MiniLM and creates FAISS index  

### 3. **Hybrid Semantic Retrieval**
- Retrieves top sections with section-based boosting  
- Optimizes for relevant financial data context  

### 4. **RAG Prompt & JSON Extraction**
- Builds structured, token-limited RAG prompts  
- Uses LLaMA to extract metrics as strict JSON  
- Robust validation and fallback parsing  

### 5. **Excel Model Generation**
- Converts JSON into Income, Balance, and Cash Flow DataFrames  
- Adds **Retained Earnings** and **Revenue Forecasting**  
- Outputs well-formatted `.xlsx` financial model  

### 6. **Transcript Intelligence Engine**
- Segments transcript into key sections  
- Extracts speakers and Q&A structure  
- Generates:
  - Executive Summary  
  - Financial Highlights  
  - Sentiment Score (1–10)  
  - Key Quotes  
  - Risks & Opportunities  

---


## 📊 Results & Benefits
- **80–90% automation** of analyst prep from raw filings  
- **Traceable JSON** improves audit and validation workflows  
- **Instant Excel reports** for Income, Balance, Cash Flow  
- **Transcript analysis** gives analysts key highlights in seconds  

---
