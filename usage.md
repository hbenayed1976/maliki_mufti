🚀Usage
streamlit run rag_gemini_embeddings.py


Then open: http://localhost:8501

🔑 API Key

You need a Google API Key for Gemini.
Create .env and add:

GOOGLE_API_KEY=your_api_key_here

📜 License

GPL v3.0 © 2025 Hassan BEN AYED


---

### 📄 `USAGE.md`
```markdown
# 📖 User Guide

## Step 1: Prepare datasets
- Place `fatwa-tounisia.pdf`, `qa.txt`, and `tafri3.pdf` in the project folder.  

---

## Step 2: Set up environment
- Install dependencies:
  ```bash
  pip install -r requirements.txt


Add your Google API Key in .env.

Step 3: Run the application
streamlit run rag_only_gemini_embeddings_pdf_txt.py

Step 4: Interact

Open http://localhost:8501
.

Ask questions in Arabic.

Answers will include sources:

📖 Fatwa

📜 Qur’an

❓ Q&A

If no reference is found, a language model fallback will be used.


---
