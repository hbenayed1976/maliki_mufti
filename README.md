# 🕌 Fatwa & Qur’an Consultation System (RAG + Gemini)

This project is a **Streamlit-based application** designed to provide answers to user questions using:
- 📖 Fatwas (`fatwa-tounisia.pdf`)
- ❓ Q&A dataset (`qa.txt`)
- 📜 Qur’an (`tafri3.pdf`)

It leverages **Retrieval-Augmented Generation (RAG)** with:
- **Embeddings:** AraBERT, MARBERT, multilingual MiniLM
- **LLM:** Google Gemini API
- **Vector Database:** FAISS
- **Interface:** Streamlit Chat UI

---

## ✨ Features
- 🔎 **Context-aware Q&A** with Fatwa, Qur’an, and Q&A datasets
- 📜 **Qur’an verse detection** (with surah & ayah metadata)
- 🧠 **Multiple embedding models** (AraBERT, MARBERT, MiniLM)
- 🌐 **Arabic text reshaping** for proper display
- ⚡ **Gemini LLM integration** (`gemini-2.0-flash-exp` with fallback)
- 📚 **Source attribution** (Fatwa / Qur’an / Q&A) in every answer
- 💬 **Interactive chat interface** with persistent history

---

## 📂 Repository Structure
├── rag_only_gemini_embeddings_pdf_txt.py # Main Streamlit application

├── fatwa-tounisia.pdf # 

├── qa.txt # Q&A dataset 

├── tafri3.pdf # Qur’an dataset 

├── README.md # Project documentation
