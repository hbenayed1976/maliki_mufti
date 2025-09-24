import os
import streamlit as st
import time
import torch
from dotenv import load_dotenv
import asyncio
import sys
import re

# Configuration asyncio pour Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Bibliothèques pour le traitement de l'arabe
import arabic_reshaper
from bidi.algorithm import get_display

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document

# --------------------------------------------------
# CONFIGURATION INITIALE
# --------------------------------------------------
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuration PyTorch pour éviter les erreurs meta tensor
torch.set_default_dtype(torch.float32)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "cpu"  # Force l'utilisation du CPU pour éviter les problèmes MPS
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="نظام استشارة الفتاوى (PDF + TXT + Coran)",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CONFIGURATION DES EMBEDDINGS CORRIGÉE
# --------------------------------------------------
EMBEDDING_MODELS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "Sentence Paraphrase Multilangue",
        "description": "Modèle multilingue optimisé pour la similarité",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 8}
    },
    "aubmindlab/bert-base-arabertv2": {
        "name": "AraBERTv2",
        "description": "Modèle BERT spécialisé pour l'arabe",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 4}
    },
    "UBC-NLP/MARBERT": {
        "name": "MARBERT",
        "description": "Modèle BERT arabe-anglais",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 4}
    }
}

# --------------------------------------------------
# FONCTION POUR INITIALISER LES EMBEDDINGS DE FAÇON SÉCURISÉE
# --------------------------------------------------
@st.cache_resource
def load_embedding_model(model_name):
    """Charge le modèle d'embedding de façon sécurisée"""
    try:
        model_config = EMBEDDING_MODELS[model_name]
        
        # Utiliser la configuration exacte du modèle
        model_kwargs = model_config["model_kwargs"].copy()
        encode_kwargs = model_config["encode_kwargs"].copy()
        
        # Créer le modèle d'embedding avec gestion d'erreur
        with st.spinner(f"Chargement du modèle d'embedding {model_config['name']}..."):
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Test du modèle avec un texte simple
            test_text = "test"
            _ = embeddings.embed_query(test_text)
            
        st.success(f"✅ Modèle d'embedding {model_config['name']} chargé avec succès")
        return embeddings
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle d'embedding: {str(e)}")
        
        # Fallback vers un modèle plus simple avec configuration minimale
        st.warning("⚠️ Tentative de chargement d'un modèle de secours...")
        try:
            fallback_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            # Test du modèle de secours
            _ = fallback_embeddings.embed_query("test")
            st.success("✅ Modèle de secours chargé avec succès")
            return fallback_embeddings
        except Exception as fallback_error:
            st.error(f"❌ Impossible de charger même le modèle de secours: {str(fallback_error)}")
            
            # Dernier fallback - modèle très simple
            try:
                simple_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                _ = simple_embeddings.embed_query("test")
                st.success("✅ Modèle très simple chargé avec succès")
                return simple_embeddings
            except Exception as final_error:
                st.error(f"❌ Échec total: {str(final_error)}")
                return None

# --------------------------------------------------
# SUPPORT DE L'ARABE
# --------------------------------------------------
def setup_arabic_support():
    config = {
        'language': 'Arabic',
        'support_ligatures': True,
        'delete_harakat': False,
        'delete_tatweel': False,
        'shift_harakat_position': False
    }
    return arabic_reshaper.ArabicReshaper(configuration=config)

arabic_reshaped = setup_arabic_support()

def format_arabic(text):
    try:
        if not text or not isinstance(text, str):
            return ""
        return arabic_reshaped.reshape(text.strip())
    except:
        return text

# --------------------------------------------------
# EXTRACTION DES MÉTADONNÉES DU CORAN
# --------------------------------------------------
def extract_pdf_metadata(doc):
    """Extraire les métadonnées des versets coraniques avec détection améliorée"""
    metadata = {}
    content = doc.page_content
    
    # Patterns pour identifier les versets et sourates (plus larges)
    surah_pattern = r'سورة\s+([^\n]+)'
    verse_pattern = r'﴿([^﴾]+)﴾'
    ayah_number_pattern = r'﴾\s*\((\d+)\)'
    
    # Extraire le nom de la sourate
    surah_match = re.search(surah_pattern, content)
    if surah_match:
        metadata['sourate'] = surah_match.group(1).strip()
    
    # Extraire les numéros de versets
    ayah_matches = re.findall(ayah_number_pattern, content)
    if ayah_matches:
        metadata['versets'] = [int(num) for num in ayah_matches]
    
    # Détection améliorée du contenu coranique
    quran_indicators = [
        '﴿', '﴾',           # Marqueurs de début/fin de verset
        'سورة',             # Mot "sourate"
        'بسم الله',          # Basmalah
        'الحمد لله',         # Hamdallah
        'قل هو الله أحد',    # Début sourate Al-Ikhlas
        'تبارك الذي',       # Mots coraniques courants
        'يا أيها الذين آمنوا', # Appel aux croyants
        'والله أعلم',       # Formule coranique
        'رب العالمين'       # Seigneur des mondes
    ]
    
    # Marquer comme contenu coranique si au moins 2 indicateurs ou si source est tafri3.pdf
    indicators_found = sum(1 for pattern in quran_indicators if pattern in content)
    source_file = doc.metadata.get('source', '')
    
    if indicators_found >= 2 or 'tafri3.pdf' in source_file.lower():
        metadata['type'] = 'coran'
        metadata['source'] = 'tafri3.pdf'
        print(f"🔍 Segment coranique détecté - Indicateurs: {indicators_found}, Source: {source_file}")
    else:
        print(f"❓ Segment non-coranique - Indicateurs: {indicators_found}, Source: {source_file}, Contenu début: {content[:100]}")
    
    return metadata

# --------------------------------------------------
# PROMPT FIQH AMÉLIORÉ
# --------------------------------------------------
FIQH_TEMPLATE = """أنت خبير في الفقه المالكي والسيرة النبوية والقرآن الكريم.
مهمتك هي الإجابة على الأسئلة وفق الترتيب التالي، مع الإشارة دائماً إلى مصدر الإجابة:

1- إذا وجدت إجابة مباشرة وصريحة في الوثائق المقدمة، فاعرضها كما هي مع الدليل إن وُجد، وقل بوضوح: "✅ الإجابة من الوثائق".
2- إذا وجدت آيات قرآنية ذات صلة، فاذكرها مع اسم السورة ورقم الآية إن أمكن، وقل: "📖 من القرآن الكريم".
3- إذا لم تجد إجابة مباشرة، فحاول استنباط الحكم الشرعي أو الجواب بالاعتماد على مجموع ما ورد في الوثائق فقط، وقل بوضوح: "📚 الإجابة باجتهاد مبني على الوثائق".
4- إذا لم تتمكن من الاستنباط من الوثائق، فقدم إجابة بالاعتماد على معرفتك كنموذج لغوي، وقل بوضوح: "⚠️ هذه الإجابة ليست من الوثائق بل من معرفتي كنموذج لغوي".

السؤال: {question}

النصوص:
{context}

الإجابة:"""

# --------------------------------------------------
# CHUNKING POUR TXT ET PDF
# --------------------------------------------------
def custom_chunking_text(documents):
    """Découpage du TXT avec ***"""
    chunks = []
    for doc in documents:
        if "***" in doc.page_content:
            parts = re.split(r'\s*\*\*\*\s*', doc.page_content)
        else:
            parts = [doc.page_content]
        for i, part in enumerate(parts):
            if part.strip():
                chunk_doc = Document(
                    page_content=part.strip(),
                    metadata=doc.metadata.copy()
                )
                chunk_doc.metadata["segment_id"] = i
                chunks.append(chunk_doc)
    return chunks

def custom_chunking_pdf(documents):
    """Découpage du PDF avec 2 lignes vides ou plus"""
    chunks = []
    for doc in documents:
        parts = re.split(r'\n\s*\n\s*\n*', doc.page_content)
        for i, part in enumerate(parts):
            if part.strip():
                chunk_doc = Document(
                    page_content=part.strip(),
                    metadata=doc.metadata.copy()
                )
                chunk_doc.metadata["segment_id"] = i
                chunks.append(chunk_doc)
    return chunks

def custom_chunking_quran(documents):
    """Découpage spécialisé pour le Coran avec séparateurs arabes"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "۝", "؛", ".", "،", "﴾", "﴿"]  # Ajout de séparateurs arabes pertinents
    )
    chunks = splitter.split_documents(documents)
    return chunks

# --------------------------------------------------
# INITIALISATION AMÉLIORÉE
# --------------------------------------------------
@st.cache_resource
def init_system(embedding_model_name, llm_choice, google_api_key=None):
    try:
        docs_all = []

        # 1️⃣ Charger le PDF des fatwas
        if os.path.exists("fatwa-tounisia.pdf"):
            pdf_loader = PyPDFLoader("fatwa-tounisia.pdf")
            pdf_docs = pdf_loader.load()
            pdf_chunks = custom_chunking_pdf(pdf_docs)
            # Marquer les documents de fatwa
            for chunk in pdf_chunks:
                chunk.metadata["type"] = "fatwa"
                chunk.metadata["source"] = "fatwa-tounisia.pdf"
            docs_all.extend(pdf_chunks)
            st.success(f"✅ Chargement de {len(pdf_chunks)} segments de fatwas réussi")
        else:
            st.warning("⚠️ Fichier fatwa-tounisia.pdf introuvable")

        # 2️⃣ Charger le TXT
        if os.path.exists("qa.txt"):
            text_loader = TextLoader("qa.txt", encoding="utf-8")
            text_docs = text_loader.load()
            text_chunks = custom_chunking_text(text_docs)
            # Marquer les documents Q&A
            for chunk in text_chunks:
                chunk.metadata["type"] = "qa"
                chunk.metadata["source"] = "qa.txt"
            docs_all.extend(text_chunks)
            st.success(f"✅ Chargement de {len(text_chunks)} segments Q&A réussi")
        else:
            st.warning("⚠️ Fichier qa.txt introuvable")

        # 3️⃣ Charger le PDF du Coran (tafri3.pdf)
        try:
            pdf_path = "tafri3.pdf"
            if os.path.exists(pdf_path):
                pdf_loader = PyPDFLoader(pdf_path)
                pdf_docs = pdf_loader.load()
                
                # Enrichir les documents PDF avec des métadonnées
                for doc in pdf_docs:
                    # Extraire les métadonnées potentielles du contenu
                    metadata = extract_pdf_metadata(doc)
                    # Mettre à jour les métadonnées du document
                    doc.metadata.update(metadata)
                
                # Utiliser le découpage spécialisé pour le Coran
                quran_chunks = custom_chunking_quran(pdf_docs)
                docs_all.extend(quran_chunks)
                
                st.success(f"✅ Chargement de {len(quran_chunks)} segments du Coran réussi")
                print(f"Chargement de {len(pdf_docs)} pages du PDF réussi")
            else:
                st.warning(f"⚠️ Fichier PDF du Coran introuvable: {pdf_path}")
                print(f"Fichier PDF introuvable: {pdf_path}")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du PDF du Coran: {e}")
            print(f"Erreur lors du chargement du PDF: {e}")

        if not docs_all:
            st.error("❌ Aucun document n'a pu être chargé")
            return None, None

        # Afficher les statistiques de chargement avec diagnostic
        fatwa_count = len([d for d in docs_all if d.metadata.get("type") == "fatwa"])
        qa_count = len([d for d in docs_all if d.metadata.get("type") == "qa"])
        quran_count = len([d for d in docs_all if d.metadata.get("type") == "coran"])
        unknown_count = len([d for d in docs_all if d.metadata.get("type") not in ["fatwa", "qa", "coran"]])
        
        # Diagnostic détaillé
        print(f"🔍 DIAGNOSTIC DÉTAILLÉ:")
        print(f"- Segments initiaux du Coran: {len(quran_chunks) if 'quran_chunks' in locals() else 'N/A'}")
        print(f"- Segments finaux du Coran: {quran_count}")
        print(f"- Segments sans type: {unknown_count}")
        print(f"- Total calculé: {fatwa_count + qa_count + quran_count + unknown_count}")
        print(f"- Total réel: {len(docs_all)}")
        
        # Types uniques présents
        all_types = set(d.metadata.get("type", "unknown") for d in docs_all)
        print(f"- Types détectés: {all_types}")
        
        st.info(f"""
        📊 **Statistiques de chargement:**
        - 📖 Segments de fatwas: {fatwa_count}
        - ❓ Segments Q&A: {qa_count}
        - 📜 Segments du Coran: {quran_count}
        - ❓ Segments non classifiés: {unknown_count}
        - 🔢 **Total**: {len(docs_all)} segments
        
        🔍 **Vérification**: {fatwa_count + qa_count + quran_count + unknown_count} segments comptabilisés
        """)
        
        # Alert si discordance
        if len(quran_chunks if 'quran_chunks' in locals() else []) != quran_count:
            st.warning(f"⚠️ Discordance détectée : {len(quran_chunks if 'quran_chunks' in locals() else [])} segments du Coran créés initialement, mais seulement {quran_count} classifiés correctement.")

        # Charger les embeddings de façon sécurisée
        embeddings = load_embedding_model(embedding_model_name)
        if embeddings is None:
            st.error("❌ Impossible de charger le modèle d'embedding")
            return None, None

        # Créer le vectorstore avec gestion d'erreur
        try:
            with st.spinner("Création de la base de données vectorielle..."):
                vectorstore = FAISS.from_documents(docs_all, embeddings)
            st.success("✅ Base de données vectorielle créée avec succès")
        except Exception as e:
            st.error(f"❌ Erreur lors de la création du vectorstore: {str(e)}")
            return None, None

        # Choix du LLM - Fixé sur Gemini
        if not google_api_key:
            st.error("❌ GOOGLE_API_KEY manquant pour utiliser Gemini")
            return None, None
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=google_api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
        except Exception as e:
            st.warning(f"⚠️ Erreur avec gemini-2.0-flash-exp, fallback vers gemini-1.5-pro: {str(e)}")
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=google_api_key,
                    temperature=0.1,
                    convert_system_message_to_human=True
                )
            except Exception as e2:
                st.error(f"❌ Impossible de charger Gemini: {str(e2)}")
                return None, None

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={
                    "k": 15,
                    "fetch_k": 30,  # Recherche plus large avant MMR
                    "lambda_mult": 0.7  # Balance diversité/pertinence
                }
            ),
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(template=FIQH_TEMPLATE, input_variables=["context", "question"])
            },
            return_source_documents=True
        )

        return qa_chain, f"{llm_choice}"
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None, None

# --------------------------------------------------
# INTERFACE
# --------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("## ⚙️ إعدادات النظام")

        embedding_options = [f"{cfg['name']} - {cfg['description']}" for cfg in EMBEDDING_MODELS.values()]
        embedding_keys = list(EMBEDDING_MODELS.keys())
        selected_embedding_model = embedding_keys[st.selectbox("اختر نموذج التضمين:", range(len(embedding_options)), format_func=lambda i: embedding_options[i], index=0)]

        llm_choice = "gemini-2.0-flash-exp"
        
        google_api_key = st.text_input("🔑 Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

        if st.button("🔄 إعادة التهيئة"):
            st.cache_resource.clear()
            if 'qa_chain' in st.session_state:
                del st.session_state['qa_chain']
            st.session_state['embedding_model'] = selected_embedding_model
            st.session_state['llm_choice'] = llm_choice
            st.session_state['google_api_key'] = google_api_key
            st.rerun()

        # Indicateur du modèle actif
        if 'llm_choice' in st.session_state and 'embedding_model' in st.session_state:
            llm_used = st.session_state.get('llm_choice', 'N/A')
            embedding_used = st.session_state.get('embedding_model', 'N/A')
            st.markdown(f"""
            <div style='background:#f0f8ff; padding:10px; border-radius:8px; margin-top:15px;'>
            <strong>النموذج الحالي:</strong><br>
            🧠 LLM: {llm_used}<br>
            🔤 Embedding: {embedding_used}
            </div>
            """, unsafe_allow_html=True)

        # Informations sur les sources
        st.markdown("""
        ### 📚 المصادر المتاحة
        - 📖 **الفتاوى**: fatwa-tounisia.pdf
        - ❓ **أسئلة وأجوبة**: qa.txt
        - 📜 **القرآن الكريم**: tafri3.pdf
        """)

        # ✅ Copyright
        st.markdown("<p style='font-size:0.8rem; color:gray; text-align:center'>© 2025 Hassan BEN AYED</p>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center'>📚 نظام استشارة الفتاوى والقرآن الكريم</h1>", unsafe_allow_html=True)

    if 'qa_chain' not in st.session_state:
        qa_chain, model_name = init_system(selected_embedding_model, llm_choice, google_api_key)
        st.session_state['qa_chain'] = qa_chain
        st.session_state['model_name'] = model_name
        st.session_state['embedding_model'] = selected_embedding_model
        st.session_state['llm_choice'] = llm_choice

    qa_chain = st.session_state.get('qa_chain')
    if not qa_chain:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": format_arabic("السلام عليكم! كيف يمكنني مساعدتك؟ يمكنني الإجابة باستخدام الفتاوى والقرآن الكريم المتاحين.")}]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input(format_arabic("❓ اطرح سؤالك هنا")):
        st.session_state["messages"].append({"role": "user", "content": format_arabic(prompt)})
        with st.chat_message("user"):
            st.markdown(format_arabic(prompt), unsafe_allow_html=True)

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            try:
                result = qa_chain({"question": prompt, "chat_history": st.session_state["chat_history"]})
                answer = result["answer"]
                
                # Analyser les sources utilisées
                source_docs = result.get("source_documents", [])
                source_types = set()
                for doc in source_docs:
                    doc_type = doc.metadata.get("type", "unknown")
                    source_types.add(doc_type)
                
                # Ajouter information sur les sources utilisées
                if source_types:
                    sources_info = []
                    if "fatwa" in source_types:
                        sources_info.append("📖 الفتاوى")
                    if "coran" in source_types:
                        sources_info.append("📜 القرآن الكريم")
                    if "qa" in source_types:
                        sources_info.append("❓ أسئلة وأجوبة")
                    
                    if sources_info:
                        answer += f"\n\n**المصادر المستخدمة**: {' | '.join(sources_info)}"
                
                st.session_state["chat_history"].append((prompt, answer))

                # Ajouter info modèle dans la réponse
                llm_used = st.session_state.get('llm_choice', 'Unknown LLM')
                embedding_used = st.session_state.get('embedding_model', 'Unknown Embedding')
                answer += f"\n\n*({llm_used} + {embedding_used})*"

                for char in answer:
                    full_response += char
                    time.sleep(0.01)
                    response_container.markdown(format_arabic(full_response) + "▌", unsafe_allow_html=True)
                response_container.markdown(format_arabic(full_response), unsafe_allow_html=True)
            except Exception as e:
                error_message = f"⚠️ خطأ: {str(e)}"
                response_container.markdown(error_message)
                full_response = error_message

            st.session_state["messages"].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
