import re
from pathlib import Path

import streamlit as st
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hospitality Information Retrieval Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HOSPITALITY FRONTEND STYLING
# ============================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f8f5ef 0%, #fffaf3 48%, #f4efe7 100%); }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {background:transparent;}
.block-container { max-width:1250px; padding-top:2rem; padding-bottom:3rem; }
.hero { padding:2.2rem 2.4rem; border-radius:24px; background:linear-gradient(135deg,#183b56 0%,#255b76 55%,#4d7c8f 100%); color:white; box-shadow:0 18px 45px rgba(24,59,86,.18); margin-bottom:1.4rem; position:relative; overflow:hidden; }
.hero:after { content:"🏨"; position:absolute; right:35px; bottom:8px; font-size:7rem; opacity:.12; }
.hero-kicker { font-size:.88rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; opacity:.85; margin-bottom:.55rem; }
.hero-title { font-size:2.65rem; font-weight:800; line-height:1.05; margin:0; }
.hero-subtitle { font-size:1.08rem; opacity:.92; margin-top:.8rem; max-width:760px; line-height:1.55; }
.pill { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:.35rem .75rem; font-size:.78rem; margin-right:.35rem; margin-top:.35rem; }
.feature-card { background:rgba(255,255,255,.88); border:1px solid rgba(24,59,86,.09); border-radius:18px; padding:1.1rem 1.15rem; min-height:135px; box-shadow:0 8px 25px rgba(0,0,0,.05); }
.feature-icon { font-size:1.65rem; } .feature-title { font-weight:750; font-size:1rem; margin-top:.45rem; }
.feature-text { color:#5b6670; font-size:.88rem; line-height:1.45; margin-top:.25rem; }
.section-label { color:#183b56; font-size:1.35rem; font-weight:800; margin:1.3rem 0 .65rem 0; }
.section-caption { color:#68747d; margin-bottom:.9rem; }
div[data-testid="stTextInput"] input { border-radius:14px!important; border:1px solid #cfd9df!important; padding:.8rem 1rem!important; background:rgba(255,255,255,.95)!important; }
div[data-testid="stTextInput"] input:focus { border-color:#255b76!important; box-shadow:0 0 0 2px rgba(37,91,118,.12)!important; }
.stButton > button { border-radius:12px!important; font-weight:700!important; min-height:2.65rem; }
.result-card { background:white; border-left:5px solid #255b76; border-radius:15px; padding:1rem 1.15rem; margin:.6rem 0 .9rem 0; box-shadow:0 6px 18px rgba(0,0,0,.045); }
.project-footer { margin-top:2rem; padding:1rem 1.2rem; background:#183b56; color:white; border-radius:15px; text-align:center; font-size:.84rem; }
@media (max-width:800px) {.hero-title{font-size:2rem}.hero{padding:1.5rem}}
</style>
""", unsafe_allow_html=True)



# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Smart Hospitality Knowledge System</div>
  <div class="hero-title">🏨 Hospitality Information Retrieval Assistant</div>
  <div class="hero-subtitle">A RAG-powered hotel information portal that retrieves relevant information from hotel documents when guests ask questions.</div>
  <div><span class="pill">RAG</span><span class="pill">ChromaDB</span><span class="pill">Semantic Search</span><span class="pill">Hotel Knowledge</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-caption"><b>Domain:</b> Hospitality / Hotel Management &nbsp; • &nbsp; <b>Objective:</b> Retrieve relevant hotel information from uploaded documents based on guest queries.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-label">✨ What this system can help with</div>
<div class="section-caption">Upload hotel documents, then ask natural-language questions.</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
  <div class="feature-card"><div class="feature-icon">🛏️</div><div class="feature-title">Rooms</div><div class="feature-text">Room types, facilities, pricing notes and guest amenities.</div></div>
  <div class="feature-card"><div class="feature-icon">🍽️</div><div class="feature-title">Dining</div><div class="feature-text">Restaurant services, meal timings and food-related information.</div></div>
  <div class="feature-card"><div class="feature-icon">🕐</div><div class="feature-title">Policies</div><div class="feature-text">Check-in, check-out, cancellation and guest policies.</div></div>
  <div class="feature-card"><div class="feature-icon">🏊</div><div class="feature-title">Facilities</div><div class="feature-text">Wi-Fi, parking, pool, room service and other amenities.</div></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "rag_database"

# Create database folder automatically
DATABASE_PATH.mkdir(
    parents=True,
    exist_ok=True
)

COLLECTION_NAME = "hospitality_document_store"


# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "pages" not in st.session_state:
    st.session_state.pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

if "stored_count" not in st.session_state:
    st.session_state.stored_count = 0

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "retrieved_documents" not in st.session_state:
    st.session_state.retrieved_documents = []

if "retrieved_metadatas" not in st.session_state:
    st.session_state.retrieved_metadatas = []

if "retrieved_distances" not in st.session_state:
    st.session_state.retrieved_distances = []

if "question_embedding" not in st.session_state:
    st.session_state.question_embedding = []


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# ============================================================
# GET CHROMADB COLLECTION
# ============================================================

def get_collection():

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return client, collection


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(
        uploaded_file
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text is None:
            text = ""

        text = text.strip()

        pages.append(
            {
                "page": page_number,
                "text": text
            }
        )

    return pages


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(uploaded_file):

    text = uploaded_file.getvalue().decode(
        "utf-8",
        errors="replace"
    )

    return [
        {
            "page": 1,
            "text": text.strip()
        }
    ]


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=500,
    overlap=100
):
    """
    Split document text into overlapping chunks.

    chunk_size = maximum characters in a chunk
    overlap    = characters shared with next chunk
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        if not text:
            continue

        # Normalize whitespace
        text = " ".join(
            text.split()
        )

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            # Stop at last chunk
            if end >= len(text):
                break

            # Keep overlap
            start = end - overlap

    return chunks


# ============================================================
# DELETE OLD RECORDS
# ============================================================

def clear_old_records(collection):

    try:

        old_data = collection.get(
            include=[]
        )

        old_ids = old_data.get(
            "ids",
            []
        )

        if old_ids:

            collection.delete(
                ids=old_ids
            )

    except Exception as e:

        st.warning(
            f"Could not clear old records: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:0.6rem 0 1rem 0;">
      <div style="font-size:2.3rem;">🏨</div>
      <div style="font-size:1.25rem;font-weight:800;color:#183b56;">Hotel Knowledge Hub</div>
      <div style="font-size:.82rem;color:#6c757d;">Hospitality RAG Demonstration</div>
    </div>
    """, unsafe_allow_html=True)

    st.header("🏨 Hospitality Project")

    st.write("Domain")
    st.code("Hospitality / Hotel Management")

    st.write("Project Objective")
    st.caption(
        "Retrieve relevant information from hotel documents based on guest questions."
    )

    st.header("⚙️ Configuration")

    st.write("Embedding Model")

    st.code(
        "all-MiniLM-L6-v2"
    )

    st.write("Vector Database")

    st.code(
        "ChromaDB"
    )

    st.write("Hospitality Collection")

    st.code(
        COLLECTION_NAME
    )

    st.write("Chunk Size")

    st.code(
        "500 characters"
    )

    st.write("Chunk Overlap")

    st.code(
        "100 characters"
    )

    st.write("Hotel Database Location")

    st.code(
        str(DATABASE_PATH)
    )

    st.divider()

    if st.session_state.processed:

        st.success(
            "Hotel Document Ready"
        )

        st.write(
            f"Stored Chunks: "
            f"{st.session_state.stored_count}"
        )

    else:

        st.info(
            "No hotel document processed yet."
        )


# ============================================================
# HOSPITALITY PROJECT SCOPE
# ============================================================

st.subheader("🏨 Hospitality Information Covered")

st.write(
    "This system can retrieve information about hotel rooms, facilities, "
    "restaurant services, check-in/check-out timings, Wi-Fi, parking, "
    "room service, cancellation rules, and other information contained "
    "in the uploaded hotel documents."
)


# ============================================================
# STEP 1 — UPLOAD DOCUMENT
# ============================================================

st.header(
    "1️⃣ Upload Hotel Document"
)

uploaded_file = st.file_uploader(
    "Upload a hotel PDF or TXT document",
    type=["pdf", "txt"]
)


# ============================================================
# PROCESS BUTTON
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Selected File: {uploaded_file.name}"
    )

    if st.button(
        "🏨 Process Hotel Document",
        type="primary"
    ):

        # ====================================================
        # STEP 2 — EXTRACT TEXT
        # ====================================================

        with st.spinner(
            "Extracting text from document..."
        ):

            try:

                if uploaded_file.name.lower().endswith(
                    ".pdf"
                ):

                    pages = extract_pdf(
                        uploaded_file
                    )

                else:

                    pages = extract_txt(
                        uploaded_file
                    )

            except Exception as e:

                st.error(
                    "Error reading the document."
                )

                st.code(
                    str(e)
                )

                st.stop()


        # ----------------------------------------------------
        # Check extracted text
        # ----------------------------------------------------

        total_characters = sum(
            len(page["text"])
            for page in pages
        )

        if total_characters == 0:

            st.error(
                "No text could be extracted."
            )

            st.warning(
                "This may be a scanned/image PDF. "
                "OCR would be required for scanned PDFs."
            )

            st.stop()


        # ====================================================
        # STEP 3 — CREATE CHUNKS
        # ====================================================

        with st.spinner(
            "Splitting document into chunks..."
        ):

            chunks = create_chunks(
                pages,
                chunk_size=500,
                overlap=100
            )

        if not chunks:

            st.error(
                "No chunks were created."
            )

            st.stop()


        # ====================================================
        # STEP 4 — CREATE EMBEDDINGS
        # ====================================================

        with st.spinner(
            "Generating local embeddings..."
        ):

            model = load_embedding_model()

            texts = [
                item["text"]
                for item in chunks
            ]

            embedding_array = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Convert NumPy array into Python list
            embeddings = embedding_array.tolist()


        # ====================================================
        # STEP 5 — CHROMADB
        # ====================================================

        with st.spinner(
            "Saving document into ChromaDB..."
        ):

            try:

                client, collection = get_collection()

                # Clear only old records.
                # DO NOT delete the collection.

                clear_old_records(
                    collection
                )

                # Create IDs

                ids = [
                    f"chunk_{i}"
                    for i in range(
                        len(chunks)
                    )
                ]

                # Metadata

                metadatas = [
                    {
                        "source": uploaded_file.name,
                        "page": item["page"]
                    }
                    for item in chunks
                ]

                # Store

                collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                stored_count = collection.count()

            except Exception as e:

                st.error(
                    "ChromaDB storage failed."
                )

                st.code(
                    str(e)
                )

                st.stop()


        # ====================================================
        # SAVE INTO SESSION STATE
        # ====================================================

        st.session_state.processed = True

        st.session_state.pages = pages

        st.session_state.chunks = chunks

        st.session_state.embeddings = embeddings

        st.session_state.stored_count = stored_count

        st.session_state.file_name = (
            uploaded_file.name
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "🎉 Hospitality Document Store Created Successfully!"
        )

        st.info(
            f"Created {len(chunks)} chunks "
            f"and stored {stored_count} records."
        )


# ============================================================
# DISPLAY DOCUMENT STORE RESULTS
# ============================================================

if st.session_state.processed:

    pages = st.session_state.pages

    chunks = st.session_state.chunks

    embeddings = st.session_state.embeddings

    stored_count = st.session_state.stored_count


    # ========================================================
    # STEP 2 — EXTRACTED TEXT
    # ========================================================

    st.header(
        "2️⃣ Extracted Hotel Information"
    )

    total_pages = len(
        pages
    )

    total_characters = sum(
        len(page["text"])
        for page in pages
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            total_pages
        )

    with col2:

        st.metric(
            "Characters",
            f"{total_characters:,}"
        )

    with col3:

        st.metric(
            "File",
            st.session_state.file_name
        )


    for page in pages:

        with st.expander(
            f"📄 Page {page['page']}"
        ):

            st.text_area(
                f"Extracted Text - Page {page['page']}",
                page["text"],
                height=220,
                key=f"extracted_page_{page['page']}"
            )


    # ========================================================
    # STEP 3 — CHUNKS
    # ========================================================

    st.header(
        "3️⃣ Hotel Document Chunks"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Chunks",
            len(chunks)
        )

    with col2:

        st.metric(
            "Chunk Size",
            "500"
        )

    with col3:

        st.metric(
            "Overlap",
            "100"
        )


    st.write(
        "The original document has been divided "
        "into smaller pieces."
    )


    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        with st.expander(
            f"🔹 Chunk {i} — Page {chunk['page']}"
        ):

            st.write(
                chunk["text"]
            )

            st.caption(
                f"Characters: "
                f"{len(chunk['text'])}"
            )


    # ========================================================
    # STEP 4 — EMBEDDINGS
    # ========================================================

    st.header(
        "4️⃣ Hotel Information Embeddings"
    )

    embedding_dimension = (
        len(embeddings[0])
        if embeddings
        else 0
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Embeddings",
            len(embeddings)
        )

    with col2:

        st.metric(
            "Vector Dimension",
            embedding_dimension
        )

    with col3:

        st.metric(
            "Model",
            "all-MiniLM-L6-v2"
        )


    st.write(
        "Every document chunk has been converted "
        "into a numerical vector."
    )


    # Show embeddings

    for i, vector in enumerate(
        embeddings,
        start=1
    ):

        with st.expander(
            f"🧠 Embedding {i} — Chunk {i}"
        ):

            st.write(
                f"Vector dimension: "
                f"{len(vector)}"
            )

            st.code(
                str(vector)
            )


    # ========================================================
    # STEP 5 — CHROMADB
    # ========================================================

    st.header(
        "5️⃣ Hospitality ChromaDB Storage"
    )


    try:

        client, collection = get_collection()

        current_count = collection.count()

    except Exception as e:

        st.error(
            "Could not connect to ChromaDB."
        )

        st.code(
            str(e)
        )

        st.stop()


    st.success(
        "✅ ChromaDB is connected."
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Stored Records",
            current_count
        )

    with col2:

        st.metric(
            "Hospitality Collection",
            COLLECTION_NAME
        )

    with col3:

        st.metric(
            "Storage Type",
            "Persistent"
        )


    st.write(
        "Database Location:"
    )

    st.code(
        str(DATABASE_PATH)
    )


    # ========================================================
    # STEP 6 — VERIFY CHROMADB
    # ========================================================

    st.header(
        "6️⃣ Verify Stored Hotel Data"
    )


    try:

        stored_data = collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

    except Exception as e:

        st.error(
            "Could not read ChromaDB."
        )

        st.code(
            str(e)
        )

        st.stop()


    stored_ids = stored_data.get(
        "ids",
        []
    )

    stored_documents = stored_data.get(
        "documents",
        []
    )

    stored_metadatas = stored_data.get(
        "metadatas",
        []
    )


    st.write(
        f"Total records found: "
        f"**{len(stored_ids)}**"
    )


    for i, record_id in enumerate(
        stored_ids
    ):

        with st.expander(
            f"💾 Stored Record {i + 1}"
        ):

            st.write(
                "Record ID:"
            )

            st.code(
                str(record_id)
            )


            st.write(
                "Document Chunk:"
            )

            if i < len(
                stored_documents
            ):

                st.info(
                    stored_documents[i]
                )


            st.write(
                "Metadata:"
            )

            if i < len(
                stored_metadatas
            ):

                st.json(
                    stored_metadatas[i]
                )


            st.write(
                "Embedding:"
            )

            # Use the embedding we already generated.
            # It is a normal Python list, so there is
            # no NumPy truth-value error.

            if i < len(
                embeddings
            ):

                st.code(
                    str(
                        embeddings[i]
                    )
                )


    # ========================================================
    # STEP 7 — RETRIEVAL
    # ========================================================

    st.divider()

    st.header(
        "7️⃣ Retrieve Hospitality Information"
    )

    st.write(
        "Ask a question about the hotel, its facilities, services, policies, or guest information."
    )


    st.write("### 💬 Example Hospitality Questions")
    st.write(
        "• What time is check-in?  • What facilities are available?  "
        "• Is Wi-Fi available?  • What are the restaurant timings?  "
        "• What is the cancellation policy?"
    )

    question = st.text_input(
        "💬 Ask about the hotel",
        placeholder=(
            "Example: What time is hotel check-in?"
        ),
        key="question_input"
    )


    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3
    )


    if st.button(
        "🔍 Retrieve Hotel Information",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()


        # ====================================================
        # STEP 8 — QUESTION EMBEDDING
        # ====================================================

        st.subheader(
            "8️⃣ Convert Question into Embedding"
        )


        with st.spinner(
            "Generating question embedding..."
        ):

            model = load_embedding_model()

            question_embedding = model.encode(
                question,
                normalize_embeddings=True
            ).tolist()


        st.session_state.last_question = (
            question
        )

        st.session_state.question_embedding = (
            question_embedding
        )


        st.success(
            "✅ Question converted into embedding."
        )


        st.write(
            f"Question vector dimension: "
            f"**{len(question_embedding)}**"
        )


        with st.expander(
            "🧠 View Question Embedding"
        ):

            st.code(
                str(question_embedding)
            )


        # ====================================================
        # STEP 9 — CHROMADB SEARCH
        # ====================================================

        st.subheader(
            "9️⃣ Similarity Search in ChromaDB"
        )


        with st.spinner(
            "Searching ChromaDB..."
        ):

            try:

                client, collection = get_collection()

                total_records = collection.count()

                number_to_retrieve = min(
                    top_k,
                    total_records
                )

                results = collection.query(
                    query_embeddings=[
                        question_embedding
                    ],
                    n_results=number_to_retrieve,
                    include=[
                        "documents",
                        "metadatas",
                        "distances"
                    ]
                )

            except Exception as e:

                st.error(
                    "ChromaDB retrieval failed."
                )

                st.code(
                    str(e)
                )

                st.stop()


        # ====================================================
        # GET RESULTS
        # ====================================================

        retrieved_documents = (
            results.get(
                "documents",
                [[]]
            )[0]
        )

        retrieved_metadatas = (
            results.get(
                "metadatas",
                [[]]
            )[0]
        )

        retrieved_distances = (
            results.get(
                "distances",
                [[]]
            )[0]
        )


        st.session_state.retrieved_documents = (
            retrieved_documents
        )

        st.session_state.retrieved_metadatas = (
            retrieved_metadatas
        )

        st.session_state.retrieved_distances = (
            retrieved_distances
        )


        # ====================================================
        # STEP 10 — RETRIEVED INFORMATION
        # ====================================================

        st.subheader(
            "🔟 Retrieved Information"
        )


        if not retrieved_documents:

            st.warning(
                "No matching hotel information was retrieved."
            )

        else:

            st.success(
                f"✅ Retrieved "
                f"{len(retrieved_documents)} "
                f"relevant chunks."
            )


            for i in range(
                len(retrieved_documents)
            ):

                st.markdown(
                    f"### Result {i + 1}"
                )


                # --------------------------------------------
                # Retrieved text
                # --------------------------------------------

                st.info(
                    retrieved_documents[i]
                )


                # --------------------------------------------
                # Similarity distance
                # --------------------------------------------

                if i < len(
                    retrieved_distances
                ):

                    distance = (
                        retrieved_distances[i]
                    )

                    # Since the collection uses
                    # cosine distance:
                    #
                    # similarity ≈ 1 - distance

                    similarity = max(
                        0.0,
                        1.0 - distance
                    )

                    st.write(
                        "📏 Vector Distance:"
                    )

                    st.code(
                        f"{distance:.4f}"
                    )

                    st.write(
                        "📊 Approx. Cosine Similarity:"
                    )

                    st.code(
                        f"{similarity:.4f}"
                    )


                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                if i < len(
                    retrieved_metadatas
                ):

                    st.write(
                        "📌 Metadata:"
                    )

                    st.json(
                        retrieved_metadatas[i]
                    )


                st.divider()


        # ====================================================
        # STEP 11 — RETRIEVAL FLOW
        # ====================================================

        st.subheader(
            "🔄 Hospitality Retrieval Flow"
        )

        st.code(
            f"""
User Question
      ↓
"{question}"
      ↓
Sentence Transformer
      ↓
Question Embedding
      ↓
ChromaDB
      ↓
Similarity Search
      ↓
Top {len(retrieved_documents)} Relevant Chunks
""",
            language="text"
        )


        # ====================================================
        # EXPLANATION
        # ====================================================

        st.subheader(
            "💡 How the Hospitality RAG System Works"
        )

        st.write(
            "1. The question was converted into an embedding."
        )

        st.write(
            "2. ChromaDB compared the question vector "
            "with the stored document vectors."
        )

        st.write(
            "3. The closest vectors were retrieved."
        )

        st.write(
            "4. The corresponding text chunks were "
            "returned as the relevant information."
        )


    # ========================================================
    # COMPLETE RAG ARCHITECTURE
    # ========================================================

    st.divider()

    st.header(
        "🧠 Complete Hospitality RAG Architecture"
    )

    st.code(
        """
                DOCUMENT STORE
                ────────────────

              PDF / TXT
                  ↓
             Text Extraction
                  ↓
               Chunking
                  ↓
        Sentence Transformers
                  ↓
             Embeddings
                  ↓
              ChromaDB
                  │
                  │
                  ▼
              RETRIEVAL
              ─────────

            User Question
                  ↓
          Question Embedding
                  ↓
             ChromaDB
                  ↓
          Similarity Search
                  ↓
        Relevant Document Chunks
""",
        language="text"
    )


    st.markdown("""
<div class="project-footer">🏨 Hospitality Information Retrieval Assistant &nbsp; | &nbsp; Python + Streamlit + Sentence Transformers + ChromaDB</div>
""", unsafe_allow_html=True)


else:

    st.info(
        "👆 Upload a hotel PDF or TXT file to begin."
    )