import math
import os
import re
import time
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# WebBaseLoader reads this at import time; without it many sites reject the request
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; EquityBot/1.0)")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
NUM_URLS = 3
EMBED_BATCH_SIZE = 50
MAX_EMBED_RETRIES = 5
RETRIEVER_K = 4  # chunks per sub-question
RETRIEVER_FETCH_K = 20
MAX_SUB_QUESTIONS = 4

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Split the user's question into the separate, self-contained sub-questions it asks, "
     "one per line, with no numbering or extra text. If it asks only one thing, return "
     "it unchanged on a single line."),
    ("human", "{question}"),
])

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial research assistant. Answer the question using only the "
     "article excerpts below. If the excerpts do not contain the answer, say you "
     "don't know. Be concise and factual.\n\n{context}"),
    ("human", "{question}"),
])

# ======================
# CHECK API KEY
# ======================
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ======================
# CACHED MODELS
# ======================
# Shared across sessions and reruns so they're built once per server process
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def build_vectorstore(docs, embeddings, status):
    # The Gemini free tier allows ~100 embedding requests per minute, so embed in
    # batches and wait out any rate-limit error instead of failing the whole run
    store = None
    for start in range(0, len(docs), EMBED_BATCH_SIZE):
        batch = docs[start:start + EMBED_BATCH_SIZE]
        status.text(f"Embedding Vector Building... ({start}/{len(docs)} chunks)")
        for attempt in range(MAX_EMBED_RETRIES):
            try:
                if store is None:
                    store = FAISS.from_documents(batch, embeddings)
                else:
                    store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" not in str(e) or attempt == MAX_EMBED_RETRIES - 1:
                    raise
                match = re.search(r"retry in ([\d.]+)s", str(e))
                wait = math.ceil(float(match.group(1))) + 1 if match else 60
                for remaining in range(wait, 0, -1):
                    status.text(f"Gemini rate limit reached, resuming in {remaining}s... "
                                f"({start}/{len(docs)} chunks)")
                    time.sleep(1)
    return store


def retrieve(vectorstore, question):
    # A compound question ("who founded X, and who owns Y?") embeds close to only one
    # of its topics, so retrieve for each sub-question separately and merge the results
    decompose = DECOMPOSE_PROMPT | get_llm() | StrOutputParser()
    sub_questions = [q.strip() for q in decompose.invoke({"question": question}).splitlines() if q.strip()]
    sub_questions = sub_questions[:MAX_SUB_QUESTIONS] or [question]

    # MMR keeps the chunks for each sub-question from being near-duplicates
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K, "lambda_mult": 0.5},
    )
    seen, docs = set(), []
    for sub_question in sub_questions:
        for doc in retriever.invoke(sub_question):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
    return sub_questions, docs


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# ======================
# UI
# ======================
st.title("EquityBot: AI Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(NUM_URLS)]
process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# ======================
# PROCESS URLS
# ======================
if process_url_clicked:

    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        st.error("Please enter at least one valid URL")
        st.stop()

    invalid = [u for u in urls if not is_valid_url(u)]
    if invalid:
        st.error("Invalid URL(s), must start with http:// or https://: " + ", ".join(invalid))
        st.stop()

    # Each browser session keeps its own index, so users never see each other's articles
    st.session_state.pop("vectorstore", None)

    main_placeholder.text("Data Loading...Started...")

    try:
        loader = WebBaseLoader(web_paths=urls)
        data = loader.load()
    except Exception as e:
        main_placeholder.empty()
        st.error(f"Error loading URLs: {str(e)}")
        st.stop()

    # ======================
    # VALIDATION
    # ======================
    if not data:
        main_placeholder.empty()
        st.error("No data loaded. The site may be blocking automated requests.")
        st.stop()

    # ======================
    # SPLITTING
    # ======================
    main_placeholder.text("Text Splitter...Started...")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    # remove empty docs
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    if len(docs) == 0:
        main_placeholder.empty()
        st.error("No valid content after splitting.")
        st.stop()

    # ======================
    # EMBEDDINGS + FAISS
    # ======================
    try:
        st.session_state["vectorstore"] = build_vectorstore(docs, get_embeddings(), main_placeholder)
    except Exception as e:
        main_placeholder.empty()
        st.error(f"Embedding/FAISS error: {str(e)}")
        st.stop()

    main_placeholder.empty()
    st.success(f"Processing complete! Indexed {len(docs)} chunks from {len(data)} article(s).")

# ======================
# QUERY
# ======================
query = st.text_input("Question:")

if query:

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.error("Please process URLs first")
        st.stop()

    with st.spinner("Generating answer..."):
        try:
            sub_questions, source_docs = retrieve(vectorstore, query)
            chain = PROMPT | get_llm() | StrOutputParser()
            answer = chain.invoke({"context": format_docs(source_docs), "question": query})
        except Exception as e:
            st.error(f"Model error: {str(e)}")
            st.stop()

    st.header("Answer")
    st.write(answer or "No answer found")

    # Sources come from the retrieved chunks, not from parsing model output
    sources = list(dict.fromkeys(doc.metadata.get("source") for doc in source_docs if doc.metadata.get("source")))
    if sources:
        st.subheader("Sources:")
        for source in sources:
            st.write(source)

    with st.expander(f"Retrieved passages ({len(source_docs)})"):
        if len(sub_questions) > 1:
            st.caption("Searched for: " + " · ".join(sub_questions))
        for i, doc in enumerate(source_docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source', 'unknown')}**")
            st.text(doc.page_content)
