import math
import os
import re
import time
from datetime import date
from urllib.parse import urlparse

import altair as alt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# WebBaseLoader reads this at import time; without it many sites reject the request
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; EquityBot/1.0)")

import yfinance as yf
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
NUM_URLS = 3
NUM_NEWS_ARTICLES = 3  # kept small: every chunk costs a free-tier embedding request
EMBED_BATCH_SIZE = 50
MAX_EMBED_RETRIES = 5
RETRIEVER_K = 4  # chunks per sub-question
RETRIEVER_FETCH_K = 20
MAX_SUB_QUESTIONS = 4
HISTORY_TURNS = 3  # previous question/answer pairs given to the model
AVATARS = {"user": ":material/person:", "assistant": ":material/query_stats:"}

ACCENT = "#3FB68B"  # keep in sync with primaryColor in .streamlit/config.toml

st.set_page_config(page_title="EquityBot", page_icon=":material/query_stats:")
# Streamlit's default metric font truncates values in four narrow columns
st.markdown("<style>[data-testid='stMetricValue'] {font-size: 1.5rem;}</style>", unsafe_allow_html=True)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Using the conversation so far to resolve references like \"it\" or \"that company\", "
     "rewrite the user's latest question as the separate, self-contained sub-questions it "
     "asks, one per line, with no numbering or extra text. If it asks only one thing, "
     "return a single line.\n\nConversation so far:\n{history}"),
    ("human", "{question}"),
])

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial research assistant. Answer the question using only the "
     "excerpts below, which come from market data and news articles; use the "
     "conversation only to understand what the question refers to. If the excerpts do "
     "not contain the answer, say you don't know. Be concise and factual, and give "
     "figures with their date or period when the excerpt states one.\n\n"
     "Conversation so far:\n{history}\n\nExcerpts:\n{context}"),
    ("human", "{question}"),
])

# ======================
# CHECK API KEY
# ======================
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it to a .env file in the project root.")
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


# ======================
# MARKET DATA
# ======================
def short_company_name(name):
    # "NVIDIA Corporation" -> "NVIDIA", "Apple Inc." -> "Apple"
    return re.sub(r"[,\s]+(Corporation|Corp\.?|Incorporated|Inc\.?|Limited|Ltd\.?|plc|PLC|Co\.?|Company|Holdings?|Group|N\.V\.|S\.A\.|AG|SE)$",
                  "", name).strip() or name


@st.cache_data(ttl=600, show_spinner=False)
def get_snapshot(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="6mo")
    if history.empty:
        return None
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    close = history["Close"]
    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "short_name": short_company_name(info.get("shortName") or ticker),
        "currency": info.get("currency", ""),
        "price": float(close.iloc[-1]),
        "previous_close": float(close.iloc[-2]) if len(close) > 1 else None,
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "year_low": info.get("fiftyTwoWeekLow"),
        "year_high": info.get("fiftyTwoWeekHigh"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": info.get("longBusinessSummary"),
        "history": close.rename("Close"),
    }


@st.cache_data(ttl=600, show_spinner=False)
def get_news_links(ticker, company_word):
    # Yahoo tags many loosely related stories with a ticker, so prefer headlines
    # that actually name the company
    try:
        news = yf.Search(ticker, news_count=20).news
    except Exception:
        return []
    related = [n for n in news if ticker in (n.get("relatedTickers") or []) and n.get("link")]
    related.sort(key=lambda n: company_word.lower() not in n.get("title", "").lower())
    return list(dict.fromkeys(n["link"] for n in related))[:NUM_NEWS_ARTICLES]


def format_big(value):
    if value is None:
        return "n/a"
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= divisor:
            return f"{value / divisor:,.2f}{suffix}"
    return f"{value:,.0f}"


def format_ratio(value):
    return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"


def snapshot_document(snap):
    # Market data goes into the index as a document so the chat can answer
    # valuation questions alongside the news
    lines = [
        f"Market data for {snap['name']} ({snap['ticker']}) from Yahoo Finance, as of {date.today():%d %B %Y}.",
        f"Share price: {snap['price']:,.2f} {snap['currency']}.",
        f"Market capitalisation: {format_big(snap['market_cap'])} {snap['currency']}.",
        f"Trailing P/E: {format_ratio(snap['trailing_pe'])}. Forward P/E: {format_ratio(snap['forward_pe'])}.",
    ]
    if snap["year_low"] and snap["year_high"]:
        lines.append(f"52-week range: {snap['year_low']:,.2f} to {snap['year_high']:,.2f} {snap['currency']}.")
    if snap["sector"]:
        lines.append(f"Sector: {snap['sector']}. Industry: {snap['industry']}.")
    if snap["summary"]:
        lines.append(f"Business description: {snap['summary']}")
    return Document(
        page_content="\n".join(lines),
        metadata={"source": f"https://finance.yahoo.com/quote/{snap['ticker']}",
                  "title": f"{snap['ticker']} market data (Yahoo Finance)"},
    )


# ======================
# INDEXING
# ======================
def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def load_articles(urls):
    # Load one at a time so a single blocked site doesn't sink the whole batch
    docs, failed = [], []
    for url in urls:
        try:
            docs.extend(d for d in WebBaseLoader(web_paths=[url]).load() if d.page_content.strip())
        except Exception:
            failed.append(url)
    return docs, failed


def build_vectorstore(docs, embeddings, status):
    # The Gemini free tier allows ~100 embedding requests per minute, so embed in
    # batches and wait out any rate-limit error instead of failing the whole run
    store = None
    for start in range(0, len(docs), EMBED_BATCH_SIZE):
        batch = docs[start:start + EMBED_BATCH_SIZE]
        status.text(f"Embedding {start}/{len(docs)} chunks")
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
                    status.text(f"Embedding {start}/{len(docs)} chunks · Gemini rate limit, "
                                f"resuming in {remaining}s")
                    time.sleep(1)
    status.empty()
    return store


# ======================
# RETRIEVAL
# ======================
def format_history(messages):
    recent = messages[-HISTORY_TURNS * 2:]
    if not recent:
        return "(none)"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def retrieve(vectorstore, question, history):
    # A compound question ("who founded X, and who owns Y?") embeds close to only one
    # of its topics, and a follow-up ("what about its revenue?") embeds close to
    # nothing, so first rewrite it into standalone sub-questions and retrieve for each
    decompose = DECOMPOSE_PROMPT | get_llm() | StrOutputParser()
    rewritten = decompose.invoke({"question": question, "history": history})
    sub_questions = [q.strip() for q in rewritten.splitlines() if q.strip()]
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
# RENDERING
# ======================
def render_snapshot(snap):
    st.subheader(f"{snap['name']} · {snap['ticker']}")
    if snap["sector"]:
        st.caption(f"{snap['sector']} · {snap['industry']}")

    change = None
    if snap["previous_close"]:
        change = (snap["price"] / snap["previous_close"] - 1) * 100

    price_col, cap_col, pe_col, range_col = st.columns(4)
    price_col.metric(f"Price ({snap['currency']})" if snap["currency"] else "Price", f"{snap['price']:,.2f}",
                     f"{change:+.2f}%" if change is not None else None)
    cap_col.metric("Market cap", format_big(snap["market_cap"]))
    pe_col.metric("P/E (TTM)", format_ratio(snap["trailing_pe"]))
    if snap["year_low"] and snap["year_high"]:
        range_col.metric("52-week range", f"{snap['year_low']:,.0f}–{snap['year_high']:,.0f}")

    prices = snap["history"].rename_axis("Date").reset_index()
    chart = alt.Chart(prices).mark_line(color=ACCENT, strokeWidth=1.5).encode(
        x=alt.X("Date:T", title=None),
        # Not anchored at zero, otherwise a six-month move looks like a flat line
        y=alt.Y("Close:Q", title=snap["currency"] or None, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("Close:Q", format=",.2f")],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)


def source_label(source, limit=60):
    title = st.session_state.get("source_titles", {}).get(source) or urlparse(source).netloc or source
    return title if len(title) <= limit else title[:limit - 1].rstrip() + "…"


def source_link(source):
    return f"[{source_label(source)}]({source})" if is_valid_url(source) else source


def render_references(message):
    if message["sources"]:
        st.caption("Sources: " + " · ".join(source_link(s) for s in message["sources"]))

    with st.expander(f"Passages used ({len(message['passages'])})"):
        if len(message["sub_questions"]) > 1:
            st.caption("Searched for: " + " · ".join(message["sub_questions"]))
        for i, (source, text) in enumerate(message["passages"], 1):
            st.markdown(f"**{i}.** {source_link(source)}")
            st.text(text)


def suggested_questions(snap):
    if snap:
        name = snap["short_name"]
        return [f"Summarise the latest news on {name}",
                f"What risks do the articles raise for {name}?",
                f"How is {name} valued right now?"]
    return ["Summarise these articles",
            "What risks do the articles mention?",
            "What figures or guidance do the articles give?"]


# ======================
# SIDEBAR
# ======================
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Research set")
    ticker = st.text_input("Ticker", placeholder="NVDA, AAPL, RELIANCE.NS",
                           help="Any symbol on Yahoo Finance. Adds live market data to the research set.")
    include_news = st.toggle("Add latest news for this ticker", value=True,
                             help=f"Pulls the {NUM_NEWS_ARTICLES} most relevant recent articles from Yahoo Finance.")
    st.markdown("**Article URLs**")
    urls = [st.text_input(f"URL {i+1}", placeholder="https://", label_visibility="collapsed")
            for i in range(NUM_URLS)]
    build_clicked = st.button("Build research set", type="primary", use_container_width=True)

    if st.session_state["messages"] and st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []

# ======================
# HEADER
# ======================
st.title("EquityBot")
st.caption("Research a company from live market data and the latest coverage. "
           "Every answer links to the passages it came from.")

# ======================
# BUILD RESEARCH SET
# ======================
if build_clicked:
    ticker = ticker.strip().upper()
    urls = [u.strip() for u in urls if u.strip()]

    if not ticker and not urls:
        st.error("Enter a ticker or at least one article URL.")
        st.stop()

    invalid = [u for u in urls if not is_valid_url(u)]
    if invalid:
        st.error("These URLs need to start with http:// or https://: " + ", ".join(invalid))
        st.stop()

    # Each browser session keeps its own index, so users never see each other's articles
    for key in ("vectorstore", "snapshot", "indexed_sources", "source_titles"):
        st.session_state.pop(key, None)
    st.session_state["messages"] = []

    with st.status("Building research set", expanded=True) as status:
        documents = []

        if ticker:
            st.write(f"Fetching market data for {ticker}")
            snapshot = get_snapshot(ticker)
            if snapshot is None:
                status.update(label="Couldn't build the research set", state="error")
                st.error(f"No market data found for {ticker}. Check the symbol on finance.yahoo.com "
                         "(non-US listings need a suffix, e.g. RELIANCE.NS).")
                st.stop()
            st.session_state["snapshot"] = snapshot
            documents.append(snapshot_document(snapshot))

            if include_news:
                company_word = snapshot["short_name"].split()[0]
                news_links = get_news_links(ticker, company_word)
                st.write(f"Found {len(news_links)} recent news article(s)")
                urls = list(dict.fromkeys(urls + news_links))

        if urls:
            st.write(f"Loading {len(urls)} article(s)")
            articles, failed = load_articles(urls)
            documents.extend(articles)
            for url in failed:
                st.write(f"Skipped {url} (the site blocked the request)")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = [c for c in text_splitter.split_documents(documents) if c.page_content.strip()]
        if not chunks:
            status.update(label="Couldn't build the research set", state="error")
            st.error("None of the sources returned readable text.")
            st.stop()

        try:
            st.session_state["vectorstore"] = build_vectorstore(chunks, get_embeddings(), st.empty())
        except Exception as e:
            status.update(label="Couldn't build the research set", state="error")
            st.error(f"Embedding failed: {e}")
            st.stop()

        st.session_state["source_titles"] = {
            d.metadata["source"]: (d.metadata.get("title") or "").strip() for d in documents
        }
        st.session_state["indexed_sources"] = list(st.session_state["source_titles"])
        status.update(label=f"Research set ready · {len(st.session_state['indexed_sources'])} sources, "
                            f"{len(chunks)} passages", state="complete", expanded=False)

# ======================
# RESEARCH SET OVERVIEW
# ======================
vectorstore = st.session_state.get("vectorstore")
snapshot = st.session_state.get("snapshot")

if snapshot:
    render_snapshot(snapshot)

if vectorstore is not None:
    sources = st.session_state.get("indexed_sources", [])
    with st.expander(f"Sources in this research set ({len(sources)})"):
        for source in sources:
            st.markdown(f"- {source_link(source)}")
else:
    st.info("Enter a ticker or paste article URLs in the sidebar, then click **Build research set**.")

# ======================
# CHAT
# ======================
for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_references(message)

query = st.chat_input(
    "Ask about this company or the articles" if vectorstore is not None else "Build a research set first",
    disabled=vectorstore is None,
)

if vectorstore is not None and not st.session_state["messages"] and not query:
    suggestions_box = st.empty()
    with suggestions_box.container():
        st.caption("Try asking")
        for column, suggestion in zip(st.columns(3), suggested_questions(snapshot)):
            if column.button(suggestion, use_container_width=True):
                query = suggestion
    if query:
        suggestions_box.empty()

if query:
    history = format_history(st.session_state["messages"])

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching the research set"):
                sub_questions, source_docs = retrieve(vectorstore, query, history)
            chain = PROMPT | get_llm() | StrOutputParser()
            answer = st.write_stream(chain.stream(
                {"context": format_docs(source_docs), "question": query, "history": history}
            )) or "No answer found"
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

        # Sources come from the retrieved chunks, not from parsing model output
        message = {
            "role": "assistant",
            "content": answer,
            "sources": list(dict.fromkeys(
                doc.metadata.get("source") for doc in source_docs if doc.metadata.get("source")
            )),
            "passages": [(doc.metadata.get("source", "unknown"), doc.page_content) for doc in source_docs],
            "sub_questions": sub_questions,
        }
        render_references(message)

    st.session_state["messages"] += [{"role": "user", "content": query}, message]
