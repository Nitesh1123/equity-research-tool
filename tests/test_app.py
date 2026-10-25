"""End-to-end tests for the Streamlit app.

Gemini, Yahoo Finance and web loading are replaced with fakes, so the tests run
offline, need no API key and use no quota.
"""
import langchain_community.document_loaders as document_loaders
import langchain_google_genai as google_genai
import pandas as pd
import pytest
import streamlit as st
import yfinance as yf
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

APP = "../main.py"  # resolved relative to this file


class FakeLoader:
    def __init__(self, web_paths):
        self.url = web_paths[0]

    def load(self):
        if "blocked" in self.url:
            raise RuntimeError("403 Forbidden")
        return [Document(page_content=f"Nvidia revenue rose 20% in Q2. ({self.url})",
                         metadata={"source": self.url, "title": f"Story {self.url.rsplit('/', 1)[-1]}"})]


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period):
        if self.symbol != "NVDA":
            return pd.DataFrame()
        dates = pd.date_range("2026-05-01", periods=120, freq="D")
        return pd.DataFrame({"Close": [100.0 + i for i in range(120)]}, index=dates)

    @property
    def info(self):
        return {"longName": "NVIDIA Corporation", "shortName": "NVIDIA Corporation", "currency": "USD",
                "marketCap": 5.17e12, "trailingPE": 27.07, "forwardPE": 13.4,
                "fiftyTwoWeekLow": 164.27, "fiftyTwoWeekHigh": 243.37,
                "sector": "Technology", "industry": "Semiconductors",
                "longBusinessSummary": "Nvidia designs GPUs."}


class FakeSearch:
    def __init__(self, query, news_count):
        self.news = [
            {"title": "Microsoft target hike", "link": "https://news.example/msft", "relatedTickers": ["NVDA", "MSFT"]},
            {"title": "Nvidia expands alliance", "link": "https://news.example/nvda", "relatedTickers": ["NVDA"]},
            {"title": "Apple launches phone", "link": "https://news.example/aapl", "relatedTickers": ["AAPL"]},
            {"title": "NVIDIA story", "link": "https://blocked.example/nvda", "relatedTickers": ["NVDA"]},
        ]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    # Model calls alternate: rewrite the question, then answer it
    llm = FakeListChatModel(responses=["How is NVIDIA valued?", "Trailing P/E is 27.1."])
    monkeypatch.setattr(google_genai, "ChatGoogleGenerativeAI", lambda **kwargs: llm)
    monkeypatch.setattr(google_genai, "GoogleGenerativeAIEmbeddings",
                        lambda **kwargs: DeterministicFakeEmbedding(size=16))
    monkeypatch.setattr(document_loaders, "WebBaseLoader", FakeLoader)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yf, "Search", FakeSearch)
    # Streamlit caches outlive a single AppTest run, so start each test clean
    st.cache_resource.clear()
    st.cache_data.clear()


def run_app():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception, app.exception
    return app


def build(app, ticker="", urls=()):
    app.text_input(key="ticker").input(ticker)
    for i, url in enumerate(urls, 1):
        app.text_input(key=f"url_{i}").input(url)
    app.button(key="build").click().run()
    assert not app.exception, app.exception
    return app


def ask(app, question):
    app.chat_input[0].set_value(question).run()
    assert not app.exception, app.exception
    return app


def test_chat_is_disabled_until_a_research_set_is_built():
    app = run_app()
    assert app.chat_input[0].disabled
    assert [app.button(key=f"example_{t}").label for t in ("NVDA", "AAPL", "MSFT")] == ["NVDA", "AAPL", "MSFT"]


def test_sidebar_explains_the_project_and_links_the_code():
    app = run_app()
    sidebar_text = " ".join(m.value for m in app.sidebar.markdown)
    assert "How it works" in sidebar_text
    assert "https://github.com/Nitesh1123/equity-research-tool" in sidebar_text


def test_example_ticker_builds_a_research_set_in_one_click():
    app = run_app()
    app.button(key="example_NVDA").click().run()
    assert not app.exception, app.exception
    assert app.text_input(key="ticker").value == "NVDA"
    assert "vectorstore" in app.session_state
    # Examples are only offered before the first research set
    assert not [b for b in app.button if b.key and b.key.startswith("example_")]


def test_requires_a_ticker_or_url():
    app = build(run_app())
    assert app.error[0].value == "Enter a ticker or at least one article URL."


def test_rejects_urls_without_http_scheme():
    app = build(run_app(), urls=["ftp://example.com/file"])
    assert "ftp://example.com/file" in app.error[0].value
    assert "vectorstore" not in app.session_state


def test_unknown_ticker_shows_a_clear_error():
    app = build(run_app(), ticker="zzzz")
    assert "No market data found for ZZZZ" in app.error[0].value
    assert "vectorstore" not in app.session_state


def test_ticker_shows_snapshot_and_indexes_relevant_news():
    app = build(run_app(), ticker="nvda")

    assert app.title[0].value == "219.00"
    metrics = {m.label: m.value for m in app.metric}
    assert metrics == {"Market cap": "5.17T", "P/E (TTM)": "27.1",
                       "Forward P/E": "13.4", "52-week range": "164–243"}

    sources = app.session_state["indexed_sources"]
    assert sources[0] == "https://finance.yahoo.com/quote/NVDA"
    assert "https://news.example/nvda" in sources
    # Not tagged with the ticker, so never fetched
    assert "https://news.example/aapl" not in sources
    # Blocked sites are skipped instead of failing the build
    assert "https://blocked.example/nvda" not in sources
    assert not app.chat_input[0].disabled


def test_suggested_question_answers_with_sources():
    app = build(run_app(), ticker="NVDA")
    assert app.button(key="suggestion_2").label == "How is NVIDIA valued right now?"

    app.button(key="suggestion_2").click().run()
    assert not app.exception, app.exception

    user, assistant = app.chat_message
    assert user.markdown[0].value == "How is NVIDIA valued right now?"
    assert assistant.markdown[0].value == "Trailing P/E is 27.1."
    assert "NVDA market data (Yahoo Finance)" in assistant.caption[0].value
    # Suggestions disappear once the conversation starts
    assert not [b for b in app.main.button if b.key and b.key.startswith("suggestion_")]


def test_follow_ups_keep_history_and_clear_chat_resets_it():
    app = build(run_app(), urls=["https://example.com/a"])
    ask(app, "What happened to revenue?")
    ask(app, "And why?")
    assert len(app.session_state["messages"]) == 4
    assert len(app.chat_message) == 4

    app.button(key="clear_chat").click().run()
    assert app.session_state["messages"] == []
    assert len(app.chat_message) == 0


def test_url_only_research_set_has_no_snapshot():
    app = build(run_app(), urls=["https://example.com/a"])
    assert len(app.metric) == 0
    assert app.session_state["indexed_sources"] == ["https://example.com/a"]
    assert app.button(key="suggestion_0").label == "Summarise these articles"


def test_each_session_has_its_own_research_set():
    build(run_app(), urls=["https://example.com/a"])
    other_session = run_app()
    assert "vectorstore" not in other_session.session_state
    assert other_session.chat_input[0].disabled
