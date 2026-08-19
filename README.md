# AI Equity Research Assistant

# EquityBot :- News Research Tool

EquityBot is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain powered by **Google Gemini**.

## Features

- Load URLs to fetch article content.
- Process article content through LangChain's document loaders.
- Construct embedding vectors using **Google's `text-embedding-004`** and leverage **FAISS** to enable swift similarity search and retrieval.
- Interact with **Google Gemini (`gemini-1.5-flash`)** to query articles and receive grounded answers along with source URLs.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Nitesh1123/equity-research-tool.git
```

2. Navigate to the project directory:

```bash
cd equity-research-tool
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your Google Gemini API key by creating a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage/Examples

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. The web app will open in your browser:
- On the sidebar, input news article URLs.
- Click **"Process URLs"** to parse, split, and embed the text into a local FAISS index.
- Ask questions in the search box to receive answers with source citations.

## Project Structure

- `main.py`: The main Streamlit application script.
- `requirements.txt`: Python package dependencies.
- `faiss_store_gemini/`: Local directory storing the FAISS vector index.
- `.env`: Local configuration file storing your `GOOGLE_API_KEY`.
