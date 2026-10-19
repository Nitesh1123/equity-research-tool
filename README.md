# EquityBot: AI Equity Research Tool

EquityBot is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain powered by **Google Gemini**.

## Links

- **GitHub:** https://github.com/Nitesh1123/equity-research-tool
- **Live app:** https://equity-research-tool-7kkour4wxz3wuccdakcuhi.streamlit.app/

## Features

- Load URLs to fetch article content.
- Process article content through LangChain's document loaders.
- Construct embedding vectors using **Google's `gemini-embedding-001`** and leverage **FAISS** to enable swift similarity search and retrieval.
- Interact with **Google Gemini (`gemini-2.5-flash`)** to query articles and receive grounded answers along with source URLs.
- Compound questions spanning several articles are split into sub-questions, each retrieved separately, and the passages behind every answer can be inspected in the app.
- Chat interface with memory: ask follow-ups like "what about its revenue?" and answers stream in as they're written.
- Each browser session gets its own in-memory index, so multiple users of the hosted app never see each other's articles.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Nitesh1123/equity-research-tool.git
```

2. Navigate to the project directory:

```bash
cd equity-research-tool
```

3. Install the required dependencies (Python 3.10+):

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
- Click **"Process URLs"** to parse, split, and embed the text into an in-memory FAISS index for your session.
- Chat with the articles in the box at the bottom; each answer lists its sources and the passages it used. Use **Clear chat** in the sidebar to start over.

## Project Structure

- `main.py`: The main Streamlit application script.
- `requirements.txt`: Python package dependencies.
- `.env`: Local configuration file storing your `GOOGLE_API_KEY`.
