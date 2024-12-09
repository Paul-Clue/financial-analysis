# Financial Analysis Application

An AI-powered financial analysis tool that provides intelligent insights about stocks using natural language queries and custom filters.

## Overview

This application combines multiple AI technologies to analyze stock market data and provide detailed responses to user queries. It uses embeddings to understand questions semantically and leverages large language models to generate comprehensive, natural language responses about stocks and financial markets.

## Features

- Natural language queries about stocks and financial markets
- Custom filtering by:
  - Industry
  - Sector
  - Market Cap
  - Trading Volume
- Real-time stock data integration via Yahoo Finance
- AI-powered analysis using LangChain and Groq
- Vector similarity search using Pinecone
- Web interface built with Streamlit

## Prerequisites

- Python 3.10+
- Pinecone API key
- OpenAI API key
- Groq API key
- HuggingFace API key
- Yahoo Finance API key
- NewsAPI key

## Installation and Setup

1. Clone the repository: 

```bash
git clone https://github.com/Paul-Clue/financial-analysis.git
cd financial-analysis
```

2. Create and activate a virtual environment (recommended):

```bash
On Windows
python -m venv venv
.\venv\Scripts\activate
On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:

PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_huggingface_key
YAHOO_ACCESS_TOKEN=your_yahoo_key
GROQ_API_KEY=your_groq_key
NEWSAPI_API_KEY=your_newsapi_key

## Running the Application

There are two ways to run the application:

### Option 1: Using Jupyter Notebook

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open `finance.ipynb` in your browser
3. Run the cells in order to:
   - Initialize the vector database
   - Process stock data
   - Set up the embeddings
   - Launch the Streamlit interface

### Option 2: Using Streamlit Directly

If you've already run the notebook once to initialize everything:

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

## Using the Application

1. In the web interface, you can:
   - Enter natural language questions about stocks in the query box
   - Use the filters to narrow down results by:
     - Industry
     - Sector
     - Market Cap
     - Volume

2. Example queries:
   - "What are the top performing tech companies?"
   - "Show me companies in the healthcare sector with market cap over 1B"
   - "Which companies have the highest trading volume in the energy sector?"

3. The application will:
   - Process your query
   - Search the vector database
   - Generate a detailed response using AI
   - Display the results in a readable format

## Project Structure

financial-analysis/
├── app.py # Streamlit web application
├── finance.ipynb # Jupyter notebook with development code
├── requirements.txt # Python dependencies
├── .env # API keys and configuration
├── .gitignore # Git ignore rules
├── company_tickers.json # Stock ticker data
├── successful_tickers.txt # Processing tracking
└── unsuccessful_tickers.txt # Error tracking

## How It Works

1. **Data Collection**: The application fetches stock data using the Yahoo Finance API.

2. **Vector Embeddings**: Stock descriptions and user queries are converted into vector embeddings using HuggingFace's sentence transformers.

3. **Similarity Search**: Pinecone performs vector similarity search to find relevant stock information.

4. **AI Analysis**: Groq's LLM processes the matched information and generates natural language responses.

5. **Web Interface**: Streamlit provides an intuitive interface for interacting with the system.

## API Keys Required

- **Pinecone**: Vector database for similarity search
- **OpenAI**: Alternative LLM provider
- **Groq**: Primary LLM for analysis
- **HuggingFace**: Embedding models
- **Yahoo Finance**: Stock data
- **NewsAPI**: Financial news integration

## Troubleshooting

If you encounter issues:

1. Ensure all API keys are correctly set in `.env`
2. Check that the virtual environment is activated
3. Verify all dependencies are installed correctly
4. Make sure Pinecone index is properly initialized (run the notebook first)
5. Check the console for any error messages

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://python.langchain.com/docs/get_started/introduction) for AI integration
- [Pinecone](https://www.pinecone.io/) for vector search capabilities
- [Groq](https://groq.com/) for LLM processing
- [Streamlit](https://streamlit.io/) for the web interface
- [Yahoo Finance](https://finance.yahoo.com/) for financial data

## Contact

Paul Clue - [GitHub Profile](https://github.com/Paul-Clue)

Project Link: [https://github.com/Paul-Clue/financial-analysis](https://github.com/Paul-Clue/financial-analysis)
