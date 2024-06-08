**Overview**

This repository provides a robust web scraping solution tailored for AI applications. It extracts documents from websites, preprocesses the text, generates semantic embeddings, and indexes the enriched data into Elasticsearch. This enables powerful semantic search and retrieval capabilities.

**Features**

Dynamic Content Handling: Employs Selenium for JavaScript rendering and BeautifulSoup for parsing HTML.
Text Preprocessing: Cleans and standardizes text with NLTK (tokenization, stop word removal, stemming, lemmatization).
Semantic Embeddings: Leverages Sentence Transformers to create embeddings for semantic understanding.
Elasticsearch Integration: Indexes documents with embeddings into Elasticsearch, facilitating advanced search.
Efficient Processing: Utilizes multiprocessing for optimal performance with large datasets.

Install
Python 3.7+
Elasticsearch 7.x
Python libraries: nltk, requests, selenium, beautifulsoup4, sentence-transformers, torch, tqdm, haystack

Elasticsearch: Ensure your Elasticsearch cluster is running and accessible. Update connection details in the script or use arguments.

NLTK Resources: Download necessary resources:

``import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')``


``python scraper.py --es_host localhost --es_port 9200 --es_user user --es_pass pass --index_name web_data --model_name sentence-transformers/paraphrase-mpnet-base-v2 --urls https://example.com``

``Script Arguments
--es_host: Elasticsearch host (e.g., localhost)
--es_port: Elasticsearch port (e.g., 9200)
--es_user: Elasticsearch username
--es_pass: Elasticsearch password
--es_scheme: Elasticsearch scheme (http or https)
--index_name: Index name for Elasticsearch
--model_name: Sentence Transformer model (e.g., sentence-transformers/paraphrase-mpnet-base-v2)
--dataset_name: Dataset name (default: web_data)
--use_gpu: Use GPU for embeddings
--urls: URLs to scrape (space-separated)``
