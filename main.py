import argparse
import logging
import hashlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Dict
import nltk
import requests
import torch
from bs4 import BeautifulSoup
from haystack.nodes.retriever.multimodal import MultiModalRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.schema import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from urllib3.util.retry import Retry
from webdriver_manager.chrome import ChromeDriverManager
from collections import deque
from typing import Set, List
from urllib.parse import urljoin
from PIL import Image

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def initialize_document_store(args: Dict[str, str]) -> ElasticsearchDocumentStore:
    document_store = ElasticsearchDocumentStore(
        host=args['es_host'],
        port=args['es_port'],
        username=args['es_user'],
        password=args['es_pass'],
        scheme=args['es_scheme'],
        index=args['index_name'],
        verify_certs=False,
        return_embedding=True,
        embedding_field="embedding",
        embedding_dim=768,
        search_fields=["content", "meta.title", "meta.url", "meta.scrape_date", "meta.version", "meta.content_hash"],
        content_field="content",
        name_field="meta.title",
        custom_mapping=None,
        recreate_index=False,
        similarity="dot_product",
        timeout=9000,
        duplicate_documents="overwrite",
    )
    return document_store

def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
    return ' '.join(tokens)

def scrape_website(url: str) -> Dict[str, str]:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    # Extract metadata
    version = ''
    for meta in soup.find_all('meta'):
        if 'name' in meta.attrs and meta.attrs['name'].lower() == 'version':
            version = meta.attrs['content']

    # Scrape important content based on HTML structure
    content = {
        'title': soup.title.string if soup.title else '',
        'paragraphs': [p.get_text() for p in soup.find_all('p')],
        'metadata': {
            'url': url,
            'scrape_date': datetime.now().isoformat(),
            'version': version
        }
    }

    return content

def generate_content_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def is_duplicate(document_store: ElasticsearchDocumentStore, content_hash: str) -> bool:
    query = {
        "query": {
            "term": {
                "meta.content_hash.keyword": content_hash
            }
        }
    }
    results = document_store.query(query=query)
    return len(results["documents"]) > 0

def process_web_data(data: List[str], model_name: str, dataset_name: str, use_gpu: bool) -> List[Document]:
    model = SentenceTransformer(model_name)
    if use_gpu and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    else:
        model = model.to(torch.device("cpu"))

    documents = []
    for text in data:
        try:
            if not text:
                continue

            embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy().tolist()
            content_hash = generate_content_hash(text)
            document = Document(content=text, embedding=embedding, meta={"dataset_name": dataset_name, "content_hash": content_hash})
            documents.append(document)
        except Exception as e:
            logging.warning(f"Error processing text: {e}")
            continue

    return documents

class WebsiteGraph:
    def __init__(self):
        self.graph: Dict[str, List[str]] = {}
        self.processed_urls: Set[str] = set()

    def add_edge(self, source: str, target: str):
        if source not in self.graph:
            self.graph[source] = []
        self.graph[source].append(target)

    def get_unprocessed_neighbors(self, node: str) -> List[str]:
        return [neighbor for neighbor in self.graph.get(node, []) if neighbor not in self.processed_urls]

    def mark_processed(self, url: str):
        self.processed_urls.add(url)

def collect_urls_recursive(base_url: str, max_depth: int, processed_urls: Set[str], urls_to_scrape: List[str], current_depth: int = 0, max_retries: int = 3, backoff_factor: float = 0.3) -> None:
    if current_depth >= max_depth:
        return

    response = None
    try:
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        response = session.get(base_url)
        if response.status_code == 404:
            logging.warning(f"404 Not Found error for URL: {base_url}")
            return
        response.raise_for_status()

        if "xml" in response.headers.get("Content-Type", "").lower():
            soup = BeautifulSoup(response.text, "lxml-xml")
        else:
            soup = BeautifulSoup(response.text, "lxml")

        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])
            if url.startswith(base_url) and url not in processed_urls:
                if not any(exclude_word in url for exclude_word in ['Special:', 'Wikipedia:', 'Talk:', 'User:', 'User_talk:', 'Submissions:', 'Volunteer:', 'Editors:', 'About:']):
                    text = ' '.join(link.find_parent('p').get_text() for p in link.find_parents('p'))
                    if text:
                        urls_to_scrape.append(text)
                    processed_urls.add(url)
                    collect_urls_recursive(url, max_depth, processed_urls, urls_to_scrape, current_depth + 1, max_retries, backoff_factor)

    except (requests.exceptions.RequestException, OSError) as e:
        logging.error(f"Failed to collect URLs from {base_url}: {e}")
        if "Name or service not known" in str(e) or "Cannot assign requested address" in str(e):
            logging.warning(f"DNS resolution failed for {base_url}. Skipping URL.")
        elif "Too many open files" in str(e):
            time.sleep(backoff_factor * (2 ** (max_retries - 1)))
            collect_urls_recursive(base_url, max_depth, processed_urls, urls_to_scrape, current_depth, max_retries - 1, backoff_factor)

    finally:
        if response is not None:
            response.close()

def collect_urls_bfs(website_graph: WebsiteGraph, base_url: str, max_depth: int, urls_to_scrape: List[str]):
    queue = deque([(base_url, 0)])
    website_graph.mark_processed(base_url)

    while queue:
        url, depth = queue.popleft()
        if depth >= max_depth:
            continue

        collect_urls_recursive(url, max_depth - depth, website_graph.processed_urls, urls_to_scrape, current_depth=depth)

        for child_url in urls_to_scrape:
            website_graph.add_edge(url, child_url)
            if child_url not in website_graph.processed_urls:
                website_graph.mark_processed(child_url)
                queue.append((child_url, depth + 1))

def index_web_data(document_store: ElasticsearchDocumentStore, data: List[str], batch_size: int, model_name: str, dataset_name: str, use_gpu: bool) -> None:
    num_batches = (len(data) + batch_size - 1) // batch_size

    with Pool(cpu_count()) as pool:
        results = []
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            results.append(pool.apply_async(process_web_data, (batch_data, model_name, dataset_name, use_gpu)))

        for result in results:
            documents = result.get()
            for document in documents:
                if not is_duplicate(document_store, document.meta["content_hash"]):
                    document_store.write_documents([document])
                    logging.info(f"Processed and indexed document: {document.id}")
                else:
                    logging.info(f"Duplicate document found: {document.id}")

    logging.info("Data preprocessing and indexing complete!")

def main() -> None:
    parser = argparse.ArgumentParser(description="Web Scraping, Processing, and Storing in Elasticsearch")
    parser.add_argument('--es_host', required=True, help='Elasticsearch host')
    parser.add_argument('--es_port', required=True, help='Elasticsearch port')
    parser.add_argument('--es_user', required=True, help='Elasticsearch username')
    parser.add_argument('--es_pass', required=True, help='Elasticsearch password')
    parser.add_argument('--es_scheme', default='http', help='Elasticsearch scheme (http or https)')
    parser.add_argument('--index_name', required=True, help='Index name for Elasticsearch')
    parser.add_argument('--model_name', default='sentence-transformers/paraphrase-mpnet-base-v2', help='Sentence Transformer model name')
    parser.add_argument('--dataset_name', default='web_data', help='Dataset name for the scraped data')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for embedding generation')
    parser.add_argument('--urls', nargs='+', required=True, help='URLs to scrape')
    args = vars(parser.parse_args())

    logging.info("Initializing document store")
    document_store = initialize_document_store(args)

    logging.info("Loading SentenceTransformer model")
    device = torch.device("cuda" if torch.cuda.is_available() and args['use_gpu'] else "cpu")
    model = SentenceTransformer(args['model_name']).to(device)
    
    website_graph = WebsiteGraph()
    urls_to_scrape = []

    for url in args['urls']:
        logging.info(f"Collecting URLs from website: {url}")
        collect_urls_bfs(website_graph, url, max_depth=3, urls_to_scrape=urls_to_scrape)

    all_paragraphs = []
    for url in urls_to_scrape:
        logging.info(f"Scraping website: {url}")
        scraped_data = scrape_website(url)
        all_paragraphs.extend(scraped_data['paragraphs'])

    logging.info(f"Starting data preprocessing and indexing")
    index_web_data(document_store, all_paragraphs, batch_size=128, model_name=args['model_name'], dataset_name=args['dataset_name'], use_gpu=args['use_gpu'])

    # Initialize MultiModalRetriever
    retriever_text_to_image = MultiModalRetriever(
        document_store=document_store,
        query_embedding_model="sentence-transformers/all-mpnet-base-v2",
        document_embedding_models={
            "text": "sentence-transformers/all-mpnet-base-v2",
            "table": "sentence-transformers/msmarco-roberta-base-v3",
            "image": "sentence-transformers/clip-ViT-B-32"
        },
        query_type="text",
        embed_meta_fields=["content", "meta.title", "meta.url", "meta.scrape_date", "meta.version", "meta.content_hash"],
        top_k=25,
        batch_size=16,
        similarity_function="dot_product",
        progress_bar=True
    )

    # Initialize Reader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    # Initialize Pipeline
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_text_to_image)

    # Interactive Querying
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = pipeline.run(query=query)

        # Display the results
        for result in results['answers']:
            print(f"Answer: {result['answer']}, Score: {result['score']}, Meta: {result['meta']}")

if __name__ == "__main__":
    main()
