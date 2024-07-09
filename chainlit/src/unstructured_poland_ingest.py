import os

import chromadb
import torch
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# This list is taken from LangChain's MarkdownTextSplitter class.
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
gist_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=MARKDOWN_SEPARATORS, add_start_index=True, strip_whitespace=True)
chromadb_host = os.getenv("CHROMA_HOST", "localhost")
chromadb_port = os.getenv("CHROMA_PORT", "8000")
os.environ["TRANSFORMERS_CACHE"] = "./.cache/huggingface/transformers"
enviro = os.getenv("ENVIRONMENT", "local")
persistent_client_chroma = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
# create the open-source embedding function
opensource_embedding_function = SentenceTransformerEmbeddings(model_name='intfloat/multilingual-e5-large-instruct',
                                                              model_kwargs=  # {'device': 'cuda'} if torch.cuda.is_available() else
                                                              {
                                                                  'device': 'cpu'},
                                                              encode_kwargs={'normalize_embeddings': True})  # set True to compute cosine similarity


def load_handbook_pl(filepath: str):
    print("Loading file name ", filepath)
    docs = []
    try:
        loader = UnstructuredPDFLoader(filepath, mode="elements", strategy="hi_res")
        loaded_docs = loader.load()
        # for doc in documents:
        #     print(doc.page_content[0:10], doc.metadata)
        docs = aggregate_documents(loaded_docs)
        # docs = gist_text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"Employee Handbook for TEMPLATE Poland Page {doc.metadata['page_number']}"
    except Exception as e:
        print(f"Error occurred: {e}")
    print(f"Loaded {len(docs)} documents")
    return docs


import json


def aggregate_documents(docs):
    # Create a dictionary where the keys are the page numbers and the values are lists of documents
    page_dict = {}
    category_dict = {}
    category_examples = {}  # Dictionary to store one example document for each category
    for doc in docs:
        page_number = doc.metadata['page_number']
        if page_number not in page_dict:
            page_dict[page_number] = []
        page_dict[page_number].append(doc)

        # Count the categories and store one example document for each category
        category = doc.metadata['category']
        if category not in category_dict:
            category_dict[category] = 0
            category_examples[category] = doc  # Store the first document of each category as an example
        category_dict[category] += 1

    # For each page number, create a new document object where the page_content and metadata are aggregations of all documents
    aggregated_docs = []
    for page_number, documents in page_dict.items():
        aggregated_page_content = ' '.join(doc.page_content for doc in documents)
        aggregated_metadata = {}
        categories = []
        for doc in documents:
            for key, value in doc.metadata.items():
                if key == 'coordinates':
                    continue
                if key == 'category':
                    categories.append(value)
                    continue
                if isinstance(value, list):
                    value = ', '.join(value)
                elif isinstance(value, dict):
                    value = json.dumps(value)
                elif not isinstance(value, (str, int, float, bool)):
                    continue
                aggregated_metadata[key] = value
        aggregated_metadata['category'] = ', '.join(categories)

        # Create a new document object with the aggregated page_content and metadata
        aggregated_doc = documents[0]
        aggregated_doc.page_content = aggregated_page_content
        aggregated_doc.metadata = aggregated_metadata
        aggregated_docs.append(aggregated_doc)

    # Print the category counts and one example document for each category
    for category, count in category_dict.items():
        print(f'Category: {category}, Count: {count}')
        example_doc = category_examples[category]
        print(f'Example document for category {category}:')
        print(f'Page content: {example_doc.page_content}')
        print(f'Metadata: {example_doc.metadata}')

    return aggregated_docs


poland_collection_name = "TEMPLATE_poland_handbook_unstructured"
poland_collection_name_single = "TEMPLATE_poland_handbook_unstructured_single"
# persistent_client_chroma.delete_collection(poland_collection_name)
# persistent_client_chroma.delete_collection(poland_collection_name_single)
vectorstore = Chroma(
    client=persistent_client_chroma,
    collection_name=poland_collection_name,
    embedding_function=opensource_embedding_function,
)
vectorstore_single = Chroma(
    client=persistent_client_chroma,
    collection_name=poland_collection_name_single,
    embedding_function=opensource_embedding_function,
)
count = vectorstore._collection.count()
count_single = vectorstore_single._collection.count()

if count == 0:
    filepath = os.path.join(".", "public", "MCIA-PolandHandbook.pdf")
    docs = load_handbook_pl(filepath)
    if not docs:
        print("No documents found in the handbook")
        raise ValueError("No documents found in the handbook")
    vectorstore.add_documents(docs, embedding=opensource_embedding_function)

if count_single == 0:
    filepath = os.path.join(".", "public", "MCIA-PolandHandbook.pdf")
    docs = []
    try:
        loader = UnstructuredPDFLoader(filepath, mode="single", strategy="hi_res")
        documents = loader.load()
        docs = gist_text_splitter.split_documents(documents)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"Employee Handbook for TEMPLATE Poland Part {i + 1}"
        vectorstore_single.add_documents(docs, embedding=opensource_embedding_function)
    except Exception as e:
        print(f"Error occurred: {e}")
    print(f"Loaded {len(docs)} documents")
