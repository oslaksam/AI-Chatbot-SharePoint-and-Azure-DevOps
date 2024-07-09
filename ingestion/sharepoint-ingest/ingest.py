import json
import logging
import os
import time
import uuid

import chromadb
import redis
import torch
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader, UnstructuredPDFLoader,
    UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, TextLoader, Docx2txtLoader, PyPDFLoader
)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.storage import RedisStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from pymongo import MongoClient, errors as pymongo_errors
from redis.exceptions import ConnectionError
from tqdm import tqdm

from sharepoint_filter import *

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
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

load_dotenv()
env = os.getenv("ENV", "local")

chromadb_host = os.getenv("CHROMA_HOST", "localhost")
chromadb_port = os.getenv("CHROMA_PORT", "8529")
persist_directory = os.getenv("PERSIST_DIRECTORY", "/chroma/chroma")
chroma_collection_full_docs = os.getenv("CHROMA_COLLECTION_FULL_DOCS", "sharepoint_full_docs")
chroma_collection_chunks = os.getenv("CHROMA_COLLECTION_CHUNKS", "sharepoint_chunks")

username = os.getenv('MONGO_INITDB_ROOT_USERNAME')
password = os.getenv('MONGO_INITDB_ROOT_PASSWORD')
mongo_host = os.getenv('MONGO_HOST', 'host.docker.internal')
mongo_port = os.getenv('MONGO_PORT', '27017')
mongo_uri = f"mongodb://{username}:{password}@{mongo_host}:{mongo_port}"
db_name = os.getenv('MONGO_DB_NAME', 'sharepoint')
mongo_documents_collection_name = os.getenv('MONGO_DOCUMENTS_COLLECTION_NAME', 'documentLibrary')
mongo_pages_collection_name = os.getenv('MONGO_PAGES_COLLECTION_NAME', 'pages')

redis_host = os.getenv("REDIS_HOST", "host.docker.internal")
redis_port = os.getenv("REDIS_PORT", "6379")

scrape_folder = os.getenv('SCRAPE_FOLDER_PATH', "/data")


def create_redis_client(host='host.docker.internal', port=6379, decode_responses=True):
    return redis.Redis(host=host, port=port, decode_responses=decode_responses)


def check_redis_health(redis_client):
    try:
        return redis_client.ping()
    except ConnectionError:
        return False


def update_doc_metadata_from_mongodb(doc, collection):
    source_path = doc.metadata["source"]
    filename = os.path.basename(source_path)
    mongo_doc = collection.find_one({"documentName": filename})

    if mongo_doc:
        excluded_keys = ['file', 'fileSystemInfo', 'createdBy', 'lastModifiedBy']
        mongo_doc_metadata = {k: v for k, v in mongo_doc.items() if k not in excluded_keys}
        doc.metadata.update(mongo_doc_metadata)
        logging.info(f"Document metadata updated from MongoDB for {filename}")
    else:
        logging.warning(f"No document found in MongoDB for {filename}")
        # Constructing the document to insert/update
        new_document = {
            "documentUrl": "https://teamorgchart.com/toc/chart/b4720c98-18a5-4519-8233-de4fa6a8909b",
            "documentName": filename,
            "eTag": "0",
            "documentPath": source_path,
            "description": "No Description",
            "siteName": "EmployeePortal"
        }
        new_id = "EmployeePortal" + str(uuid.uuid4())
        # Use the document ID as the filter for the update operation
        result = collection.update_one(
            {"_id": new_id},
            {"$set": new_document},
            upsert=True
        )
        new_mongo_doc = collection.find_one({"_id": new_id})
        mongo_doc_metadata = {k: v for k, v in new_mongo_doc.items()}
        doc.metadata.update(mongo_doc_metadata)
        logging.info(f"Document metadata updated from MongoDB for the new file: {filename}")


def save_document_to_redis(document, prefix="document_2000:"):
    document_id = prefix + document.metadata["_id"]
    dict_document = {
        "page_content": document.page_content,
        "metadata": document.metadata
    }
    serialized_document = json.dumps(dict_document)
    redis_store.mset([(document_id, serialized_document)])
    logging.info(f"Document saved to Redis: {document_id}")


def save_chunk_to_redis(key: str, store: InMemoryStore, prefix="document_2000:"):
    document = store.mget([key])[0]
    document_id = prefix + document.metadata["_id"]  # + ":"  + key
    dict_document = {
        "page_content": document.page_content,
        "metadata": document.metadata
    }
    serialized_document = json.dumps(dict_document)
    redis_store.mset([(document_id, serialized_document)])
    logging.info(f"Chunk saved to Redis: {document_id}")


def load_document_from_redis(document_id: str, prefix="document_2000:") -> Document:
    serialized_documents = redis_store.mget([prefix + document_id])
    serialized_document = serialized_documents[0] if serialized_documents else None
    if serialized_document:
        document_dict = json.loads(serialized_document)
        return Document(page_content=document_dict["page_content"], metadata=document_dict["metadata"])
    logging.warning(f"Document not found in Redis: {document_id}")
    return None


def save_in_memory_store_to_redis(store: InMemoryStore, prefix: str = "document_2000:") -> None:
    for key in store.yield_keys():
        save_chunk_to_redis(key, store, prefix=prefix)


def create_chunks_retriever():
    # Initialize the text splitter and retriever
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # the maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=20,  # the number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,  # the maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=20,  # the number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )
    vectorstore = Chroma(
        client=chroma_persistent_client,
        persist_directory=persist_directory,
        collection_name=chroma_collection_chunks,
        embedding_function=embedding_function,
    )

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever


def create_full_docs_retriever():
    # Initialize the text splitter and retriever
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # the maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=20,  # the number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )
    store = InMemoryStore()
    vectorstore = Chroma(
        client=chroma_persistent_client,
        persist_directory=persist_directory,
        collection_name=chroma_collection_full_docs,
        embedding_function=embedding_function,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    return retriever


def decide_collection(doc, docs_collection, pages_collection):
    source_path = doc.metadata["source"]
    filename = os.path.basename(source_path)
    if filename.endswith((".pptx", ".ppt", ".pdf", ".docx", ".xlsx", ".xlsm", ".txt")):
        return docs_collection
    elif filename.endswith(".aspx.md"):
        return pages_collection
    else:
        return None


def process_and_add_document(doc, redis_ids_full_docs, docs_collection, pages_collection, redis_client):
    decision = decide_collection(doc, docs_collection, pages_collection)
    if decision is None:
        logging.warning(f"Skipping document: {doc.metadata.get('_id')}")
        return
    update_doc_metadata_from_mongodb(doc, decision)
    logging.info(f"Processing document: {doc.metadata.get('_id')}")
    # Create and configure the composite filter
    composite_filter = CompositeFilter(filters=[HomepageFilter(), NewsFilter(), LocationsFilter()])
    if not composite_filter.apply(doc):
        logging.info(f"Filtered out document: {doc.metadata.get('_id')}")
        with open(failed_docs_log_path, 'a') as f:
            # Write each failed document path to the file
            source_path = doc.metadata["source"]
            f.write(f"{source_path}\n")
        return
    add_full_doc_to_retriever(doc, redis_ids_full_docs, redis_client)
    add_chunks_to_retriever(doc, create_chunks_retriever(), redis_client)


def add_full_doc_to_retriever(doc, redis_ids_full_docs, redis_client):
    prefix = "full_document:"
    full_docs_retriever = create_full_docs_retriever()
    doc_id = doc.metadata.get('_id')
    if not doc_id or not doc_id.strip():  # Check if doc_id is not empty or not only whitespace
        logging.warning(f"Document ID is empty: {doc.metadata}")
        logging.warning(f"Skipping document: {doc.metadata}")
        return  # Skip the document if the ID is empty
    if (prefix + doc_id) in redis_ids_full_docs:
        redis_doc = load_document_from_redis(doc_id, prefix=prefix)
        if not redis_doc or redis_doc.metadata.get("eTag") != doc.metadata.get("eTag"):
            logging.info(f"ETag has changed for document ID {doc_id}. Old ETag: {redis_doc.metadata.get('eTag')}, New ETag: {doc.metadata.get('eTag')}")
            # Remove the old document from the vectorstore
            old_chunk_keys = list(redis_client.scan_iter(match=f"{prefix}{doc_id}*"))
            decoded = [key.decode("utf-8") for key in old_chunk_keys]
            for key in decoded:
                logging.debug(f"Deleting old chunk from Redis: {key}")
                redis_client.delete(key)
            full_docs_retriever.vectorstore.delete([doc_id])
            keys = full_docs_retriever.vectorstore.get(where={"_id": doc_id})
            if keys["ids"]:
                full_docs_retriever.vectorstore.delete(keys["ids"])
            if doc_id and len([doc_id]) >= 1 and doc_id.strip():  # Check again if doc_id is not empty and if doc_id is not None and not an empty string
                try:
                    full_docs_retriever.add_documents(documents=[doc], ids=[doc_id], add_to_docstore=True)
                    save_document_to_redis(doc, prefix=prefix)
                    logging.info(f"Document {doc_id} updated in Redis and retrievers")
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    with open(failed_docs_log_path, 'a') as f:
                        # Write each failed document path to the file
                        source_path = doc.metadata["source"]
                        f.write(f"{source_path}\n")
            else:
                logging.warning(f"Document ID is empty: {doc.metadata}")
                logging.warning(f"Skipping document: {doc.metadata}")
                return
        else:
            logging.info(f"ETag is the same, not updating Redis and retriever for document {doc_id}")
    else:
        if doc_id and len([doc_id]) >= 1 and doc_id.strip():  # Check again if doc_id is not empty and if doc_id is not None and not an empty string
            try:
                save_document_to_redis(doc, prefix=prefix)
                full_docs_retriever.add_documents([doc], ids=[doc_id], add_to_docstore=True)
                logging.info(f"Document {doc_id} added to Redis and retrievers")
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                with open(failed_docs_log_path, 'a') as f:
                    # Write each failed document path to the file
                    source_path = doc.metadata["source"]
                    f.write(f"{source_path}\n")
        else:
            logging.warning(f"Document ID is empty: {doc.metadata}")
            logging.warning(f"Skipping document: {doc.metadata}")
            return

    logging.info(f"Vectorstore full_docs collection count: {full_docs_retriever.vectorstore._collection.count()}")


def add_chunks_to_retriever(doc, chunk_docs_retriever, redis_client: redis.Redis):
    prefix = "document_2000:"
    doc_id = doc.metadata.get('_id')  # Parent document ID
    if not doc_id or not doc_id.strip():  # Check if doc_id is not empty or not only whitespace
        logging.warning(f"Document ID is empty: {doc.metadata}")
        logging.warning(f"Skipping document: {doc.metadata}")
        return  # Skip the document if the ID is empty

    # Determine if the parent document's eTag has changed, indicating a need for update
    redis_doc = load_document_from_redis(doc_id, prefix=prefix)
    if redis_doc is None:
        logging.info(f"Document ID {doc_id} not found in Redis, adding it")
        save_document_to_redis(doc, prefix=prefix)
    elif redis_doc and redis_doc.metadata.get("eTag") == doc.metadata.get("eTag"):
        logging.info(f"ETag is the same, not updating chunks for document ID {doc_id}")
        return  # No need to update if eTag matches
    elif redis_doc and redis_doc.metadata.get("eTag") != doc.metadata.get("eTag"):
        logging.info(f"ETag has changed for document ID {doc_id}. Old ETag: {redis_doc.metadata.get('eTag')}, New ETag: {doc.metadata.get('eTag')}")
        chunk_docs_retriever.vectorstore.delete([doc_id])
        keys = chunk_docs_retriever.vectorstore.get(where={"_id": doc_id})
        if keys["ids"]:
            chunk_docs_retriever.vectorstore.delete(keys["ids"])
        # Clear old chunks from both Redis and the vector store
        old_chunk_keys = list(redis_client.scan_iter(match=f"{prefix}{doc_id}*"))
        decoded = [key.decode("utf-8") for key in old_chunk_keys]
        for key in decoded:
            logging.debug(f"Deleting old chunk from Redis: {key}")
            redis_client.delete(key)
        save_document_to_redis(doc, prefix=prefix)

    # Split the document into parent and child chunks
    parent_chunks = chunk_docs_retriever.parent_splitter.split_documents([doc]) if chunk_docs_retriever.parent_splitter else [doc]
    logging.info(f"Parent chunk count: {len(parent_chunks)}")
    child_chunks = []
    parents_redis = []
    doc_chunk_ids = []

    sequential_id = 0  # Initialize a counter for sequential IDs

    for i, parent_chunk in enumerate(parent_chunks):
        sub_docs = chunk_docs_retriever.child_splitter.split_documents([parent_chunk])
        parent_id = doc_id + f":{i}"
        logging.info(f"Processing parent chunk number {parent_id}")
        parent_chunk.metadata["_id"] = parent_id
        parents_redis.append((parent_id, parent_chunk))
        logging.info(f"Child chunk count: {len(sub_docs)}")
        for sub_doc in sub_docs:
            chunk_id = f"{doc_id}:{sequential_id}"
            sequential_id += 1  # Increment the sequential ID for the next chunk

            # Copy all metadata from parent, then add or overwrite specific fields for the child
            child_metadata = dict(doc.metadata)  # Start with the parent's metadata
            child_metadata.update({
                "_id": chunk_id,
                "eTag": doc.metadata.get("eTag", ""),
                "doc_id": parent_id  # Link back to the parent document
            })

            # Prepare the child document with updated metadata
            child_doc = Document(page_content=sub_doc.page_content, metadata=child_metadata)
            child_chunks.append(child_doc)
            doc_chunk_ids.append(chunk_id)
        # Use chunk_docs_retriever's add_documents method for adding child documents to the stores
    if child_chunks:
        chunk_docs_retriever.vectorstore.add_documents(child_chunks)
        chunk_docs_retriever.docstore.mset(parents_redis)
        logging.info(f"Added {len(parents_redis)} parent chunks for document ID {doc_id}")
        save_in_memory_store_to_redis(chunk_docs_retriever.docstore, prefix=prefix)
        logging.info(f"Updated {len(child_chunks)} chunks for document ID {doc_id}")
    logging.info(f"Vectorstore chunk_docs collection count: {chunk_docs_retriever.vectorstore._collection.count()}")


def process_sharepoint_sites(docs_collection, pages_collection, redis_store, siteNames):
    loader_map = {
        (".pptx", ".ppt"): (UnstructuredPowerPointLoader, {"mode": "single", "strategy": "hi_res"}),
        (".pdf",): [(UnstructuredPDFLoader, {"mode": "single", "strategy": "hi_res"}), (PyPDFLoader, {})],
        (".docx",): [(UnstructuredWordDocumentLoader, {"mode": "single", "strategy": "hi_res"}), (Docx2txtLoader, {})],
        (".md",): (UnstructuredMarkdownLoader, {"mode": "single", "strategy": "hi_res"}),
        (".xlsx", ".xlsm"): (UnstructuredExcelLoader, {"mode": "single", "strategy": "hi_res"}),
        (".txt",): (TextLoader, {"autodetect_encoding": True}),
    }

    redis_ids_chunks = list(redis_store.yield_keys(prefix="document_2000:*"))
    logging.info(f"Redis ids chunks: {len(redis_ids_chunks)}")
    redis_ids_chunks = None
    redis_ids_full_docs = list(redis_store.yield_keys(prefix="full_document:*"))
    logging.info(f"Redis ids full docs: {len(redis_ids_full_docs)}")

    # directory_paths = [os.path.join(scrape_folder, folder, site) for folder in ['site_files', 'site_pages'] for site in siteNames]
    directory_paths = [os.path.join(scrape_folder, folder, site) for folder in ['site_pages'] for site in siteNames]
    # directory_paths = []
    directory_paths.extend(
        [os.path.join(scrape_folder, folder, site) for folder in ['site_files'] for site in ["EmployeePortal"]])
    for directory_path in directory_paths:
        logging.info(f"Processing directory: {directory_path}")
        all_file_paths = []
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                all_file_paths.append((file_path, filename))

        for file_path, filename in tqdm(all_file_paths, desc="Processing files"):
            logging.info(f"Processing file: {file_path}")
            file_extension = os.path.splitext(filename)[1].lower()

            for extensions, loader_info in loader_map.items():
                if file_extension in extensions:
                    if not isinstance(loader_info, list):
                        loader_info = [loader_info]

                    for loader_class, loader_kwargs in loader_info:
                        try:
                            loader = loader_class(file_path, **loader_kwargs)
                            loaded_documents = loader.load()

                            if isinstance(loaded_documents, list) and len(loaded_documents) > 1:
                                merged_page_content = ""
                                merged_metadata = {}
                                for doc in loaded_documents:
                                    merged_page_content += doc.page_content
                                    merged_metadata.update(doc.metadata)
                                loaded_documents = [Document(page_content=merged_page_content, metadata=merged_metadata)]

                            if loaded_documents:
                                logging.info(f"Loaded {len(loaded_documents)} documents from {file_path}")
                                for doc in loaded_documents:
                                    process_and_add_document(doc, redis_ids_full_docs, docs_collection, pages_collection, redis_client)
                            break
                        except Exception as e:
                            logging.error(f"Error loading {filename}: {str(e.with_traceback(None))}")
                            with open(failed_docs_log_path, 'a') as f:
                                # Write each failed document path to the file
                                f.write(f"{filename}\n")
                    break
            else:
                logging.warning(f"Skipping file: {file_path}")


def delete_redis_keys_with_prefix(redis_client, prefixes):
    for prefix in prefixes:
        for key in redis_client.scan_iter(match=prefix):
            redis_client.delete(key)
    redis_client.close()


def delete_chroma_collections(chroma: chromadb.HttpClient, collections):
    for collection in collections:
        try:
            chroma.delete_collection(collection)
        except Exception as e:
            logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    # Create f'{scrape_folder}/sharepoint_ingestion.log' file if it doesn't exist
    # Ensure the scrape_folder exists
    if not os.path.exists(scrape_folder):
        os.makedirs(scrape_folder)

    # Now, safely create or open the file
    log_file_path = os.path.join(scrape_folder, 'sharepoint_ingestion.log')
    open(log_file_path, 'w').close()

    failed_docs_log_path = os.path.join(scrape_folder, 'failed_documents.log')
    with open(failed_docs_log_path, 'w') as f:
        f.write("")
        f.close()
    # Configure logging
    logging.basicConfig(filename=f'{scrape_folder}/sharepoint_ingestion.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.info(f"CUDA Availability: {torch.cuda.is_available()}")
    # MongoDB connection setup
    client = MongoClient(mongo_uri)
    db = client[db_name]
    try:
        client.server_info()
        logging.info("Connection to MongoDB successful")
    except pymongo_errors.PyMongoError as e:
        logging.error(f"Connection to MongoDB failed: {e}")
        exit(1)

    docs_collection = db[mongo_documents_collection_name]
    pages_collection = db[mongo_pages_collection_name]
    # ChromaDB connection setup
    chroma_persistent_client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
    heartbeat = chroma_persistent_client.heartbeat()
    if heartbeat == 0 or heartbeat is None:
        logging.error("Connection to ChromaDB failed")
        exit(1)
    else:
        logging.info("Connection to ChromaDB successful")

    # Redis client connection setup
    redis_client = create_redis_client(redis_host, int(redis_port), decode_responses=True)
    # Perform a health check
    is_healthy = check_redis_health(redis_client)
    if is_healthy:
        logging.info("Redis is healthy.")
    else:
        logging.error("Redis is not healthy.")
        exit(1)

    # delete_chroma_collections(chroma_persistent_client, [chroma_collection_full_docs, chroma_collection_chunks])
    # delete_redis_keys_with_prefix(redis_client, ["document_2000:*", "full_document:*"])

    try:
        chroma_persistent_client.create_collection(chroma_collection_full_docs)
        chroma_persistent_client.create_collection(chroma_collection_chunks)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    redis_client = create_redis_client(redis_host, int(redis_port), decode_responses=False)
    embedding_function = SentenceTransformerEmbeddings(model_name='intfloat/multilingual-e5-large-instruct',
                                                       model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {
                                                           'device': 'cpu'},
                                                       encode_kwargs={'normalize_embeddings': True})
    redis_store = RedisStore(client=redis_client)
    # Define the sites to ingest
    siteNames = ["asd", "ddd"]
    process_sharepoint_sites(docs_collection, pages_collection, redis_store, siteNames)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    logging.info(f"Ingestion complete. Total execution time: {execution_time} seconds")
