# SharePoint Ingestion Script Documentation

## Overview

This Python script is designed to process and ingest documents from SharePoint into a structured storage and retrieval system. It utilizes MongoDB, Redis, and ChromaDB for data persistence, indexing, and retrieval. The script is capable of handling various document formats, including PDFs, Word documents, PowerPoint presentations, Excel sheets, and Markdown files. The core functionality includes document loading, metadata extraction and updating, content filtering, chunking for efficient storage, and embedding for future retrieval.

### Dependencies

- `re`
- `json`
- `logging`
- `os`
- `time`
- `uuid`
- `chromadb`
- `redis`
- `torch`
- `dotenv`
- `langchain.retrievers`
- `langchain.storage`
- `langchain.text_splitter`
- `langchain_community.document_loaders`
- `langchain_community.embeddings`
- `langchain_community.storage`
- `langchain_community.vectorstores`
- `langchain_core.documents`
- `pymongo`
- `tqdm`

## Components

### Document Filters

#### `DocumentFilter`

- Abstract base class for defining document filters.
- Method: `apply(doc)` - Determines whether a document passes the filter criteria.

#### `CompositeFilter`

- Inherits from `DocumentFilter`.
- Supports the combination of multiple filter objects.
- Method: `add_filter(filter)` - Adds a filter to the composite filter.
- Overrides `apply(doc)` - Returns `True` if the document passes all filters.

#### Specific Filters

- `HomepageFilter`, `NewsFilter`, `LocationsFilter`
- Each implements a specific filtering logic based on document content and metadata.

### Document Processing and Storage

#### MongoDB and Redis Integration

- Functions for updating document metadata from MongoDB and for saving documents and their chunks to Redis.
- Methods include `update_doc_metadata_from_mongodb`, `save_document_to_redis`, `save_chunk_to_redis`, etc.

#### Document Chunking and Retrieval

- Utilizes `langchain` libraries for document splitting and chunk retrieval.
- `create_chunks_retriever` and `create_full_docs_retriever` configure retrievers for document chunking.
- Supports hierarchical document splitting with custom separators tailored for Markdown documents.

### Environment Configuration

- Loads environment variables for configuring database connections, collection names, and document processing parameters.

### Main Ingestion Flow

The script follows a structured sequence to ingest documents from SharePoint into a structured data format suitable for indexing, searching, and retrieval through MongoDB, Redis, and ChromaDB. The process is as follows:

1. **Initialization**:
   - **Logging Setup**: Initializes a logging system to capture the operational logs of the script. This includes setting up a file-based logging mechanism that records the activities, warnings, and errors encountered during the execution.
   - **Environment Configuration**: Loads environment variables from a `.env` file. These variables configure database connections (MongoDB, Redis, ChromaDB), collection names, and other operational parameters like directories for scraping and log file locations.

2. **Database and Store Connections**:
   - **MongoDB**: Establishes a connection to MongoDB using the credentials and connection string specified in the environment variables. The script checks the connection by attempting to retrieve server information.
   - **Redis**: Initializes a Redis client and conducts a health check to ensure the Redis service is responsive.
   - **ChromaDB**: Connects to ChromaDB for embedding storage and retrieval, verifying the connection through a heartbeat check.

3. **Preparation for Ingestion**:
   - **Clearing Existing Data** (optional): The script may include steps (commented out in the provided script) to clear existing data from Redis and ChromaDB collections. This is critical when a fresh start is needed.
   - **Collection Creation in ChromaDB**: Ensures the necessary collections for storing document embeddings are available in ChromaDB.

4. **Document Loading and Processing**:
   - **Directory Traversal**: Iterates over the directories specified for SharePoint sites. It looks for documents within the `site_pages` and `site_files` directories, targeting specific sites as outlined in the `siteNames` list.
   - **Document Loading**: For each file discovered, the script determines the document's format and selects the appropriate loader from a predefined map. This map associates file extensions with document loaders capable of processing and extracting content from various file types (PDF, Word, PowerPoint, etc.).
   - **Document Processing**: Each loaded document undergoes a series of processing steps:
     - **Metadata Updating**: The script attempts to update the document's metadata from MongoDB, ensuring that the most current metadata is associated with each document.
     - **Filtering**: Utilizes a composite filter that aggregates several specific filters (e.g., `HomepageFilter`, `NewsFilter`, `LocationsFilter`) to determine if the document should proceed in the pipeline. Documents failing to pass these filters are logged and excluded from further processing.
     - **Chunking and Retrieval Configuration**: For documents passing the filters, the script configures retrievers for both full documents and chunks, leveraging the `ParentDocumentRetriever` class from LangChain. This step involves splitting documents into manageable pieces for efficient storage and retrieval.
     - **Storage in Redis and ChromaDB**: Both the full documents and their respective chunks are stored in Redis. Concurrently, document embeddings generated during the chunking process are stored in ChromaDB, facilitating efficient future retrieval through semantic similarity searches.

5. **Post-processing and Cleanup**:
   - **Logging**: Upon processing all documents, the script logs the total execution time, providing insights into the efficiency of the document ingestion process.
   - **Resource Release**: Ensures that all database connections and resources are appropriately closed or released, preventing potential memory leaks or connection issues.


### Utility Functions

- Functions to check Redis health, delete keys with specific prefixes, and manage ChromaDB collections.

## Execution

The script's execution begins by setting up necessary resources and then processing documents from specified SharePoint sites. It concludes by logging the total execution time. Document processing involves updating metadata, filtering, chunking, and saving both metadata and chunks into Redis and embeddings into ChromaDB.
