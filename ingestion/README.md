# GitHub and Azure DevOps Content Processor

## Overview

The Python scripts in the `src`  processes content from GitHub repositories and Azure DevOps content directories. It leverages OpenAI embeddings to create vector representations and stores them in a Redis vector store, enabling advanced searching and clustering operations.

## How It Works

### 1. Configuration and Initialization:

- The script starts by loading environment variables from a `.env` file, including OpenAI API keys and Redis connection details.
- A logger is configured to capture essential information, with both file and console handlers set at the INFO level.

### 2. Utility Functions:

#### `get_all_github_files(client: Github, repo_name: str, paths_to_ignore: list[str]) -> list[ContentFile]`
Fetches all markdown files from a specified GitHub repository, excluding files in `paths_to_ignore`.

#### `is_acceptable_file_type(filename: str) -> bool`
Checks if a file is of an accepted markdown type.

#### `should_flag_file(file_path: str, flag_terms: list[str]) -> bool`
Determines if a file should be flagged based on specific terms in its path.

#### `calculate_hash(content: str) -> str`
Computes a SHA-256 hash of the file content.

#### `is_mostly_todo(content: str) -> bool`
Evaluates if the majority of the file content consists of TODO-like keywords.

#### `is_below_word_threshold(content: str, threshold: int) -> bool`
Checks if the content is below a specified word count threshold.

#### `contains_only_specified_syntax(content: str) -> bool`
Determines if the content contains only specific Markdown syntax (like `_TOC_` or `_TOSP_`).

### 3. Main Process:

#### Initialization:
- Sets up OpenAI embeddings, Redis vector store, and a text splitter for markdown files.
- Initializes the GitHub client using the provided access token.

#### GitHub Content Processing:
- Retrieves markdown files from GitHub, applying various checks (file type, specific syntax, flagged terms, word threshold, and TODO content).
- For each valid file, the content is split into chunks and processed.
- Each chunk is added to the Redis vector store, with metadata including the file name, path, source, chunk ID, URL, and content hash.

#### Azure DevOps Content Processing:
- Reads a global mapping file to determine project names and related URLs.
- Processes markdown files from Azure DevOps 'WIKI' and 'CODE' directories.
- Each valid file is similarly split, processed, and added to the Redis vector store.

#### Cleanup and Deduplication:
- Old entries in Redis are cleaned up based on the new set of URLs.
- Duplicate documents in Redis are identified and removed.
- Checks for and deletes empty indexes in Redis.

#### Final Steps:
- Removes flag files indicating completion of previous scraping and filtering processes.
- Cleans up temporary directories used for processing.

### 4. Execution:

The script can be executed directly, with an optional environment variable `SKIP_EXECUTION` to bypass the main function for testing purposes.

# Ingestion

This module handles ingestion from wiki to a vector store.
Currently we support Azure DevOps, TEMPLATE wiki, using the Redis Vectorstore.

## Quick start

### Docker compose

```shell
docker compose ingestion up --build 
```

### Docker

```shell
make redis-run-docker
make ingestion-build-image
make ingestion-run-docker
```

### Local

Make sure you have a clean python3 virtualenv created and activated

```shell
make ingestion-scripts-install-requirements
make ingestion-scripts-run
```

