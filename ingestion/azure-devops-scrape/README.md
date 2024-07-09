
# Azure Downloader Script

## Overview
This script is designed to interact with Azure DevOps, specifically to fetch and download Markdown files from both repositories and wikis within the platform. It optimizes its operations by maintaining a mapping of previously downloaded files, ensuring that only new or updated files are fetched.

## How It Works

### 1. Configuration
The script is configured with a predefined organization name, a personal access token, and necessary headers for making API requests. It also specifies output directories for the downloaded Markdown files.

### 2. Utility Functions:

#### `safe_request(url, headers)`
Safely sends HTTP requests to specified URLs. If there's an issue with the request, the function logs the error and avoids crashing the script.

#### `calculate_hash(content)`
Computes the SHA-256 hash of the given content. This hash is used to determine whether a file's content has changed and needs to be re-downloaded.

#### `download_repo_file(...)`
Handles the downloading of specific files from Azure DevOps repositories. It also checks if the file has been previously downloaded or if its content has changed by comparing its hash value.

#### `get_wiki_content(...)`
Fetches content from a specific Azure DevOps wiki page.

#### `sanitize_filename(name)`
Ensures that the filename is valid by replacing any characters that might cause issues on the filesystem.

#### `save_to_markdown(...)`
Stores the provided content as a markdown file. Before saving, it verifies if the content has changed by comparing hashes, ensuring that unchanged files are not overwritten.

#### `process_wiki_pages(...)`
Recursively processes and downloads content from Azure DevOps wiki pages.

#### `process_wikis()`
Iterates through all Azure DevOps projects, accessing each of their wikis and processing them using the `process_wiki_pages()` function.

#### `process_repositories()`
Navigates through all Azure DevOps repositories, downloading markdown files found within.

### 3. Main Execution:

In the `main()` function, the script executes the following steps:

- It begins by processing all repositories to fetch markdown files from them.
- Subsequently, it processes and downloads content from all wikis.
- After downloading, a global mapping of all processed files, including their URLs and content hashes, is created and saved in `global_mapping.json`. This file helps in optimizing future runs of the script by avoiding re-downloading unchanged files.
  
Lastly, the script logs the total time taken for its execution.

## Purpose
The primary objective of this script is to automate the process of fetching and downloading Markdown files from Azure DevOps. By maintaining a mapping of previously downloaded files, the script ensures efficiency and avoids redundant operations.
