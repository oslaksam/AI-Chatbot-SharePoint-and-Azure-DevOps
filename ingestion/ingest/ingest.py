import json
import logging
import shutil
import time
import os
import redis
import chardet
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Redis
from helpers import *
from redis.commands.search.query import Query

# Locate a `.env` file and load contents into the environment
load_dotenv()
data_path = '/data'
filtered_ado_files_path = '/data/clean/clean'

# logger config
# logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logging level

# Create a file handler that logs messages to a file
file_handler = logging.FileHandler(f'{data_path}/ingestion.log')
file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler

# Create a console handler that logs messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

# Create a formatter
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def is_index_empty(redis_client, index_name):
    try:
        # Create a query to search for all documents but retrieve none (paging set to 0, 0)
        query = Query("*").paging(0, 0)

        # Perform the search on the specified index
        result = redis_client.ft(index_name).search(query)

        # Check if the total count of documents is 0
        return result.total == 0

    except UnicodeDecodeError:
        # Handle the decoding error
        logger.error(f"Decoding error encountered when checking index: {index_name}")
        return False

    except Exception as e:
        # Handle other potential exceptions
        logger.error(f"Error encountered when checking index: {index_name}: {e}")
        return False



def document_key_contains_url(redis_client, url):
    """
    This function returns the keys which contain the given URL in their values.
    """
    for key in redis_client.scan_iter("doc:*"):
        value = redis_client.hget(key, "url")
        if value == url:
            yield key


def document_key_contains_url(redis_client, url, redis_index_name):
    """
    This function returns the keys which contain the given URL in their values.
    """
    for key in redis_client.scan_iter(f"doc:{redis_index_name}:*"):
        value = redis_client.hget(key, "url")
        if value == url:
            yield key


# Function to update the set of URLs in Redis
def cleanup_redis_entries(redis_client, new_urls, urls_index_key):
    logger.info("Starting cleanup of Redis entries...")

    # Fetch the set of existing URLs from Redis; if none, it will be an empty set
    existing_urls = set(redis_client.smembers(urls_index_key))

    # Find URLs that need to be removed (they are in Redis but not in the new set of URLs)
    stale_urls = existing_urls.difference(new_urls)
    logger.info(f"Found {len(stale_urls)} stale URLs to remove.")

    # Iterate over stale URLs and remove associated documents
    for url in stale_urls:
        removed_count = 0
        # Find keys with documents containing the stale URL
        for key in redis_client.scan_iter(match=f"doc:*"):
            if redis_client.hget(key, 'url') == url:
                redis_client.delete(key)
                logger.debug(f"Removed document with key: {key}")
                removed_count += 1
        logger.info(f"Removed {removed_count} documents associated with URL: {url}")

        # Remove the URL from the set of all URLs
        redis_client.srem(urls_index_key, url)
        logger.info(f"Removed URL from the index: {url}")

    # Add new URLs to the Redis set that are not already present
    new_to_add = new_urls.difference(existing_urls)
    if new_to_add:
        redis_client.sadd(urls_index_key, *new_to_add)
        logger.info(f"Added {len(new_to_add)} new URLs to the index.")

    logger.info("Cleanup process completed.")


def remove_duplicates(redis_client):
    logger.info("Starting duplicate removal...")
    count = 0
    url_to_docs = {}
    # Step 1: Fetch all document keys and group by URL
    for key in redis_client.scan_iter("doc:*"):
        url = redis_client.hget(key, "url")
        content = redis_client.hget(key, "content")
        if url not in url_to_docs:
            url_to_docs[url] = {}
        # Use content as the key to identify duplicates
        if content in url_to_docs[url]:
            url_to_docs[url][content].append(key)
        else:
            url_to_docs[url][content] = [key]

    # Step 2: Identify and remove duplicates
    for url, contents in url_to_docs.items():
        for content, keys in contents.items():
            # Keep the first entry and remove the rest
            for key in keys[1:]:
                redis_client.delete(key)
                count += 1
                logger.info(f"Duplicate removed: {key}")
    logger.info(f"Removed {count} duplicates.")


def index_exists(redis_client, index_name):
    """
    Check if the given index name exists in the Redisearch indexes.
    """
    existing_indexes = redis_client.execute_command("FT._LIST")
    return index_name in existing_indexes


def delete_index(redis_client, index_name):
    """
    Delete a given index.
    """
    redis_client.execute_command("FT.DROPINDEX", index_name)


def main():
    """
    Main invocation function.
    """
    logger.info("Starting redis and openai services...")
    # STEP 0: Initialize all services
    embedding_model = "text-embedding-ada-002"
    embedding_function = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_version="2023-05-15",
        openai_api_type="azure",
        deployment=embedding_model,
    )
    redis_index_name = "TEMPLATE Wiki"
    # The name of the index where all URLs will be stored
    urls_index_key = "all_urls_set"
    m = hashlib.md5()  # this will convert URL into unique ID
    new_urls = set()

    # Need to export REDIS_URL to avoid error in subsequent calls to Redis()
    os.environ["REDIS_URL"] = f"redis://{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}"

    vector_store = Redis(
        redis_url=os.environ["REDIS_URL"],
        index_name=redis_index_name,
        embedding=embedding_function
    )
    vector_store._create_index()
    # split REDIS_URL into host and port
    redis_host, redis_port = os.environ["REDIS_URL"].split("redis://")[1].split(":")
    redis_client = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )

    text_splitter = MarkdownTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base")

    # Get all files from GitHub
    github_client = Github(os.environ["GITHUB_ACCESS_TOKEN"])
    logger.info("Getting all files from github")
    files = get_all_github_files(
        github_client,
        "template/wikijs-content",
        paths_to_ignore=[
            "private",
        ],
    )
    logger.info(f"Found {len(files)} files in github TEMPLATE Wiki")

    # STEP 4: Add all files to vector store
    logger.info("Splitting documents")
    documents = []

    for f in files:
        content = f.decoded_content.decode("utf-8")
        # Replace the original size filter with the new logic from script2
        if not is_acceptable_file_type(f.name):
            logger.info(f"Skipping {f.path} because it is not an accepted file type.")
            continue

        # Check if the file only contains specified syntax
        if contains_only_specified_syntax(content):
            logger.info(f"Skipping {f.path} due to containing only specified syntax.")
            continue

        # Check for FLAG_TERMS in file name or path
        if should_flag_file(f.path, FLAG_TERMS):
            logger.info(f"Skipping {f.path} due to containing a flag term.")
            continue

        # Check for documents below the word threshold
        if is_below_word_threshold(content, 10):
            logger.info(f"Skipping {f.path} for being below the word threshold.")
            continue

        # Check for mostly TODO content
        if is_mostly_todo(content):
            logger.info(f"Skipping {f.path} as mostly TODO.")
            continue

        url = f"https://wiki.TEMPLATE.cloud/en/{f.path}"
        # Add a url to later delete old documents
        new_urls.add(url)
        new_filehash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        # Fetch all keys that have the URL in their values
        existing_keys_with_url = list(document_key_contains_url(redis_client, url, redis_index_name))

        if existing_keys_with_url:
            # Check the file hash of the existing document
            existing_filehash = redis_client.hget(existing_keys_with_url[0], "filehash")
            if existing_filehash == new_filehash:
                logger.info(f"File {f.path} has not changed. Skipping update.")
                continue
            else:
                # File has changed, remove all existing keys
                logger.info(f"File {f.path} has changed. Removing old documents.")
                for key in existing_keys_with_url:
                    redis_client.delete(key)
                    logger.debug(f"Removed document with key: {key}")

        # create a unique id for each document
        m.update(url.encode("utf-8"))
        uid = m.hexdigest()[:12]

        for idx, chunk in enumerate(text_splitter.split_text(content)):
            index = f'{uid}-{idx}'
            logger.info(f"Adding chunk {index} of {f.path} ...")
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "name": f.name,
                        "path": f.path,
                        "source": redis_index_name,
                        "chunk_id": index,
                        "url": url,
                        "filehash": new_filehash,
                    },
                )
            )
            logger.info("Done")
        logger.info(f"Done adding {f.path}")

    # STEP 5: Update documents in vector store
    documents_number = len(documents)
    logger.info(f"{documents_number} new documents")
    if documents_number > 0:
        logger.info("Adding documents to vector store...")
        vector_store.add_documents(documents=documents)
        logger.info(f"Added/Updated {len(documents)} documents to vector store for project: {redis_index_name}")

    # STEP 6: Add azure DevOps content content
    logger.info("Starting processing Azure Devops files...")
    # Load global mapping from local directory
    with open(f'{data_path}/global_mapping.json', 'r', encoding="utf-8") as f:
        global_mapping = json.load(f)

    # Get unique project names from the directory structure
    project_names = set([dir.split('/')[1] for dir in global_mapping.keys() if dir.startswith(('WIKI/', 'CODE/'))])
    # print all project names in a loop
    logger.info("Found the following projects:")
    for project_name in project_names:
        logger.info(project_name)

    for project_name in project_names:
        vector_store = Redis(
            redis_url=os.environ["REDIS_URL"],
            index_name=project_name,
            embedding=embedding_function
        )

        documents = []
        # Check both 'WIKI' and 'CODE' directories for each project
        for primary_dir in ['WIKI', 'CODE']:
            project_dir = os.path.join(filtered_ado_files_path, primary_dir, project_name)

            if not os.path.exists(project_dir):
                continue

            # Walk through each primary directory and its subdirectories
            for dirpath, _, filenames in os.walk(project_dir):
                for filename in filenames:
                    if filename.endswith('.md'):
                        file_path = os.path.join(dirpath, filename)

                        try:
                            with open(file_path, 'rb') as file:
                                rawdata = file.read()
                                result = chardet.detect(rawdata)
                                charenc = result['encoding']

                            with open(file_path, 'r', encoding=charenc) as file:
                                content = file.read()

                            relative_path = os.path.relpath(file_path, filtered_ado_files_path).replace(os.sep, '/')
                            # remove filtered_ado_files_path from the relative path
                            url = global_mapping.get(relative_path, {}).get('url', '')
                            new_urls.add(url)  # Add URL for later cleanup
                            loaded_hash = global_mapping.get(relative_path, {}).get('hash', '')
                            # new_filehash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                            # if loaded_hash != new_filehash:
                            #     logger.warning(
                            #         f"Loaded and actual hash mismatch for file {relative_path} LOADED HASH: {loaded_hash} NEW HASH: {new_filehash} URL: {url}")
                            # Check if the file hash has changed or if it's a new file
                            existing_keys_with_url = list(
                                document_key_contains_url(redis_client, url, project_name))
                            if existing_keys_with_url:
                                # Fetch the existing hash for comparison
                                existing_filehash = redis_client.hget(existing_keys_with_url[0], "filehash")
                                if existing_filehash == loaded_hash:
                                    logger.info(f"File {relative_path} has not changed. Skipping update.")
                                    continue
                                else:
                                    # Hash is different, proceed with updating
                                    logger.info(f"File {relative_path} has changed. Updating documents.")
                                    for key in existing_keys_with_url:
                                        redis_client.delete(key)
                                        logger.debug(f"Removed document with key: {key}")

                            # Create a unique id for each document
                            m.update(url.encode("utf-8"))
                            uid = m.hexdigest()[:12]

                            # Add document chunks to Redis
                            for idx, chunk in enumerate(text_splitter.split_text(content)):
                                index = f'{uid}-{idx}'
                                logger.info(f"Adding/updating chunk {index} of {relative_path} ...")
                                documents.append(Document(
                                    page_content=chunk,
                                    metadata={
                                        "name": filename,
                                        "path": relative_path,
                                        "source": primary_dir + '-' + project_name,
                                        "chunk_id": index,
                                        "url": url,
                                        "filehash": loaded_hash,
                                    }
                                ))
                            logger.info(f"Added/Updated {relative_path}")

                        except Exception as e:
                            logger.error(f"Failed to process file {relative_path}. Error: {e}")
        documents_number = len(documents)
        logger.info(f"{documents_number} new documents")
        if documents_number > 0:
            # Check if the index already exists
            if not index_exists(redis_client, project_name):
                logger.info(f"Creating index {project_name} using _create_index()")
                vector_store.index_name = project_name
                vector_store._create_index()  # Call this only if the index doesn't exist
            logger.info("Adding documents to vector store...")
            vector_store.add_documents(documents=documents)
            logger.info(f"Added/Updated {len(documents)} documents to vector store for project: {project_name}")
        else:
            logger.warning(f"No documents to add/update for project: {project_name}")
    # At the end of processing all files, cleanup old entries and update the URLs index if new_urls is not empty an
    # empty set cleanup redis entries will remove all documents that are not in the new_urls set
    if new_urls:
        cleanup_redis_entries(redis_client, new_urls, urls_index_key)
    remove_duplicates(redis_client)
    # Remove empty indexes
    existing_indexes = redis_client.execute_command("FT._LIST")
    # Iterate over each index and check if it is empty
    for index_name in existing_indexes:
        if is_index_empty(redis_client, index_name):
            delete_index(redis_client, index_name)
            logger.info(f"Deleted empty index: {index_name}")
    # Remove flags empty files created by previous scripts scrape_complete and filter_complete if they exist
    if os.path.exists(f'{data_path}/scrape_complete'):
        os.remove(f'{data_path}/scrape_complete')
        logger.info("Removed scrape_complete flag file")
    if os.path.exists(f'{data_path}/filter_complete'):
        os.remove(f'{data_path}/filter_complete')
        logger.info("Removed filter_complete flag file")
    if os.path.exists(f'{data_path}/CODE'):
        shutil.rmtree(f'{data_path}/CODE')
        logger.info("Removed CODE directory")
    if os.path.exists(f'{data_path}/WIKI'):
        shutil.rmtree(f'{data_path}/WIKI')
        logger.info("Removed WIKI directory")


if __name__ == "__main__":
    # If there is an environmental variable. Called skip. Then skip running the main function.
    if os.environ.get("SKIP_EXECUTION", "false").lower() == "true":
        logger.info('Skipping execution.')
    else:
        start_time = time.time()
        main()
        end_time = time.time()
        logger.info(f"Execution time: {end_time - start_time} seconds")
