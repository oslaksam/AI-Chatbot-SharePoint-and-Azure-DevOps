import json
import logging
import os
import re
import time

import requests
from datetime import datetime, timedelta
import msal
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import html2text
from pymongo.errors import PyMongoError
import base64
from io import BytesIO
import html

from requests.adapters import HTTPAdapter
from urllib3 import Retry

# Load environment variables
load_dotenv()

# Global settings
graphURI = os.getenv('GRAPH_URI')
tenantID = os.getenv('TENANT_ID')
authority = os.getenv('AUTHORITY') + tenantID
clientID = os.getenv('CLIENT_ID')
scope = [os.getenv('SCOPE')]
thumbprint = os.getenv('THUMBPRINT')
cert_file_base64 = os.getenv('CERT_FILE_BASE64')
scrape_folder = os.getenv('SCRAPE_FOLDER_PATH', "/data")

# MongoDB setup
username = os.getenv('MONGO_INITDB_ROOT_USERNAME')
password = os.getenv('MONGO_INITDB_ROOT_PASSWORD')
mongo_host = os.getenv('MONGO_HOST', 'host.docker.internal')
mongo_port = os.getenv('MONGO_PORT', '27017')
mongo_uri = f"mongodb://{username}:{password}@{mongo_host}:{mongo_port}"
db_name = os.getenv('MONGO_DB_NAME', 'sharepoint')
documents_collection_name = os.getenv('MONGO_DOCUMENTS_COLLECTION_NAME', 'documentLibrary')
pages_collection_name = os.getenv('MONGO_PAGES_COLLECTION_NAME', 'pages')

# Create f'{scrape_folder}/sharepoint_scrape.log' file if it doesn't exist
# Ensure the scrape_folder exists
if not os.path.exists(scrape_folder):
    os.makedirs(scrape_folder)

# Now, safely create or open the file
log_file_path = os.path.join(scrape_folder, 'sharepoint_scrape.log')
open(log_file_path, 'a').close()
# Configure logging
logging.basicConfig(filename=f'{scrape_folder}/sharepoint_scrape.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# MongoDB connection
client = MongoClient(mongo_uri)
db = client[db_name]
documents_collection = db[documents_collection_name]
pages_collection = db[pages_collection_name]

# Authentication and token management
accessToken = None
requestHeaders = None
tokenExpiry = None


def msal_certificate_auth():
    global accessToken, tokenExpiry, requestHeaders
    # Decode the base64-encoded certificate
    certfile_bytes = base64.b64decode(cert_file_base64)
    # Use a BytesIO object as a file-like object for the certificate
    certfile_in_memory = BytesIO(certfile_bytes)

    app = msal.ConfidentialClientApplication(clientID, authority=authority,
                                             client_credential={"thumbprint": thumbprint,
                                                                "private_key": certfile_in_memory.read()})
    result = app.acquire_token_for_client(scopes=scope)
    if 'access_token' in result and 'expires_in' in result:
        accessToken = result['access_token']
        expiresIn = result['expires_in']  # Time in seconds until the token expires
        tokenExpiry = datetime.now() + timedelta(seconds=expiresIn)
        logging.info("New access token acquired")
        requestHeaders = {'Authorization': f'Bearer {accessToken}'}
    else:
        logging.error("Failed to acquire access token.")


def refresh_token_if_needed():
    global accessToken, tokenExpiry
    if datetime.now() >= tokenExpiry:
        logging.info("Access token expired, refreshing...")
        msal_certificate_auth()
    else:
        logging.info("Access token is still valid.")


def requests_retry_session(
        retries=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
        session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


failed_resources = []  # Global list to track unavailable resources


def msgraph_request(resource, requestHeaders):
    refresh_token_if_needed()
    try:
        response = requests_retry_session().get(resource, headers=requestHeaders, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error for {resource}: {errt}")
    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error for {resource}: {errh}")
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Error Connecting to {resource}: {errc}")
    except requests.exceptions.RequestException as err:
        logging.error(f"Other Error for {resource}: {err}")
    failed_resources.append(resource)
    return {}  # Return an empty dictionary to avoid AttributeError in the caller


def sanitize_name(name):
    name_part, _, extension = name.rpartition('.')
    sanitized_name_part = re.sub(r'\W+', '_', name_part)
    return f"{sanitized_name_part}.{extension}" if extension else sanitized_name_part


def sanitize_path_element(path_element):
    return re.sub(r'[\\/*?:"<>|]', '_', path_element)


def get_site(siteName, tenantName):
    site_response = msgraph_request(f"{graphURI}/beta/sites/{tenantName}.sharepoint.com:/sites/{siteName}",
                                    requestHeaders)
    return site_response.get('id')


def list_site_pages(site_id):
    pages_response = msgraph_request(f"{graphURI}/beta/sites/{site_id}/pages", requestHeaders)
    return pages_response.get('value', [])


def get_page_content(site_id, page_id):
    page_response = msgraph_request(
        f"{graphURI}/beta/sites/{site_id}/pages/{page_id}/microsoft.graph.sitePage?expand=canvasLayout", requestHeaders)
    return page_response


def get_page_webparts(site_id, page_id):
    webparts_response = msgraph_request(
        f"{graphURI}/beta/sites/{site_id}/pages/{page_id}/microsoft.graph.sitePage/webparts",
        requestHeaders)
    return webparts_response


def update_document_if_changed(collection, document_id, new_data):
    existing_doc = collection.find_one({"_id": document_id})
    if existing_doc:
        # Compare relevant fields; this might need adjustment based on your data structure
        needs_update = any(existing_doc.get(key) != new_data.get(key)
                           for key in new_data if key != "_id")
        if needs_update:
            result = collection.update_one({"_id": document_id}, {"$set": new_data}, upsert=True)
            logging.info(
                f"Document updated, matched_count: {result.matched_count}, modified_count: {result.modified_count}")
        else:
            logging.info("No update needed as the document has not changed.")
    else:
        # If document does not exist, insert new document
        result = collection.insert_one(new_data)
        logging.info(f"New document inserted, id: {result.inserted_id}")


def save_page_metadata_to_mongo(page_metadata, siteName, collection):
    try:
        # Extracting createdBy and lastModifiedBy information
        created_by = page_metadata.get("createdBy", {}).get("user", {}).get("displayName", "Unknown")
        created_by_email = page_metadata.get("createdBy", {}).get("user", {}).get("email", "Unknown")
        last_modified_by = page_metadata.get("lastModifiedBy", {}).get("user", {}).get("displayName", "Unknown")
        last_modified_by_email = page_metadata.get("lastModifiedBy", {}).get("user", {}).get("email", "Unknown")
        document_id = page_metadata.get("id", "Unknown ID")

        # Constructing the document to insert/update
        document = {
            "siteName": siteName,
            "pageName": page_metadata.get("name", "Unknown Page"),
            "pageUrl": page_metadata.get("webUrl", "Unknown URL"),
            "eTag": page_metadata.get("eTag", "Unknown eTag"),
            "lastModifiedDateTime": page_metadata.get("lastModifiedDateTime", "Unknown Last Modified Date"),
            "description": page_metadata.get("description", "No Description"),
            "createdBy": {"name": created_by, "email": created_by_email},
            "lastModifiedBy": {"name": last_modified_by, "email": last_modified_by_email},
            "title": page_metadata.get("title", "No Title"),
            "pageLayout": page_metadata.get("pageLayout", "Unknown Layout"),
            # Add more fields as needed
        }

        # Use the document ID as the filter for the update operation
        result = collection.update_one(
            {"_id": document_id},
            {"$set": document},
            upsert=True
        )
        # Logging the outcome
        # Use the new function to update or insert the document
        update_document_if_changed(collection, document.get("_id"), document)
        logging.info(
            f"Page metadata saved to MongoDB, upserted_id: {result.upserted_id}, matched_count: {result.matched_count}, modified_count: {result.modified_count}")
        return document_id
    except pymongo.errors.PyMongoError as e:
        logging.error(f"Failed to save page metadata to MongoDB: {e}")


def save_document_metadata_to_mongo(document_metadata, siteName, siteId, file_path):
    try:
        # Construct document metadata to insert/update
        metadata_document = {
            "_id": document_metadata.get("id", "Unknown ID"),
            "eTag": document_metadata.get("eTag", "Unknown ETag"),
            "siteName": siteName,
            "documentName": document_metadata.get("name", "Unknown Document"),
            "documentUrl": document_metadata.get("webUrl", "Unknown URL"),
            "cTag": document_metadata.get("cTag", "Unknown CTag"),
            "path": document_metadata.get("parentReference", {}).get("path", "Unknown Path"),
            "file": document_metadata.get("file", {}),
            "fileSystemInfo": document_metadata.get("fileSystemInfo", {}),
            "driveId": document_metadata.get("parentReference", {}).get("driveId", "Unknown driveId"),
            "folderId": document_metadata.get("parentReference", {}).get("id", "Unknown folderId"),
            "documentType": document_metadata.get("file", {}).get("mimeType", "Unknown Type"),
            "documentSize": document_metadata.get("size", "Unknown Size"),
            "documentCreated": document_metadata.get("createdDateTime", "Unknown Date"),
            "documentModified": document_metadata.get("lastModifiedDateTime", "Unknown Date"),
            "documentPath": file_path,
            "siteId": siteId,
            "documentCreatedBy": document_metadata.get("createdBy", {}).get("user", {}).get("displayName", "Unknown"),
            "documentLastModifiedBy": document_metadata.get("lastModifiedBy", {}).get("user", {}).get("displayName",
                                                                                                      "Unknown"),
            "documentCreatedByEmail": document_metadata.get("createdBy", {}).get("user", {}).get("email", "Unknown"),
            "documentLastModifiedByEmail": document_metadata.get("lastModifiedBy", {}).get("user", {}).get("email",
                                                                                                           "Unknown"),
            # Include other metadata fields as needed
        }

        # Insert or update the document metadata in MongoDB
        result = documents_collection.update_one(
            {"_id": metadata_document["_id"]},
            {"$set": metadata_document},
            upsert=True
        )

        logging.info(
            f"Page metadata saved to MongoDB, upserted_id: {result.upserted_id}, matched_count: {result.matched_count}, modified_count: {result.modified_count}")
    except pymongo.errors.PyMongoError as e:
        logging.error(f"Failed to save page metadata to MongoDB: {e}")


# Define an array of desired file extensions (e.g., markdown, excel, powerpoint)
scrapable_file_extensions = ['.md', '.xlsx', '.xlsm', '.pptx', '.ppt', '.docx', '.txt', '.pdf']


def is_scrapable_file(file_name):
    """
    Check if the file has a scrapable extension.
    """
    extension = os.path.splitext(file_name)[1].lower()  # Extract the extension and convert to lowercase
    return extension in scrapable_file_extensions


def download_file_with_retry(download_url, file_path, headers, max_retries=3, chunk_size=512):
    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=10) as response:
                response.raise_for_status()  # Check for HTTP errors
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                return True  # Successful download
        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")
            attempt += 1
    logging.error(f"Failed to download file after {max_retries} attempts.")
    return False


def download_files_in_folder(driveId, folderDirectory, siteName, siteId, parentDirectory):
    queryResults = msgraph_request(graphURI + f'/v1.0/drives/{driveId}/root:/{folderDirectory}:/children',
                                   requestHeaders)
    if not queryResults:
        logging.error("Failed to retrieve folder contents.")
        return

    for item in queryResults['value']:
        if 'folder' in item:
            new_folder_name = item['name']
            new_folder_directory = os.path.join(parentDirectory, sanitize_path_element(new_folder_name))
            os.makedirs(new_folder_directory, exist_ok=True)
            download_files_in_folder(driveId, f"{folderDirectory}/{new_folder_name}", siteName, siteId,
                                     new_folder_directory)
        elif is_scrapable_file(item.get('name')):
            download_url = item.get('@microsoft.graph.downloadUrl')
            file_name = item.get('name')
            if download_url and file_name:
                file_path = os.path.join(parentDirectory, sanitize_path_element(file_name))
                success = download_file_with_retry(download_url, file_path, requestHeaders)
                if success:
                    save_document_metadata_to_mongo(item, siteName, siteId, file_path)


def format_html_content(title, description, sections):
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
    <p>{description}</p>
{sections}
</body>
</html>
"""
    formatted_sections = ""
    for section in sections:
        formatted_sections += "    <div class=\"section\">\n"
        for column in section:
            formatted_sections += "        <div class=\"column\">\n"
            for webpart_html in column:
                formatted_sections += f"            {webpart_html}\n"
            formatted_sections += "        </div>\n"
        formatted_sections += "    </div>\n"

    return html_template.format(title=html.escape(title), description=html.escape(description),
                                sections=formatted_sections)


def extract_inner_html_and_generate_html(json_folder):
    for filename in os.listdir(json_folder):
        if filename.endswith(".json") and not filename.endswith("_webparts.json"):
            try:
                json_path = os.path.join(json_folder, filename)
                html_filename = filename.replace(".json", ".html")
                html_path = os.path.join(json_folder, html_filename)

                logging.debug(f"Processing JSON file: {json_path}")

                with open(json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                title = data.get('title', 'No Title')
                description = data.get('description', 'No Description')

                sections_content = []
                for section in data.get('canvasLayout', {}).get('horizontalSections', []):
                    section_content = []
                    for column in section.get('columns', []):
                        column_content = [webpart.get('innerHtml', '') for webpart in column.get('webparts', [])]
                        section_content.append(column_content)
                    sections_content.append(section_content)

                formatted_html_content = format_html_content(title, description, sections_content)

                with open(html_path, 'w', encoding='utf-8') as html_file:
                    html_file.write(formatted_html_content)

                logging.info(f"Generated HTML file: {html_path}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error in file {filename}: {e}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
            else:
                logging.debug(f"Successfully processed {filename}")


# Adjusted function to convert HTML to Markdown and update MongoDB
def convert_html_to_markdown_and_update_mongo(html_folder, page_id_to_filename_map, collection):
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0

    for document_id, filename in page_id_to_filename_map.items():
        html_filename = filename + ".html"
        markdown_filename = filename + ".md"
        html_path = os.path.join(html_folder, html_filename)
        markdown_path = os.path.join(html_folder, markdown_filename)

        try:
            with open(html_path, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()
            markdown_content = h.handle(html_content)
            with open(markdown_path, 'w', encoding='utf-8') as markdown_file:
                markdown_file.write(markdown_content)
            update_mongodb_with_markdown_name(document_id, markdown_filename, collection)
        except Exception as e:
            logging.error(f"Error converting file {html_filename} to Markdown: {e}")


# Function to update MongoDB with markdownName, using correct document_id
def update_mongodb_with_markdown_name(document_id, markdown_filename, collection):
    try:
        result = collection.update_one(
            {"_id": document_id},
            {"$set": {"documentName": markdown_filename}}
        )
        logging.info(f"Updated MongoDB document with _id: {document_id} with documentName: {markdown_filename}")
    except PyMongoError as e:
        logging.error(f"Failed to update MongoDB document with _id: {document_id} with documentName: {e}")


# Main function to execute the script logic
def main():
    global accessToken, tokenExpiry, requestHeaders
    # Authenticate and acquire a new token if necessary
    if not accessToken or datetime.now() >= tokenExpiry:
        msal_certificate_auth()
        # Define base directories for documents and pages within the /data/ directory
        tenantName = "TEMPLATE"
        siteNames = ["asd", "ddd"]

        for siteName in siteNames:
            queryResults = msgraph_request(f"{graphURI}/v1.0/sites/{tenantName}.sharepoint.com:/sites/{siteName}",
                                           requestHeaders)
            siteId = queryResults.get('id')
            if siteId:
                # Document scraping
                base_directory = os.path.join(scrape_folder, "site_files", sanitize_path_element(siteName))
                os.makedirs(base_directory, exist_ok=True)
                queryResults = msgraph_request(f"{graphURI}/v1.0/sites/{siteId}/drives", requestHeaders)
                for drive in queryResults.get('value', []):
                    driveId = drive.get('id')
                    if driveId:
                        download_files_in_folder(driveId, "/", siteName, siteId, base_directory)

                # Page scraping
                pages_directory = os.path.join(scrape_folder, "site_pages", sanitize_path_element(siteName))
                os.makedirs(pages_directory, exist_ok=True)
                page_id_to_filename_map = {}
                pages = list_site_pages(siteId)
                for page in pages:
                    page_id = page.get('id')
                    page_name = sanitize_name(page.get('name', 'unknown_page'))
                    page_content = get_page_content(siteId, page_id)
                    document_id = save_page_metadata_to_mongo(page_content, siteName, pages_collection)
                    page_id_to_filename_map[document_id] = page_name
                    file_path = os.path.join(pages_directory, f"{page_name}.json")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(page_content, f, ensure_ascii=False, indent=4)
                    logging.info(f"Saved page: {page.get('name')}")
                extract_inner_html_and_generate_html(pages_directory)
                convert_html_to_markdown_and_update_mongo(pages_directory, page_id_to_filename_map, pages_collection)


if __name__ == "__main__":
    start_time = time.time()  # Record the start time
    main()
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    logging.info(f"Ingestion complete. Total execution time: {execution_time} seconds")
    if failed_resources:
        logging.error("The following resources were unavailable during execution:")
        for resource in failed_resources:
            logging.error(resource)
        with open(os.path.join(scrape_folder, 'failed_resources.json'), 'w') as f:
            json.dump(failed_resources, f, indent=4)
