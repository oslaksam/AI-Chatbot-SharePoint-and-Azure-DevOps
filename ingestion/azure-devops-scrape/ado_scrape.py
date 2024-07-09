import base64
import hashlib
import json
import logging
import os
import time
from urllib.parse import quote
import requests

# Logger setup
logger = logging.getLogger('azure_downloader')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('errors.log', mode='w')
file_handler.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

organization = os.environ.get('ORGANIZATION_NAME')
personal_access_token = os.environ.get('ADO_PAT')
encoded_pat = base64.b64encode(f":{personal_access_token}".encode('utf-8')).decode('utf-8')
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Basic {encoded_pat}'
}
api_version = "7.1"
wiki_output_dir = "/data/WIKI"  # PVC directory to save wiki markdown files
code_output_dir = "/data/CODE"  # PVC directory to save code markdown files
data_dir = "/data"  # PVC directory to save global mapping file
base_url = f'https://dev.azure.com/{organization}/'
projects_url = f'{base_url}_apis/projects?api-version={api_version}'
global_mapping_info = {}
existing_global_mapping = {}
if os.path.exists(f'{data_dir}/global_mapping.json'):
    with open(f'{data_dir}/global_mapping.json', 'r', encoding="utf-8") as f:
        existing_global_mapping = json.load(f)


def safe_request(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(f'Error: {e}')
        return None


def calculate_hash(content):
    sha = hashlib.sha256()
    sha.update(content)
    return sha.hexdigest()

def standardize_line_endings(raw_content):
    return raw_content.replace(b'\r\n', b'\n')


def download_repo_file(project_name, repo_name, file_path, folder_path='.'):
    try:
        repo_directory, file_name = os.path.split(file_path)
        local_directory = os.path.join(folder_path, repo_directory.lstrip('/'))
        os.makedirs(local_directory, exist_ok=True)
        file_path_on_disk = os.path.join(local_directory, file_name)

        encoded_file_path = quote(file_path, safe='')  # First encoding
        double_encoded_file_path = quote(encoded_file_path, safe='')  # Second encoding
        double_encoded = False
        # URL for file download
        download_file_url = f'{base_url}{project_name}/_apis/git/repositories/{repo_name}/items?scopePath={encoded_file_path}&api-version={api_version}'
        response = safe_request(download_file_url, headers)
        if response is None:
            # Retry with double encoding if double encoding fails
            download_file_url = f'{base_url}{project_name}/_apis/git/repositories/{repo_name}/items?scopePath={double_encoded_file_path}&api-version={api_version}'
            response = safe_request(download_file_url, headers)
            double_encoded = True
            if response is None:
                logger.error(f'Error downloading file at {file_path} from {repo_name}')
                return None, None, None
        # URL to view file in Azure DevOps
        file_url = f'{base_url}{project_name}/_git/{repo_name}?path={encoded_file_path}'
        if double_encoded:
            file_url = f'{base_url}{project_name}/_git/{repo_name}?path={double_encoded_file_path}'

        # Get the objectId for the file
        file_info_json_url = f'{base_url}{project_name}/_apis/git/repositories/{repo_name}/items?scopePath={encoded_file_path}&api-version={api_version}&$format=json'
        if double_encoded:
            file_info_json_url = f'{base_url}{project_name}/_apis/git/repositories/{repo_name}/items?scopePath={double_encoded_file_path}&api-version={api_version}&$format=json'
        file_info_response = safe_request(file_info_json_url, headers)
        # Parse the JSON
        data = json.loads(file_info_response.content.decode('utf-8'))
        # Extract objectId
        object_id = data['value'][0]['objectId']
        if object_id == '':
            print(f'Error: object_id is empty for {file_path}')
            exit(1)

        formatted_path = "CODE/{}/{}".format(project_name, repo_name) + file_path.replace('\\', '/')
        # Check if the hash exists in the global mapping
        existing_hash = existing_global_mapping.get(formatted_path, {}).get('hash')
        if existing_hash != {}:
            logger.info(f'Existing hash: {existing_hash}')
        if existing_hash == object_id:
            logger.info(f'No update for {formatted_path}.')

        with open(file_path_on_disk, 'wb') as f:
            f.write(response.content)

        # Update global mapping
        global_mapping_info[formatted_path] = {'url': file_url, 'hash': object_id}

        logger.info(f'Successfully downloaded Markdown file {file_name} to {formatted_path}. Url: {file_url}')
        return file_path, file_url, object_id

    except Exception as e:
        logger.error(f'Error downloading file at {file_path} from {repo_name}: {e}', exc_info=True)
        return None, None, None


def get_wiki_content(organization, project, wikiIdentifier, path="/"):
    try:
        url = f"https://dev.azure.com/{organization}/{project}/_apis/wiki/wikis/{wikiIdentifier}/pages"
        params = {
            "api-version": api_version,
            "path": path,
            "includeContent": True,
            "recursionLevel": "oneLevel"
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"404 Error for URL {e.request.url}")
        return None


def sanitize_filename(name):
    # Remove or replace invalid characters for filenames
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '&']
    for char in invalid_chars:
        name = name.replace(char, '-')
    return name


def save_to_markdown(directory, path, content, remote_url, etag):
    sanitized_path = sanitize_filename(path.lstrip("/"))
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, (sanitize_filename(os.path.basename(sanitized_path)) or "index") + ".md")

    # Normalize the slashes for the filename
    filename_normalized = filename.replace('\\', '/')

    # Check if the hash exists in the current mappings
    existing_hash = existing_global_mapping.get(filename_normalized, {}).get('hash')
    if existing_hash == etag:
        logger.info(f'No update for {filename_normalized}. Url: {remote_url}')

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
    # Remove leading '/data' if present in the path
    if filename_normalized.startswith("/data/"):
        filename_normalized = filename_normalized[len("/data/"):]
    # Save the URL and hash to the global mapping
    global_mapping_info[filename_normalized] = {'url': remote_url, 'hash': etag}
    logger.info(f'Successfully saved Markdown file {filename_normalized}. Url: {remote_url}')


def process_wiki_pages(organization, project_name, wiki_name, wikiIdentifier, path="/"):
    response = get_wiki_content(organization, project_name, wikiIdentifier, path)
    if response:
        data = response.json()
        etag = response.headers.get('ETag')
        if not etag:
            print(f'No ETag found for {path}')
            exit(1)
        directory = os.path.join(wiki_output_dir, sanitize_filename(project_name), sanitize_filename(wiki_name),
                                 sanitize_filename(os.path.dirname(path.lstrip("/"))))
        filename = os.path.join(directory, (sanitize_filename(os.path.basename(path.lstrip("/"))) or "index") + ".md")

        if os.path.exists(filename):  # Check if file already exists
            logger.info(f'File already exists: {filename}')
        save_to_markdown(directory, data['path'], data['content'], data['remoteUrl'], etag)

        for subpage in data.get('subPages', []):
            process_wiki_pages(organization, project_name, wiki_name, wikiIdentifier, subpage['path'])


def process_wikis():
    project_url = f"https://dev.azure.com/{organization}/_apis/projects?api-version={api_version}"
    response = requests.get(project_url, headers=headers)
    projects = json.loads(response.text)["value"]

    for project in projects:
        project_name = project['name']
        project_id = project['id']
        logger.info(f"Processing project: {project_name}")

        wikis_url = f"https://dev.azure.com/{organization}/{project_id}/_apis/wiki/wikis?api-version={api_version}"
        response = requests.get(wikis_url, headers=headers)
        wikis = json.loads(response.text)["value"]

        for wiki in wikis:
            logger.info(f"Processing wiki: {wiki['name']} in project: {project_name}")
            process_wiki_pages(organization, project_name, wiki['name'], wiki['id'])


def process_repositories():
    response = requests.get(projects_url, headers=headers)
    response.raise_for_status()
    projects_data = response.json()['value']

    for project in projects_data:
        project_name = project['name']
        repos_url = f'{base_url}{project_name}/_apis/git/repositories?api-version={api_version}'
        response = safe_request(repos_url, headers)
        if response is None:
            continue
        repos_data = response.json()['value']

        for repo in repos_data:
            repo_name = repo['name']
            repo_id = repo['id']

            folder_path = f'{code_output_dir}/{project_name}/{repo_name}'
            os.makedirs(folder_path, exist_ok=True)

            items_url = f'{base_url}{project_name}/_apis/git/repositories/{repo_id}/items?scopePath=/&recursionLevel=full&api-version={api_version}'
            response = safe_request(items_url, headers)
            if response is None:
                continue
            items_data = response.json()['value']

            for item in items_data:
                if item['gitObjectType'] == 'blob' and item['path'].endswith('.md'):
                    file_path, file_url, file_hash = download_repo_file(
                        project_name, repo_name, item['path'], folder_path=folder_path)


def main():
    start_time = time.time()
    try:
        # Ensure PVC directories exist
        for output_dir in [wiki_output_dir, code_output_dir]:
            os.makedirs(output_dir, exist_ok=True)

        # Process the repositories
        process_repositories()
        # Process the Wikis using the new approach:
        process_wikis()

        with open(f'{data_dir}/global_mapping.json', 'w', encoding="utf-8") as f:
            json.dump(global_mapping_info, f, indent=4)
        logger.info('Successfully created global mapping.json.')
        # create an empty file to signal that the process is complete
        with open(f'{data_dir}/scrape_complete', 'w', encoding="utf-8") as f:
            f.write('')
            logger.info('Successfully created scrape_complete file.')

    except Exception as e:
        logger.error(f'Error in main execution: {e}', exc_info=True)

    finally:
        end_time = time.time()
        logger.info(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    # If there is an environmental variable. Called skip. Then skip running the main function.
    if os.environ.get("SKIP_EXECUTION", "false").lower() == "true":
        logger.info('Skipping execution.')
    else:
        main()
