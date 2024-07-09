# SharePoint Data Scrape Script

### Overview

This script automates the process of extracting data from SharePoint sites, including both documents and pages, and then storing this information in MongoDB. It leverages the Microsoft Graph API for data extraction, employing MSAL for authentication via certificates. The script is structured to handle errors gracefully and log its operations extensively for troubleshooting and auditing purposes.

### Setup and Configuration

- **Environment Variables:** Configuration parameters are managed via environment variables, ensuring sensitive information is kept secure. These variables include connection details for MongoDB, credentials for SharePoint access, and local storage paths.
- **Logging:** The script initializes logging at the start, creating a log file within a specified directory. This allows for tracking the script's execution and debugging issues.
- **MongoDB Connection:** Utilizes PyMongo to establish a connection to MongoDB, where extracted data will be stored.

### Authentication

- **MSAL Certificate Authentication:** A function `msal_certificate_auth` is responsible for authenticating against Azure AD using a certificate, thereby acquiring an access token for the Microsoft Graph API.

### SharePoint Site Data Extraction

The SharePoint Site Data Extraction section of the script focuses on interfacing with Microsoft SharePoint via the Microsoft Graph API to retrieve and manage data from SharePoint sites. This process is fundamental for aggregating content such as documents and site pages, which are then processed and stored in MongoDB. The operations within this section are meticulously designed to handle various data types and structures found within SharePoint, ensuring a comprehensive data extraction process.

#### Key Components and Processes

1. **Site Identification and Access:**
   - **Functionality:** The script initiates its interaction with SharePoint by identifying and accessing specific SharePoint sites. This is achieved by constructing and sending HTTP GET requests to the Microsoft Graph API endpoint dedicated to SharePoint sites.
   - **Implementation:** Utilizing the `get_site` function, the script constructs a URL that targets the Microsoft Graph API's `/sites` endpoint, passing in the tenant name and site name. The response contains crucial information, including the site's unique ID, which is used in subsequent requests to access site-specific resources.

2. **Document and File Extraction:**
   - **Functionality:** A significant portion of the data extraction process involves identifying and downloading files stored within the SharePoint sites. This includes a wide range of file types such as documents, spreadsheets, presentations, and textual content.
   - **Implementation:** The script employs the `download_files_in_folder` function to traverse the folder structure of a SharePoint site. It filters files based on predefined scrapable file extensions, downloads them, and extracts metadata. The process is supported by the `is_scrapable_file` utility, which checks file extensions against a list of desired types.
   - **Error Handling and Retry Logic:** To enhance reliability, the script includes error handling and retry mechanisms, encapsulated in the `download_file_with_retry` function. This ensures that temporary network issues or other interruptions do not halt the data extraction process.

3. **Page Scraping and Processing:**
   - **Functionality:** In addition to document files, the script extracts content from SharePoint site pages. This includes the page's metadata and content, which can range from text to embedded media and web parts.
   - **Implementation:** Functions such as `list_site_pages`, `get_page_content`, and `get_page_webparts` are used to query the Microsoft Graph API for page information. Each page is processed to extract its content, which is then transformed into both HTML and Markdown formats for easier handling and storage.
   - **Content Transformation:** The transformation of page content into HTML and Markdown is a critical step for making the content more accessible and manageable. This is achieved through the `format_html_content` function for HTML formatting and the `convert_html_to_markdown_and_update_mongo` function for Markdown conversion. The latter also updates MongoDB with the new format, ensuring that the database reflects the most current state of the data.

4. **Metadata Extraction and Storage:**
   - **Functionality:** Extracting and storing metadata for both documents and pages is crucial for later retrieval and analysis. Metadata includes information such as file names, modification dates, authors, and URLs.
   - **Implementation:** The script utilizes MongoDB's upsert functionality to either update existing records with new metadata or insert new records if they do not already exist. This is handled through functions like `save_page_metadata_to_mongo` and `save_document_metadata_to_mongo`, which prepare and execute the database operations.

#### Conclusion

The SharePoint Site Data Extraction section is a comprehensive effort to interface with SharePoint, extract a wide array of content, and prepare this content for storage and further processing. Through meticulous implementation of API interactions, file handling, content processing, and metadata management, the script effectively automates the ingestion of SharePoint site data into MongoDB. This automation facilitates enhanced data management, analysis, and utilization within organizational processes.

### MongoDB Integration

- **Data Storage:** Documents and pages are stored in MongoDB, with their metadata updated or inserted as necessary. This integration allows for robust data management and retrieval capabilities.

### Error Handling and Logging

- Extensive error handling and logging throughout the script ensure that issues are captured and can be addressed. This includes logging for HTTP errors, connection issues, and operational logs for tracking the script's progress.

### Retry Logic

- Implements a retry logic for HTTP requests, enhancing the script's resilience against temporary network issues or service unavailability.

### Utility Functions

- Includes several utility functions for tasks such as sanitizing file and directory names, formatting HTML content, and updating MongoDB documents based on changes.

### Execution Flow

- The `main` function orchestrates the entire process, from authentication to data extraction and storage.
- It iterates over a predefined list of SharePoint site names, processes each site's documents and pages, and manages data storage.

### Conclusion

This script is a comprehensive solution for organizations looking to automate the ingestion of SharePoint site data into MongoDB. It demonstrates advanced Python programming techniques, including working with APIs, handling authentication and security, managing file downloads, and integrating with databases.

