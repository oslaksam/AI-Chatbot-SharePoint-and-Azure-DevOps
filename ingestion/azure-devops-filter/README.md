# README.md for Cleanup Script

## Overview

This Python script is designed for file management and analysis within a specified directory. It performs a series of checks on markdown files, flagging them based on various criteria such as deprecated terms, content length, and unique content. The script utilizes regular expressions, hashlib for content hashing, and logging for verbose output. It includes features like loading templates, calculating hashes, and copying clean files to a separate directory.

## Features

- **Template Management**: Loads markdown templates from a specified folder for comparison.
- **File Flagging**: Flags files containing certain keywords, including deprecated or obsolete terms.
- **Content Analysis**: Analyzes file content for specific criteria such as TODOs, word count thresholds, and unique syntax.
- **Duplicate Detection**: Uses SHA256 hashing to identify duplicate content.
- **File Copying**: Copies non-flagged, clean files to a separate directory for further use.

## Requirements

- Python 3.x
- Standard Python libraries: `os`, `json`, `hashlib`, `shutil`, `collections`, `logging`, `re`, `time`

## Usage

1. **Set Up Environment**: Ensure Python 3.x is installed.
2. **Configure Script Parameters**:
   - Modify `TEMPLATES`, `FLAG_TERMS`, and `ACCEPTED_FILE_TYPES` lists as needed.
   - Set `root_folder` to the directory you wish to process.
3. **Run the Script**: Execute the script via command line or an IDE.
4. **Check Outputs**:
   - Inspect the flagged files in the generated JSON file.
   - Review the statistics saved in `statistics.txt`.
   - Examine copied clean files in the designated directory.

## Notes

- The script can be customized by modifying the lists for templates and flag terms.
- It is designed to be run in environments where markdown file management and analysis are required.
- Logging provides detailed information about the script's operations and outcomes.

