# File Cleanup Script

## Overview
The script is designed to process and clean up a directory of files. The primary focus is on Markdown files, with the goal of flagging files that are outdated, duplicates, or deemed unnecessary based on certain conditions. The script also gathers statistics on the processed files, providing insights into how many files are flagged and for what reasons.

## How It Works

### 1. Configuration
The script contains a list of templates (currently empty in the provided code) and flag terms that help identify files which need to be flagged. It also specifies the accepted file types, which are primarily Markdown files.

### 2. Utility Functions:

#### `is_acceptable_file_type(filename)`
Determines if a file is of an accepted type, like `.md`.

#### `load_templates_from_folder()`
Loads content templates from separate files within a designated templates folder. These templates are useful in identifying generic or auto-generated content.

#### `calculate_hash(content)`
Computes the SHA-256 hash of the given content to help identify duplicate files.

#### `is_mostly_todo(content)`
Checks if the content is primarily composed of placeholders like "TODO".

#### `is_below_word_threshold(content, threshold)`
Determines if the content is shorter than a specified word count threshold.

#### `contains_only_specified_syntax(content)`
Checks if the content only contains certain specified syntax patterns.

#### `process_files(root_folder, templates, word_threshold=10)`
Main function that processes files in a directory, flagging them based on multiple conditions such as matching a template, containing mostly "TODO" placeholders, being below a word threshold, etc.

#### `is_empty_mapping_file(file_name, content)`
Determines if a file is an empty mapping.json file.

#### `copy_clean_files(root_folder, flagged_files)`
After processing, the clean files (ones that aren't flagged) are copied to a separate directory, ensuring a clean set of files is available after the script runs.

#### `print_and_save_statistics(stats)`
Prints and saves the statistics collected during the file processing to a text file.

### 3. Main Execution:
In the main execution block:
- The root folder is set as the current directory.
- Content templates are loaded from a designated folder and combined with predefined templates.
- Files are processed using the `process_files` function, resulting in a list of flagged files and statistics about the processing.
- Flagged files and stats are saved to JSON files.
- Clean (unflagged) files are copied to a separate directory.
- Statistics are printed and saved to a text file.
- The execution time of the script is logged.

## Purpose
The purpose of the script is to automate the process of cleaning up a directory of files, especially focusing on Markdown files. It identifies and flags files that might be outdated, redundant, or unnecessary, helping maintain a clean and organized file system. By saving statistics, the script also provides insights into the cleanup process.

