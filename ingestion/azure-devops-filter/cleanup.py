import os
import json
import hashlib
import shutil
from collections import defaultdict
import logging
import re  # Importing the regular expression module
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

TEMPLATES = [
    # Add other templates as needed
    """
    """
]

# List of terms to flag files based on their name or path
FLAG_TERMS = [
    "deprecated",
    "old",
    "outdated",
    "obsolete",
    "legacy",
    "unused",
    "to be deleted",
    "to be removed",
    "to be deprecated",
    "useless",
    "not used",
    "LICENSE.md",
    "license.md",
    "License.md",
    "chainlit.md"
    "Release Notes.md"
    "release notes",
    "Releases",
    "changelog",
    "CHANGELOG",
    "ReleaseNotes",
    "change log",
    "bugfix.md",
    "major_feature.md",
    "minor_feature.md",
    "release_notes.md",
    "release_notes-template.md",
    "release_notes-template"
    # Add other terms as needed
]

ACCEPTED_FILE_TYPES = ['.md']


def is_acceptable_file_type(filename):
    """Check if a file is of an accepted type."""
    return any(filename.endswith(file_type) for file_type in ACCEPTED_FILE_TYPES)


def load_templates_from_folder():
    """Load content templates from separate files within the templates folder."""
    templates = []
    template_folder = "templates"
    template_files = [f for f in os.listdir(template_folder) if f.endswith('.md')]

    for template_file in template_files:
        with open(os.path.join(template_folder, template_file), 'r', encoding='utf-8', errors='ignore') as file:
            templates.append(file.read())
    return templates


def calculate_hash(content):
    """Compute the hash of file content."""
    sha = hashlib.sha256()
    sha.update(content.encode('utf-8'))
    return sha.hexdigest()


def is_mostly_todo(content):
    """Check if the content is mostly composed of TODO or similar keywords."""
    total_words = len(content.split())
    if total_words == 0:
        return False
    todo_keywords = ['todo', 'write later', 'to be done later']  # Add more as needed
    todo_count = sum(content.lower().count(keyword) for keyword in todo_keywords)
    return todo_count / total_words > 0.3


def is_below_word_threshold(content, threshold):
    """Check if the content contains fewer words than the given threshold."""
    total_words = len(content.split())
    return total_words <= threshold


def contains_only_specified_syntax(content):
    patterns = [
        re.compile(r'^\s*\[\[_TOC_\]\]\s*$'),
        re.compile(r'^\s*\[\[_TOSP_\]\]\s*$')
    ]
    for pattern in patterns:
        if pattern.match(content):
            return True
    return False


def should_flag_file(file_path, flag_terms):
    """
    Checks if the file path contains any of the flagged terms as whole words within the path components, considering underscores and other non-word characters.

    :param file_path: The path of the file to check.
    :param flag_terms: A list of terms that, if found as whole words in the file path, should cause it to be flagged.
    :return: Tuple of (flag status, flag term) - flag status is True if the file should be flagged, False otherwise,
             and flag term is the term that caused the flagging.
    """
    # Split the file path into components based on slashes, dots, and underscores
    components = re.split(r'[/.]', file_path.lower())
    sub_components = [re.split(r'[\W_]+', component) for component in components]
    flattened_components = [item for sublist in sub_components for item in sublist]

    # Check if any flag term is found as a whole word within any component
    for term in flag_terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        for component in flattened_components:
            if re.search(pattern, component):
                return True, term

    return False, None

def process_files(root_folder, templates, word_threshold=10):
    """Process the files and flag based on the conditions."""
    flagged_files = defaultdict(list)
    content_hashes = set()  # Use a set to store unique content hashes

    stats = {
        'total_markdown': 0,
        'flagged': 0,
        'unique_flagged': 0,  # Unique count of flagged files
        'reasons': defaultdict(int)
    }

    unique_flagged = set()  # Use a set to keep track of unique flagged files

    # List of directories to process, with CODE as the last item, important for Redis indexes
    dirs_to_process = [d for d in os.listdir(root_folder) if
                       os.path.isdir(os.path.join(root_folder, d)) and d != "CODE" and d != "templates" and d != "clean"]
    dirs_to_process.append("CODE")

    for directory in dirs_to_process:
        dir_path = os.path.join(root_folder, directory)
        for root, _, files in os.walk(dir_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                if is_acceptable_file_type(file_name):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()

                    # Check for empty mapping.json
                    if is_empty_mapping_file(file_name, content):
                        flagged_files['empty_mapping'].append(file_path)
                        stats['reasons']['empty_mapping'] += 1
                        logging.info(f"Flagged {file_path} as empty mapping file")

                    if file_name.endswith('.md'):
                        stats['total_markdown'] += 1

                    reasons_for_current_file = []  # Store reasons for the current file

                    # Check if the file only contains the specified Azure DevOps syntax
                    if contains_only_specified_syntax(content):
                        reasons_for_current_file.append('only_azdevops_syntax')
                        stats['reasons']['only_azdevops_syntax'] += 1
                        logging.info(f"Flagged {file_path} due to containing only specified Azure DevOps syntax")

                    # Check for FLAG_TERMS in file name or path
                    flag_status, flag_term = should_flag_file(file_path, FLAG_TERMS)
                    if flag_status:
                        reasons_for_current_file.append('flag_term')
                        stats['reasons']['flag_term'] += 1
                        logging.info(f"Flagged {file_path} due to term: {flag_term}")

                    # Check if the file is an empty markdown or text file
                    if not content.strip() and (file_name.endswith('.md') or file_name.endswith('.txt')):
                        reasons_for_current_file.append('empty')
                        stats['reasons']['empty'] += 1
                        logging.info(f"Flagged {file_path} as empty")

                    # Check for documents below the word threshold
                    if is_below_word_threshold(content, word_threshold):
                        reasons_for_current_file.append('below_word_threshold')
                        stats['reasons']['below_word_threshold'] += 1
                        logging.info(f"Flagged {file_path} for being below the word threshold of {word_threshold}")

                    # Check for default autogenerated READMEs from templates
                    if content.strip() in templates:
                        reasons_for_current_file.append('template')
                        stats['reasons']['template'] += 1
                        logging.info(f"Flagged {file_path} due to matching a template")

                    # Check for mostly TODO content
                    if is_mostly_todo(content):
                        reasons_for_current_file.append('mostly_todo')
                        stats['reasons']['mostly_todo'] += 1
                        logging.info(f"Flagged {file_path} as mostly TODO")

                    # Check for duplicate content
                    file_hash = calculate_hash(content)
                    if file_hash in content_hashes:
                        reasons_for_current_file.append('duplicate')
                        stats['reasons']['duplicate'] += 1
                        logging.info(f"Flagged {file_path} as duplicate")
                    else:
                        content_hashes.add(file_hash)

                    # Add the file to flagged_files for each reason
                    for reason in reasons_for_current_file:
                        flagged_files[reason].append(file_path)

                    # Add the file to unique_flagged if it has any flag reasons
                    if reasons_for_current_file:
                        unique_flagged.add(file_path)

    stats['flagged'] = sum(len(v) for v in flagged_files.values())
    stats['unique_flagged'] = len(unique_flagged)
    stats['clean_markdown'] = stats['total_markdown'] - stats['unique_flagged']
    return flagged_files, stats


def is_empty_mapping_file(file_name, content):
    """Check if a file is an empty mapping.json file."""
    return file_name == "mapping.json" and content.strip() in ["", "{}"]


def copy_clean_files(root_folder, flagged_files):
    """Copy files not flagged to a 'clean' directory."""
    all_flagged_files = set(file for sublist in flagged_files.values() for file in sublist)
    clean_folder = os.path.join(root_folder, 'clean')

    # Delete the clean folder if it exists
    if os.path.exists(clean_folder):
        logging.info(f"Deleting {clean_folder}...")
        shutil.rmtree(clean_folder)

    script_directory = os.path.abspath(os.path.dirname(__file__))
    templates_folder = os.path.join(script_directory, 'templates')  # Path to the templates folder

    for root, _, files in os.walk(root_folder):
        # If the current root directory is the templates folder, skip this iteration
        if os.path.abspath(root) == templates_folder:
            continue

        for file_name in files:
            file_path = os.path.join(root, file_name)

            if os.path.abspath(root) == script_directory and file_name != "global_mapping.json":
                continue

            if is_acceptable_file_type(file_name) or file_name == "global_mapping.json":
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                if file_name == "mapping.json" and is_empty_mapping_file(file_name, content):
                    logging.info(f"Skipped empty mapping file: {file_path}")
                    continue

                # Copy the file to multiple folders for each reason
                reasons_for_current_file = [reason for reason, paths in flagged_files.items() if file_path in paths]
                for reason in reasons_for_current_file:
                    dest_folder = os.path.join(clean_folder, reason)
                    dest_path = os.path.join(dest_folder, os.path.relpath(file_path, root_folder))
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    logging.info(f"Copied {file_path} to {dest_path} due to reason: {reason}")

                # If the file is clean, copy it to the 'clean' folder
                if not reasons_for_current_file:
                    dest_folder = os.path.join(clean_folder, 'clean')
                    dest_path = os.path.join(dest_folder, os.path.relpath(file_path, root_folder))
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    logging.info(f"Copied {file_path} to {dest_path}")


def print_and_save_statistics(stats):
    """Print and save statistics to a file."""
    logging.info("Saving statistics...")
    total_files = stats['total_markdown']
    total_percentage = lambda x: (x / total_files) * 100

    with open('statistics.txt', 'w') as stats_file:
        for key, value in stats.items():
            if key == 'reasons':
                stats_file.write("Reasons for flagging:\n")
                for reason, count in value.items():
                    percentage = total_percentage(count)
                    logging.info(f"{reason}: {count} ({percentage:.2f}%)")
                    stats_file.write(f"{reason}: {count} ({percentage:.2f}%)\n")
            else:
                percentage = total_percentage(value) if "markdown" in key else ""
                log_msg = f"{key}: {value}" + (f" ({percentage:.2f}%)" if percentage else "")
                logging.info(log_msg)
                stats_file.write(log_msg + "\n")


def main():
    root_folder = '/data'
    logging.info("Starting the cleanup process...")
    templates = TEMPLATES + load_templates_from_folder()
    flagged_files, stats = process_files(root_folder, templates)
    with open(f'{root_folder}/flagged_files.json', 'w') as json_file:
        json.dump(flagged_files, json_file, indent=4)
    with open(f'{root_folder}/stats.json', 'w') as json_file:
        json.dump(stats, json_file, indent=4)
    copy_clean_files(root_folder, flagged_files)
    print_and_save_statistics(stats)
    # create an empty file to signal that the process is complete
    with open(f'{root_folder}/filter_complete', 'w') as f:
        f.write('')
        logging.info("Filter completed file created!")
    logging.info("Cleanup process completed!")


if __name__ == '__main__':
    # If there is an environmental variable. Called skip. Then skip running the main function.
    if os.environ.get("SKIP_EXECUTION", "false").lower() == "true":
        logging.info('Skipping execution.')
    else:
        start_time = time.time()
        main()
        end_time = time.time()
        logging.info(f"Execution time: {end_time - start_time} seconds")