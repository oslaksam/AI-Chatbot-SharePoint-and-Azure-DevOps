import hashlib
import re

from github import Github
from github.ContentFile import ContentFile

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
    # Add other terms as needed
]

ACCEPTED_FILE_TYPES = ['.md']


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
                return True

    return False


def get_all_github_files(
        client: Github, repo_name: str, paths_to_ignore: list[str]
) -> list[ContentFile]:
    output = []
    repo = client.get_repo(repo_name)
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir" and file_content.name not in paths_to_ignore:
            contents.extend(repo.get_contents(file_content.path))
        elif (
                file_content.type == "file"
                and file_content.name.endswith(".md")
                and file_content.name not in paths_to_ignore
        ):
            output.append(file_content)
    return output


def is_acceptable_file_type(filename):
    """Check if a file is of an accepted type."""
    return any(filename.endswith(file_type) for file_type in ACCEPTED_FILE_TYPES)


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
