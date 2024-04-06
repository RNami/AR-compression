import os
import string
import re

def sanitize_filename(filename):
    """
    Sanitizes a filename by removing non-ASCII characters, quotes, and replacing "a" with accents.

    Args:
        filename (str): Original filename.

    Returns:
        str: Sanitized filename.
    """
    # Remove non-ASCII characters
    sanitized_name = ''.join(c for c in filename if c in string.printable)

    # Remove quotes (single and double)
    sanitized_name = sanitized_name.replace('"', '').replace("'", '')

    # Replace "a" with accents with an ordinary "a"
    sanitized_name = re.sub(r'[áàâä]', 'a', sanitized_name, flags=re.IGNORECASE)

    return sanitized_name

def rename_files_in_directory(directory_path):
    """
    Renames files in a directory based on sanitization rules.

    Args:
        directory_path (str): Path to the directory containing files.
    """
    for filename in os.listdir(directory_path):
        original_path = os.path.join(directory_path, filename)
        sanitized_filename = sanitize_filename(filename)
        new_path = os.path.join(directory_path, sanitized_filename)

        if original_path != new_path:
            os.rename(original_path, new_path)
            print(f"Renamed '{filename}' to '{sanitized_filename}'")


target_directory = "test_pics\\Downloaded Persian Miniatures - Cropped and Resized"

rename_files_in_directory(target_directory)