import json
import os

def load_config(config_path):
    """
    Loads a JSON configuration file.
    
    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results, output_path):
    """
    Saves the analysis results to a JSON file.

    Args:
        results (dict): The dictionary containing the results.
        output_path (str): The path to save the output file.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results successfully saved to {output_path}")
    except IOError as e:
        print(f"Error saving results to {output_path}: {e}")

def ensure_dir_exists(directory_path):
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
