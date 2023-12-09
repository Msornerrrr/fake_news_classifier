import json
import os

def load_json(json_path):
    """
    Load a JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No JSON file found at {json_path}")

    with open(json_path, 'r') as file:
        data = json.load(file)

    return data

def load_json_with_key(json_path, key):
    """
    Load a JSON file and return a specific key.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No JSON file found at {json_path}")

    with open(json_path, 'r') as file:
        data = json.load(file)
        if key not in data:
            raise ValueError(f"No key {key} found in JSON file")

    return data[key]

def load_json_with_query(json_path, query):
    """
    Load a JSON file and return a specific query.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No JSON file found at {json_path}")

    with open(json_path, 'r') as file:
        data = json.load(file)
        result = data
        for key in query:
            if key not in result:
                raise ValueError(f"No key {key} found in JSON file")
            result = result[key]

    return result

def save_json(data, json_path, indent=2):
    """
    Save a JSON file.
    """
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=indent)

def add_json_with_key(data, json_path, key, indent=2):
    """
    Add a key to a JSON file.
    """
    # Check if the JSON file already exists
    if os.path.exists(json_path):
        # Read the existing data
        with open(json_path, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # Update with new data
    existing_data[key] = data

    # Write back to the JSON file
    with open(json_path, 'w') as file:
        json.dump(existing_data, file, indent=indent)
