import json
import os
from pathlib import Path


def serialize_as_json(target_object, path, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as stream:
        stream.write(json.dumps(target_object, indent=4, ensure_ascii=False))


def deserialize_from_json(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as stream:
        return json.loads(stream.read())


def try_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_paths_in_directory(path, patterns):
    return [
        str(file_path)
        for pattern in patterns
        for file_path in Path(path).rglob(pattern)
    ]


def get_image_file_paths_recursive(folder_path, patterns_string):
    patterns = patterns_string.split(",")

    return get_file_paths_in_directory(folder_path, patterns)
