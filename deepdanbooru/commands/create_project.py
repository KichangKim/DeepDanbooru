import os

import deepdanbooru as dd


def create_project(project_path):
    """
    Create new project with default parameters.
    """
    dd.io.try_create_directory(project_path)
    project_context_path = os.path.join(project_path, "project.json")
    dd.io.serialize_as_json(dd.project.DEFAULT_PROJECT_CONTEXT, project_context_path)

    print(f"New project was successfully created. ({project_path})")
