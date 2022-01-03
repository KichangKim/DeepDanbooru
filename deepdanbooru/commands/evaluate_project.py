import os

import deepdanbooru as dd


def evaluate_project(project_path, target_path, threshold):
    if not os.path.exists(target_path):
        raise Exception(f"Target path {target_path} is not exists.")

    if os.path.isfile(target_path):
        taget_image_paths = [target_path]

    else:
        patterns = [
            "*.[Pp][Nn][Gg]",
            "*.[Jj][Pp][Gg]",
            "*.[Jj][Pp][Ee][Gg]",
            "*.[Gg][Ii][Ff]",
        ]

        taget_image_paths = dd.io.get_file_paths_in_directory(target_path, patterns)

        taget_image_paths = dd.extra.natural_sorted(taget_image_paths)

    project_context, model, tags = dd.project.load_project(project_path)

    width = project_context["image_width"]
    height = project_context["image_height"]

    for image_path in taget_image_paths:
        image = dd.data.load_image_for_evaluate(image_path, width=width, height=height)

        image_shape = image.shape
        # image = image.astype(np.float16)
        image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))
        y = model.predict(image)[0]

        result_dict = {}

        for i, tag in enumerate(tags):
            result_dict[tag] = y[i]

        print(f"Tags of {image_path}:")
        for tag in tags:
            if result_dict[tag] >= threshold:
                print(f"({result_dict[tag]:05.3f}) {tag}")

        print()
