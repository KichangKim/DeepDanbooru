import os
import time

import requests

import deepdanbooru as dd


def download_category_tags(
    category,
    minimum_post_count,
    limit,
    username,
    api_key,
    page_size=1000,
    order="count",
):
    category_to_index = {"general": 0, "artist": 1, "copyright": 3, "character": 4}

    gold_only_tags = ["loli", "shota", "toddlercon"]

    if category not in category_to_index:
        raise Exception(f"Not supported category : {category}")

    category_index = category_to_index[category]

    parameters = {
        "limit": page_size,
        "page": 1,
        "search[order]": order,
        "search[category]": category_index,
        "login": username,
        "api_key": api_key,
    }

    request_url = "https://danbooru.donmai.us/tags.json"
    headers={'User-Agent': 'Mozilla/5.0'}

    tags = set()

    while True:
        response = requests.get(
            request_url,
            params=parameters,
            headers=headers,
        )
        response_json = response.json()

        response_tags = [
            tag_json["name"]
            for tag_json in response_json
            if tag_json["post_count"] >= minimum_post_count
        ]

        if not response_tags:
            break

        is_full = False

        for tag in response_tags:
            if tag in gold_only_tags:
                continue

            tags.add(tag)

            if len(tags) >= limit:
                is_full = True
                break

        if is_full:
            break
        else:
            parameters["page"] += 1

    return tags


def download_tags(
    project_path, limit, minimum_post_count, is_overwrite, username, api_key
):
    print(
        f"Start downloading tags ... (limit:{limit}, minimum_post_count:{minimum_post_count})"
    )

    log = {
        "date": time.strftime("%Y/%m/%d %H:%M:%S"),
        "limit": limit,
        "minimum_post_count": minimum_post_count,
    }

    system_tags = [
        "rating:general",
        "rating:sensitive",
        "rating:questionable",
        "rating:explicit",
        # 'score:very_bad',
        # 'score:bad',
        # 'score:average',
        # 'score:good',
        # 'score:very_good',
    ]

    category_definitions = [
        {
            "category_name": "General",
            "category": "general",
            "path": os.path.join(project_path, "tags-general.txt"),
        },
        # {
        #    'category_name': 'Artist',
        #    'category': 'artist',
        #    'path': os.path.join(path, 'tags-artist.txt'),
        # },
        # {
        #    'category_name': 'Copyright',
        #    'category': 'copyright',
        #    'path': os.path.join(path, 'tags-copyright.txt'),
        # },
        {
            "category_name": "Character",
            "category": "character",
            "path": os.path.join(project_path, "tags-character.txt"),
        },
    ]

    all_tags_path = os.path.join(project_path, "tags.txt")

    if not is_overwrite and os.path.exists(all_tags_path):
        raise Exception(f"Tags file is already exists : {all_tags_path}")

    dd.io.try_create_directory(os.path.dirname(all_tags_path))
    dd.io.serialize_as_json(log, os.path.join(project_path, "tags_log.json"))

    categories_for_web = []
    categories_for_web_path = os.path.join(project_path, "categories.json")
    tag_start_index = 0

    total_tags_count = 0

    with open(all_tags_path, "w") as all_tags_stream:
        for category_definition in category_definitions:
            category = category_definition["category"]
            category_tags_path = category_definition["path"]

            print(f"{category} tags are downloading ...")
            tags = download_category_tags(
                category, minimum_post_count, limit, username, api_key
            )

            tags = dd.extra.natural_sorted(tags)
            tag_count = len(tags)
            if tag_count == 0:
                print(f"{category} tags are not exists.")
                continue
            else:
                print(f"{tag_count} tags are downloaded.")

            with open(category_tags_path, "w") as category_tags_stream:
                for tag in tags:
                    category_tags_stream.write(f"{tag}\n")
                    all_tags_stream.write(f"{tag}\n")

            categories_for_web.append(
                {
                    "name": category_definition["category_name"],
                    "start_index": tag_start_index,
                }
            )

            tag_start_index += len(tags)
            total_tags_count += tag_count

        for tag in system_tags:
            all_tags_stream.write(f"{tag}\n")

        categories_for_web.append({"name": "System", "start_index": total_tags_count})

    dd.io.serialize_as_json(categories_for_web, categories_for_web_path)

    print(f"Total {total_tags_count} tags are downloaded.")

    print("All processes are complete.")
