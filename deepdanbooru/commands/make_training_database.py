import os
import sqlite3


def make_training_database(
    source_path,
    output_path,
    start_id,
    end_id,
    use_deleted,
    chunk_size,
    overwrite,
    vacuum,
):
    """
    Make sqlite database for training. Also add system tags.
    """
    if source_path == output_path:
        raise Exception("Source path and output path is equal.")

    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            raise Exception(f"{output_path} is already exists.")

    source_connection = sqlite3.connect(source_path)
    source_connection.row_factory = sqlite3.Row
    source_cursor = source_connection.cursor()

    output_connection = sqlite3.connect(output_path)
    output_connection.row_factory = sqlite3.Row
    output_cursor = output_connection.cursor()

    table_name = "posts"
    id_column_name = "id"
    md5_column_name = "md5"
    extension_column_name = "file_ext"
    tags_column_name = "tag_string"
    tag_count_general_column_name = "tag_count_general"
    rating_column_name = "rating"
    score_column_name = "score"
    deleted_column_name = "is_deleted"

    # Create output table
    print("Creating table ...")
    output_cursor.execute(
        f"""CREATE TABLE {table_name} (
        {id_column_name} INTEGER NOT NULL PRIMARY KEY,
        {md5_column_name} TEXT,
        {extension_column_name} TEXT,
        {tags_column_name} TEXT,
        {tag_count_general_column_name} INTEGER )"""
    )
    output_connection.commit()
    print("Creating table is complete.")

    current_start_id = start_id

    while True:
        print(f"Fetching source rows ... ({current_start_id}~)")
        source_cursor.execute(
            f"""SELECT
                {id_column_name},{md5_column_name},{extension_column_name},{tags_column_name},{tag_count_general_column_name},{rating_column_name},{score_column_name},{deleted_column_name}
            FROM {table_name} WHERE ({id_column_name} >= ?) ORDER BY {id_column_name} ASC LIMIT ?""",
            (current_start_id, chunk_size),
        )

        rows = source_cursor.fetchall()

        if not rows:
            break

        insert_params = []

        for row in rows:
            post_id = row[id_column_name]
            md5 = row[md5_column_name]
            extension = row[extension_column_name]
            tags = row[tags_column_name]
            general_tag_count = row[tag_count_general_column_name]
            rating = row[rating_column_name]
            # score = row[score_column_name]
            is_deleted = row[deleted_column_name]

            if post_id > end_id:
                break

            if is_deleted and not use_deleted:
                continue

            if rating == "g":
                tags += f" rating:general"
            elif rating == "s":
                tags += f" rating:sensitive"            
            elif rating == "q":
                tags += f" rating:questionable"
            elif rating == "e":
                tags += f" rating:explicit"

            # if score < -6:
            #     tags += f' score:very_bad'
            # elif score >= -6 and score < 0:
            #     tags += f' score:bad'
            # elif score >= 0 and score < 7:
            #     tags += f' score:average'
            # elif score >= 7 and score < 13:
            #     tags += f' score:good'
            # elif score >= 13:
            #     tags += f' score:very_good'

            insert_params.append((post_id, md5, extension, tags, general_tag_count))

        if insert_params:
            print("Inserting ...")
            output_cursor.executemany(
                f"""INSERT INTO {table_name} (
                {id_column_name},{md5_column_name},{extension_column_name},{tags_column_name},{tag_count_general_column_name})
                values (?, ?, ?, ?, ?)""",
                insert_params,
            )
            output_connection.commit()

        current_start_id = rows[-1][id_column_name] + 1

        if current_start_id > end_id or len(rows) < chunk_size:
            break

    if vacuum:
        print("Vacuum ...")
        output_cursor.execute("vacuum")
        output_connection.commit()

    source_connection.close()
    output_connection.close()
