import sys

import click

import deepdanbooru as dd


__version__ = '1.0.0'


@click.version_option(prog_name='DeepDanbooru', version=__version__)
@click.group()
def main():
    pass


@main.command('create-project')
@click.argument('project_path', type=click.Path(exists=False, resolve_path=True, file_okay=False, dir_okay=True))
def create_project(project_path):
    dd.commands.create_project(project_path)


@main.command('download-tags')
@click.option('--limit', default=10000, help='Limit for each category tag count.')
@click.option('--minimum-post-count', default=500, help='Minimum post count for tag.')
@click.option('--overwrite', help='Overwrite tags if exists.', is_flag=True)
@click.argument('path', type=click.Path(exists=False, resolve_path=True, file_okay=False, dir_okay=True))
def download_tags(path, limit, minimum_post_count, overwrite):
    dd.commands.download_tags(path, limit, minimum_post_count, overwrite)


@main.command('make-training-database')
@click.argument('source_path', type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False), nargs=1, required=True)
@click.argument('output_path', type=click.Path(exists=False, resolve_path=True, file_okay=True, dir_okay=False), nargs=1, required=True)
@click.option('--start-id', default=1, help='Start id.', )
@click.option('--end-id', default=sys.maxsize, help='End id.')
@click.option('--use-deleted', help='Use deleted posts.', is_flag=True)
@click.option('--chunk-size', default=5000000, help='Chunk size for internal processing.')
@click.option('--overwrite', help='Overwrite tags if exists.', is_flag=True)
@click.option('--vacuum', help='Execute VACUUM command after making database.', is_flag=True)
def make_training_database(source_path, output_path, start_id, end_id, use_deleted, chunk_size, overwrite, vacuum):
    dd.commands.make_training_database(source_path, output_path, start_id, end_id,
                                       use_deleted, chunk_size, overwrite, vacuum)


@main.command('train-project')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
def train_project(project_path):
    dd.commands.train_project(project_path)


@main.command('evaluate-project', help='Evaluate the project. If the target path is folder, it evaulates all images recursively.')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.argument('target_path', type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=True))
@click.option('--threshold', help='Threshold for tag estimation.', default=0.5)
def evaluate_project(project_path, target_path, threshold):
    dd.commands.evaluate_project(project_path, target_path, threshold)


@main.command('grad-cam', help='Experimental feature. Calculate activation map using Grad-CAM.')
@click.argument('project_path', type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True))
@click.argument('target_path', type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=True))
@click.argument('output_path', type=click.Path(resolve_path=True, file_okay=False, dir_okay=True), default='.')
@click.option('--threshold', help='Threshold for tag estimation.', default=0.5)
def grad_cam(project_path, target_path, output_path, threshold):
    dd.commands.grad_cam(project_path, target_path, output_path, threshold)


if __name__ == '__main__':
    main()
