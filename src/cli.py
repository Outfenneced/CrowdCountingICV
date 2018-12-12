import os

import click

import data
import conv_net


@click.group()
def cli():
    pass


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False))
def download(data_folder):
    os.makedirs(data_folder, exist_ok=True)
    data.download_dataset(data_folder)


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("output-folder", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--remove", is_flag=True)
def preprocess(data_folder, output_folder, remove):
    os.makedirs(output_folder, exist_ok=True)
    data.preprocess(data_folder, output_folder, remove)


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("classifier-out", type=click.Path(dir_okay=False, file_okay=True))
def train(data_folder, classifier_out):
    os.makedirs(os.path.dirname(classifier_out), exist_ok=True)
    conv_net.train(data_folder, classifier_out)


if __name__ == '__main__':
    cli()
