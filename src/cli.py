import csv
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
@click.option("--no-train", is_flag=True)
def preprocess(data_folder, output_folder, no_train, remove):
    os.makedirs(output_folder, exist_ok=True)
    data.preprocess(data_folder, output_folder, calc_train=not no_train, remove=remove)


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("classifier-out", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--gpu", type=int, default=0)
@click.option("--epochs", type=int, default=1)
@click.option("--batch", type=int, default=100)
@click.option("--threading", type=int, default=8)
def train(data_folder, classifier_out, gpu, epochs, batch, threading):
    os.makedirs(classifier_out, exist_ok=True)
    conv_net.train(data_folder, classifier_out, gpu=gpu, epochs=epochs, batch_size=batch, load_threading=threading)


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("classifiers", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("summary-file", type=click.Path(dir_okay=False, file_okay=True))
@click.option("--gpu", type=int, default=0)
@click.option("--batch", type=int, default=100)
@click.option("--threading", type=int, default=8)
def validate(data_folder, classifiers, summary_file, gpu, batch, threading):
    losses = conv_net.test(data_folder, classifiers, test_type="Valid", gpu=gpu, batch_size=batch, load_threading=threading)
    with open(summary_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(losses)


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("classifiers", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("summary-file", type=click.Path(dir_okay=False, file_okay=True))
@click.option("--gpu", type=int, default=0)
@click.option("--batch", type=int, default=100)
@click.option("--threading", type=int, default=8)
def test(data_folder, classifiers, summary_file, gpu, batch, threading):
    losses = conv_net.test(data_folder, classifiers, test_type="Test", gpu=gpu, batch_size=batch, load_threading=threading)
    with open(summary_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(losses)


@cli.command()
@click.argument("data-folder", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("classifiers", type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument("summary-file", type=click.Path(dir_okay=False, file_okay=True))
@click.option("--gpu", type=int, default=0)
@click.option("--batch", type=int, default=100)
@click.option("--threading", type=int, default=8)
def test_train_data(data_folder, classifiers, summary_file, gpu, batch, threading):
    losses = conv_net.test(data_folder, classifiers, test_type="Train", gpu=gpu, batch_size=batch, load_threading=threading)
    with open(summary_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(losses)


if __name__ == '__main__':
    cli()
