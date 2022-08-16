import click

from rexify.pipeline import compile_


@click.group()
@click.version_option()
def cli():
    pass


@cli.group()
def pipeline():
    pass


@pipeline.command()
def create():
    compile_()


if __name__ == "__main__":
    cli()
