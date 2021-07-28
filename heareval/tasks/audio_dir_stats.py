import click

import heareval.tasks.util.audio as audio_util


@click.command()
@click.argument("in-dir")
@click.argument("out-file")
@click.argument("exts", nargs=-1)
def audio_dir_stats(in_dir: str, out_file: str, exts):
    """Command line click endpoint to get audio directory stats"""
    audio_util.audio_dir_stats_wav(in_dir, out_file, exts)


if __name__ == "__main__":
    audio_dir_stats()
