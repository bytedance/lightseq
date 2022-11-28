import pathlib
import sys

from fairseq_cli.train import cli_main


def ls_cli_main(*args, **kwargs):
    user_path = pathlib.Path(__file__).parent.joinpath("fs_modules")
    sys.argv.extend(["--user-dir", str(user_path)])
    cli_main(*args, **kwargs)


if __name__ == "__main__":
    ls_cli_main()
