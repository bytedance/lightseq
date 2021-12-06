import pathlib
import sys

from deepspeed.launcher.runner import main


def ls_cli_main(*args, **kwargs):
    user_path = pathlib.Path(__file__).parent.joinpath("fs_modules")
    sys.argv.extend(["--user-dir", str(user_path)])
    main(*args, **kwargs)


if __name__ == "__main__":
    ls_cli_main()
