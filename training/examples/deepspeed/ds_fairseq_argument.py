import json

from fairseq import options
from deepspeed.runtime.config_utils import dict_raise_error_on_duplicate_keys


def gen_ds_fairseq_arg():
    parser = options.get_training_parser()
    parser.add_argument(
        "--deepspeed_config",
        default=None,
        type=str,
        required=True,
        help="DeepSpeed json configuration file.",
    )
    fs_args = options.parse_args_and_arch(parser, modify_parser=None)

    ds_config = gen_ds_config(fs_args)
    delattr(fs_args, "deepspeed_config")
    return fs_args, ds_config


def gen_ds_config(fs_args):
    ds_config = json.load(
        open(fs_args.deepspeed_config),
        object_pairs_hook=dict_raise_error_on_duplicate_keys,
    )

    # Different parameters in fairseq and deepspeed have the same effect.
    # For these parameters, we extract it from fairseq arguments and put it
    # int the deepspeed config file
    ds_config["steps_per_print"] = fs_args.log_interval
    return ds_config
