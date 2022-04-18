"""
Export Hugging Face ViT models to hdf5 format.
"""
import os
import h5py
from collections import OrderedDict
from transformers import ViTModel
from lightseq.training.ops.pytorch.export import fill_hdf5_layer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
enc_layer_mapping_dict = OrderedDict(
    {
        # VIT is pre_layernorm
        # NOTE: add an additional "final" at the beginning for some weight
        # to distinguish them from "attention output *"
        "multihead_norm_scale": "layernorm_before weight",
        "multihead_norm_bias": "layernorm_before bias",
        "multihead_project_kernel_qkv": "attention attention query weight&&attention attention key weight&&attention attention value weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "attention attention query bias&&attention attention key bias&&attention attention value bias",
        "multihead_project_kernel_output": "attention output dense weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "attention output dense bias",
        "ffn_norm_scale": "layernorm_after weight",
        "ffn_norm_bias": "layernorm_after bias",
        "ffn_first_kernel": "intermediate dense weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "intermediate dense bias",
        "ffn_second_kernel": "final output dense weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "final output dense bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "conv_weight": "embeddings patch_embeddings projection weight",
        "conv_bias": "embeddings patch_embeddings projection bias",
        "position_embedding": "embeddings position_embeddings",
        "cls_embedding": "embeddings cls_token",
        "norm_scale": "layernorm weight",
        "norm_bias": "layernorm bias",
    }
)


def extract_vit_weights(
    output_file,
    model_dir,
    head_num,
    image_size,
    patch_size,
):
    # load var names
    encoder_state_dict = ViTModel.from_pretrained(model_dir).state_dict()

    # Insert additional "final" to some weight to prevent ambiguous match
    def _insert_final(key):
        l = key.split(".")
        l.insert(3, "final")
        return ".".join(l)

    encoder_state_dict = OrderedDict(
        [
            (_insert_final(k), v)
            if len(k.split(".")) > 3 and k.split(".")[3] == "output"
            else (k, v)
            for k, v in encoder_state_dict.items()
        ]
    )

    enc_var_name_list = list(encoder_state_dict.keys())

    # initialize output file
    output_file += ".hdf5"
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in enc_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name_split[2].isdigit():
            continue
        layer_id = int(name_split[2])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    # fill encoder_stack
    for layer_id in sorted(enc_tensor_names.keys()):
        fill_hdf5_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            hdf5_file,
            f"encoder_stack/{layer_id}/",
            enc_layer_mapping_dict,
        )

    # fill src_embedding - except for position embedding
    fill_hdf5_layer(
        enc_var_name_list,
        encoder_state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    # save number of layers metadata
    hdf5_file.create_dataset(
        "model_conf/n_encoder_stack", data=len(enc_tensor_names), dtype="i4"
    )
    # fill in model_conf
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset("model_conf/use_gelu", data=True, dtype="?")
    hdf5_file.create_dataset("model_conf/is_post_ln", data=False, dtype="?")
    hdf5_file.create_dataset("model_conf/image_size", data=image_size, dtype="i4")
    hdf5_file.create_dataset("model_conf/patch_size", data=patch_size, dtype="i4")

    hdf5_file.close()
    # read-in again to double check
    hdf5_file = h5py.File(output_file, "r")

    def _print_pair(key, value):
        value = value[()]
        print(f"{key}: {value}")

    list(map(lambda x: _print_pair(*x), hdf5_file["model_conf"].items()))


if __name__ == "__main__":
    output_lightseq_model_name = "lightseq_vit"
    input_huggingface_vit_model = "google/vit-base-patch16-224-in21k"
    head_number = 12
    image_size = 224
    patch_size = 16

    extract_vit_weights(
        output_lightseq_model_name,
        input_huggingface_vit_model,
        head_number,
        image_size,
        patch_size,
    )
