import numpy as np
from lightseq.training.ops.pytorch.export import apply_rule


def fill_hdf5_layer(
    tensor_names, state_dict, hdf5_file, hdf5_dataset_prefix, mapping_dict
):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        hdf5_file.create_dataset(
            hdf5_dataset_prefix + proto_name, data=target_tensor.flatten().tolist()
        )


def _gather_token_embedding(tensor_names, name2var_dict, tn_pattern):
    target_tn = []
    for tn in tensor_names:
        if (tn_pattern in tn.split(".")) and ("weight" in tn.split(".")):
            target_tn.append(tn)
            continue
    target_tensor = [name2var_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    print("token embedding shape is {}".format(target_tensor.shape))
    return target_tensor


def _get_encode_output_mapping_dict(dec_layer_num):
    encode_output_kernel_pattern = [
        "encoder_attn {0} k_proj weight&&encoder_attn {0} v_proj weight".format(ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "encoder_attn {0} k_proj bias&&encoder_attn {0} v_proj bias".format(ele)
        for ele in range(dec_layer_num)
    ]

    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]
        ),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
    }
