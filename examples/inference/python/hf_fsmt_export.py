import os
from collections import OrderedDict

import tensorflow as tf
import h5py
import numpy as np
from operator import attrgetter
from utils import (
    fill_proto_layer,
    _get_encode_output_mapping_dict,
)
from transformer_pb2 import Transformer
from transformers import FSMTForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter, value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_. You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "self_attn_layer_norm weight",
        "multihead_norm_bias": "self_attn_layer_norm bias",
        "multihead_project_kernel_qkv": "self_attn q_proj weight&&self_attn k_proj weight&&self_attn v_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "self_attn q_proj bias&&self_attn k_proj bias&&self_attn v_proj bias",
        "multihead_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "self_attn out_proj bias",
        "ffn_norm_scale": "final_layer_norm weight",
        "ffn_norm_bias": "final_layer_norm bias",
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "fc2 bias",
    }
)

dec_layer_mapping_dict = OrderedDict(
    {
        "self_norm_scale": "self_attn_layer_norm weight",
        "self_norm_bias": "self_attn_layer_norm bias",
        "self_project_kernel_qkv": "self_attn q_proj weight&&self_attn k_proj weight&&self_attn v_proj weight&&expression_.transpose(0, 1)",
        "self_project_bias_qkv": "self_attn q_proj bias&&self_attn k_proj bias&&self_attn v_proj bias",
        "self_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "self_project_bias_output": "self_attn out_proj bias",
        "encdec_norm_scale": "encoder_attn_layer_norm weight",
        "encdec_norm_bias": "encoder_attn_layer_norm bias",
        "encdec_project_kernel_q": "encoder_attn q_proj weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_q": "encoder_attn q_proj bias",
        "encdec_project_kernel_output": "encoder_attn out_proj weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_output": "encoder_attn out_proj bias",
        "ffn_norm_scale": "final_layer_norm weight",
        "ffn_norm_bias": "final_layer_norm bias",
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "fc2 bias",
    }
)


def save_fsmt_proto_to_hdf5(transformer: Transformer, f: h5py.File):
    """Convert fsmt protobuf to hdf5 format to support larger weight."""
    MODEL_CONF_KEYS = [
        # model_conf
        "head_num",
        "beam_size",
        "extra_decode_length",
        "length_penalty",
        "src_padding_id",
        "trg_start_id",
        "diverse_lambda",
        "sampling_method",
        "topp",
        "topk",
        "trg_end_id",
        "is_post_ln",
        "no_scale_embedding",
        "use_gelu",
        "is_multilingual",
        "has_layernorm_embedding",
    ]

    EMBEDDING_KEYS = [
        # src_embedding
        # trg_embedding
        "token_embedding",
        "position_embedding",
        "norm_scale",
        "norm_bias",
        "encode_output_project_kernel_kv",
        "encode_output_project_bias_kv",
        "shared_bias",
        "lang_emb",
        "trg_vocab_mask",
    ]

    ENCODER_LAYER_KEYS = [
        # encoder_stack/{i}
        "multihead_norm_scale",
        "multihead_norm_bias",
        "multihead_project_kernel_qkv",
        "multihead_project_bias_qkv",
        "multihead_project_kernel_output",
        "multihead_project_bias_output",
        "ffn_norm_scale",
        "ffn_norm_bias",
        "ffn_first_kernel",
        "ffn_first_bias",
        "ffn_second_kernel",
        "ffn_second_bias",
    ]

    DECODER_LAYER_KEYS = [
        # decoder_stack/{i}
        "self_norm_scale",
        "self_norm_bias",
        "self_project_kernel_qkv",
        "self_project_bias_qkv",
        "self_project_kernel_output",
        "self_project_bias_output",
        "encdec_norm_scale",
        "encdec_norm_bias",
        "encdec_project_kernel_q",
        "encdec_project_bias_q",
        "encdec_project_kernel_output",
        "encdec_project_bias_output",
        "ffn_norm_scale",
        "ffn_norm_bias",
        "ffn_first_kernel",
        "ffn_first_bias",
        "ffn_second_kernel",
        "ffn_second_bias",
    ]
    base_attr_to_keys = {
        "src_embedding": EMBEDDING_KEYS,
        "trg_embedding": EMBEDDING_KEYS,
        "model_conf": MODEL_CONF_KEYS,
    }

    print(f"start converting protobuf to hdf5 format.")
    # load src_embedding, trg_embedding, model_conf
    for base_attr, keys in base_attr_to_keys.items():
        for key in keys:
            hdf5_key = f"{base_attr}/{key}"
            proto_attr = f"{base_attr}.{key}"

            if key not in dir(attrgetter(base_attr)(transformer)):
                print(f"key {key} not found in {base_attr}, skipping")
                continue

            print(f"loading transformer {proto_attr} -> {hdf5_key}")
            _data = attrgetter(proto_attr)(transformer)
            if type(_data) is str:
                print(
                    f"find type str, explicitly convert string to ascii encoded array."
                )
                # explict convert to array of char (int8) to avoid issues on string reading in C
                _data = np.array([ord(c) for c in _data]).astype(np.int8)
            f.create_dataset(hdf5_key, data=_data)

    # save number of layers metadata
    f.create_dataset("model_conf/n_encoder_stack", data=len(transformer.encoder_stack))
    f.create_dataset("model_conf/n_decoder_stack", data=len(transformer.decoder_stack))

    # load encoder_stack
    for layer_id, layer in enumerate(transformer.encoder_stack):
        for key in ENCODER_LAYER_KEYS:
            hdf5_key = f"encoder_stack/{layer_id}/{key}"
            proto_attr = key
            print(f"loading transformer.encoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    # load decoder_stack
    for layer_id, layer in enumerate(transformer.decoder_stack):
        for key in DECODER_LAYER_KEYS:
            hdf5_key = f"decoder_stack/{layer_id}/{key}"
            proto_attr = key
            print(f"loading transformer.decoder_stack {proto_attr} -> {hdf5_key}")
            f.create_dataset(hdf5_key, data=attrgetter(proto_attr)(layer))

    print(f"proto to hdf5 conversion completed.")


def extract_fsmt_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    extra_decode_length=50,
    beam_size=4,
    max_step=50,
    length_penalty: float = 0,
    topk=1,
    topp=0.75,
    lang="en",
    save_proto=False,
):
    transformer = Transformer()
    # load var names
    reloaded = FSMTForConditionalGeneration.from_pretrained(model_dir).state_dict()

    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in reloaded:
        if k.startswith("model.encoder."):
            encoder_state_dict[k] = reloaded[k]
        if k.startswith("model.decoder."):
            decoder_state_dict[k] = reloaded[k]

    dec_var_name_list = list(decoder_state_dict.keys())
    enc_var_name_list = list(encoder_state_dict.keys())

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in enc_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 3 or not name_split[3].isdigit():
            continue
        layer_id = int(name_split[3])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    for layer_id in sorted(enc_tensor_names.keys()):
        fill_proto_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            transformer.encoder_stack.add(),
            enc_layer_mapping_dict,
        )

    # fill each decoder layer's params
    dec_tensor_names = {}
    for name in dec_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 3 or not name.split(".")[3].isdigit():
            continue
        layer_id = int(name.split(".")[3])
        dec_tensor_names.setdefault(layer_id, []).append(name)

    for layer_id in sorted(dec_tensor_names.keys()):
        fill_proto_layer(
            dec_tensor_names[layer_id],
            decoder_state_dict,
            transformer.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    # fill src_embedding
    transformer.src_embedding.position_embedding[:] = (
        encoder_state_dict["model.encoder.embed_positions.weight"]
        .numpy()[2 : max_step + 2, :]
        .reshape([-1])
        .tolist()
    )
    print(
        "model.encoder.embed_positions.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
            encoder_state_dict["model.encoder.embed_positions.weight"].numpy().shape
        )
    )

    _, encoder_hidden_size = (
        encoder_state_dict["model.encoder.embed_tokens.weight"].numpy().shape
    )
    # FSMT does not have embedding layernorm - these are just placeholders for weights
    transformer.src_embedding.norm_scale[:] = [0 for _ in range(encoder_hidden_size)]
    transformer.src_embedding.norm_bias[:] = [0 for _ in range(encoder_hidden_size)]
    print(f"setting src_embedding.norm_scale to dummy value")
    print(f"setting src_embedding.norm_bias to dummy value")

    transformer.src_embedding.token_embedding[:] = (
        # scale embedding
        (
            encoder_state_dict["model.encoder.embed_tokens.weight"]
            * (encoder_hidden_size ** 0.5)
        )
        .numpy()
        .reshape([-1])
        .tolist()
    )

    print(
        "model.encoder.embed_tokens.weight -> src_embedding.token_embedding, shape: {}, conversion finished!".format(
            encoder_state_dict["model.encoder.embed_tokens.weight"].numpy().shape
        )
    )

    # fill trg_embedding
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    fill_proto_layer(
        dec_var_name_list,
        decoder_state_dict,
        transformer.trg_embedding,
        encode_output_mapping_dict,
    )

    transformer.trg_embedding.position_embedding[:] = (
        decoder_state_dict["model.decoder.embed_positions.weight"]
        .numpy()[2 : max_step + 2, :]
        .reshape([-1])
        .tolist()
    )
    print(
        "model.decoder.embed_positions.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
            decoder_state_dict["model.decoder.embed_positions.weight"].numpy().shape
        )
    )

    decoder_vocab_size, decoder_hidden_size = (
        decoder_state_dict["model.decoder.embed_tokens.weight"].numpy().shape
    )
    # FSMT does not have embedding layernorm - these are just placeholders for weights
    transformer.trg_embedding.norm_scale[:] = [0 for _ in range(decoder_hidden_size)]
    transformer.trg_embedding.norm_bias[:] = [0 for _ in range(decoder_hidden_size)]
    transformer.trg_embedding.shared_bias[:] = [0 for _ in range(decoder_vocab_size)]
    print(f"setting trg_embedding.norm_scale to dummy value")
    print(f"setting trg_embedding.norm_bias to dummy value")
    print(f"setting trg_embedding.shared_bias to dummy value")

    transformer.trg_embedding.token_embedding[:] = (
        # scale embedding
        (
            decoder_state_dict["model.decoder.embed_tokens.weight"]
            * (decoder_hidden_size ** 0.5)
        )
        .numpy()
        .transpose()
        .reshape([-1])
        .tolist()
    )

    print(
        "model.decoder.embed_tokens.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
            decoder_state_dict["model.decoder.embed_tokens.weight"]
            .numpy()
            .transpose()
            .shape
        )
    )

    # change encoder layer norm scale&bias position
    tmp_scale, tmp_bias = (
        transformer.src_embedding.norm_scale,
        transformer.src_embedding.norm_bias,
    )
    for i, encoder in enumerate(transformer.encoder_stack):
        print("***Fix encoder layer {} LayerNorm scale and bias***".format(i))
        new_tmp_scale, new_tmp_bias = (
            encoder.multihead_norm_scale[:],
            encoder.multihead_norm_bias[:],
        )
        encoder.multihead_norm_scale[:], encoder.multihead_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "multihead_norm_scale: {} -> {}\nmultihead_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                encoder.multihead_norm_scale[:3],
                new_tmp_bias[:3],
                encoder.multihead_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]

        new_tmp_scale, new_tmp_bias = (
            encoder.ffn_norm_scale[:],
            encoder.ffn_norm_bias[:],
        )
        encoder.ffn_norm_scale[:], encoder.ffn_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "ffn_norm_scale: {} -> {}\nffn_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                encoder.ffn_norm_scale[:3],
                new_tmp_bias[:3],
                encoder.ffn_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]
    transformer.src_embedding.norm_scale[:], transformer.src_embedding.norm_bias[:] = (
        tmp_scale,
        tmp_bias,
    )

    # change decoder layer norm scale&bias position
    tmp_scale, tmp_bias = (
        transformer.trg_embedding.norm_scale,
        transformer.trg_embedding.norm_bias,
    )
    for i, decoder in enumerate(transformer.decoder_stack):
        print("***Fix decoder layer {} LayerNorm scale and bias***".format(i))
        new_tmp_scale, new_tmp_bias = (
            decoder.self_norm_scale[:],
            decoder.self_norm_bias[:],
        )
        decoder.self_norm_scale[:], decoder.self_norm_bias[:] = tmp_scale, tmp_bias
        print(
            "self_norm_scale: {} -> {}\nself_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.self_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.self_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]

        new_tmp_scale, new_tmp_bias = (
            decoder.encdec_norm_scale[:],
            decoder.encdec_norm_bias[:],
        )
        decoder.encdec_norm_scale[:], decoder.encdec_norm_bias[:] = tmp_scale, tmp_bias
        print(
            "encdec_norm_scale: {} -> {}\nencdec_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.encdec_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.encdec_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]

        new_tmp_scale, new_tmp_bias = (
            decoder.ffn_norm_scale[:],
            decoder.ffn_norm_bias[:],
        )
        decoder.ffn_norm_scale[:], decoder.ffn_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "ffn_norm_scale: {} -> {}\nffn_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.ffn_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.ffn_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]
    transformer.trg_embedding.norm_scale[:], transformer.trg_embedding.norm_bias[:] = (
        tmp_scale,
        tmp_bias,
    )

    # fill in conf

    transformer.model_conf.head_num = head_num

    transformer.model_conf.beam_size = beam_size
    transformer.model_conf.length_penalty = length_penalty

    transformer.model_conf.extra_decode_length = extra_decode_length
    transformer.model_conf.src_padding_id = 1
    transformer.model_conf.trg_start_id = 2
    transformer.model_conf.trg_end_id = 2

    transformer.model_conf.sampling_method = generation_method
    transformer.model_conf.topk = topk
    transformer.model_conf.topp = topp
    transformer.model_conf.diverse_lambda = 0
    transformer.model_conf.is_post_ln = True
    transformer.model_conf.no_scale_embedding = (
        False  # scale_embedding is true for fsmt
    )
    transformer.model_conf.use_gelu = False
    # FSMT does NOT has layernorm for embedding
    transformer.model_conf.has_layernorm_embedding = False

    if save_proto:
        output_file += ".pb"
        print("Saving model to protobuf...")
        print("Writing to {0}".format(output_file))
        with tf.io.gfile.GFile(output_file, "wb") as fout:
            fout.write(transformer.SerializeToString())

        transformer = Transformer()
        with tf.io.gfile.GFile(output_file, "rb") as fin:
            transformer.ParseFromString(fin.read())
        print(transformer.model_conf)
    else:
        output_file += ".hdf5"
        print("Saving model to hdf5...")
        print("Writing to {0}".format(output_file))
        f = h5py.File(output_file, "w")
        save_fsmt_proto_to_hdf5(transformer, f)
        f.close()

        f = h5py.File(output_file, "r")

        def _print_pair(key, value):
            if key == "sampling_method":
                value = "".join(map(chr, value[()]))
            else:
                value = value[()]
            print(f"{key}: {value}")

        list(map(lambda x: _print_pair(*x), f["model_conf"].items()))
        f.close()


if __name__ == "__main__":
    # en->de, de->en pairs are available
    src_lang, tgt_lang = ("en", "de")  # ("de", "en")

    # if save_proto is True, extension .pb will be added, otherwise .hdf5 is added
    output_lightseq_model_name = f"lightseq_fsmt_wmt19{src_lang}{tgt_lang}"
    input_huggingface_fsmt_model = f"facebook/wmt19-{src_lang}-{tgt_lang}"
    print(
        f"exporting model {input_huggingface_fsmt_model} to {output_lightseq_model_name}"
    )

    extract_fsmt_weights(
        output_lightseq_model_name,
        input_huggingface_fsmt_model,
        head_num=16,
        # in order to get score, we should use `beam_search` inference method
        generation_method="beam_search",
        beam_size=4,
        max_step=256,
        # maximum_generation_length = min(src_length + extra_decode_length, max_step)
        extra_decode_length=256,
        length_penalty=1.0,
        save_proto=False,
    )
