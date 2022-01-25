"""
Export DeltaLM models to protobuf/hdf5 format.
"""
import os
from collections import OrderedDict

import h5py
import numpy as np
from operator import attrgetter
from lightseq.training.ops.pytorch.export import gather_token_embedding, fill_pb_layer
from proto.deltalm_pb2 import Deltalm
from modeling_deltalm import DeltaLMForConditionalGeneration


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter, value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_. You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
enc_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "attention output LayerNorm weight",
        "multihead_norm_bias": "attention output LayerNorm bias",
        "multihead_project_kernel_qkv": "attention self query weight&&attention self key weight&&attention self value weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "attention self query bias&&attention self key bias&&attention self value bias",
        "multihead_project_kernel_output": "attention output dense weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "attention output dense bias",
        "ffn_norm_scale": "last output LayerNorm weight",
        "ffn_norm_bias": "last output LayerNorm bias",
        "ffn_first_kernel": "intermediate dense weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "intermediate dense bias",
        "ffn_second_kernel": "last output dense weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "last output dense bias",
    }
)


dec_layer_mapping_dict = OrderedDict(
    {
        "self_norm_scale": "attention output LayerNorm weight",
        "self_norm_bias": "attention output LayerNorm bias",
        "self_project_kernel_qkv": "attention self query weight&&attention self key weight&&attention self value weight&&expression_.transpose(0, 1)",
        "self_project_bias_qkv": "attention self query bias&&attention self key bias&&attention self value bias",
        "self_project_kernel_output": "attention output dense weight&&expression_.transpose(0, 1)",
        "self_project_bias_output": "attention output dense bias",
        "middle_ffn_norm_scale": "middle_output LayerNorm weight",
        "middle_ffn_norm_bias": "middle_output LayerNorm bias",
        "middle_ffn_first_kernel": "middle_intermediate dense weight&&expression_.transpose(0, 1)",
        "middle_ffn_first_bias": "middle_intermediate dense bias",
        "middle_ffn_second_kernel": "middle_output dense weight&&expression_.transpose(0, 1)",
        "middle_ffn_second_bias": "middle_output dense bias",
        "encdec_norm_scale": "crossattention output LayerNorm weight",
        "encdec_norm_bias": "crossattention output LayerNorm bias",
        "encdec_project_kernel_q": "crossattention self query weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_q": "crossattention self query bias",
        "encdec_project_kernel_output": "crossattention output dense weight&&expression_.transpose(0, 1)",
        "encdec_project_bias_output": "crossattention output dense bias",
        "ffn_norm_scale": "last output LayerNorm weight",
        "ffn_norm_bias": "last output LayerNorm bias",
        "ffn_first_kernel": "intermediate dense weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "intermediate dense bias",
        "ffn_second_kernel": "last output dense weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "last output dense bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "embeddings LayerNorm weight",
        "norm_bias": "embeddings LayerNorm bias",
    }
)

trg_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "embeddings LayerNorm weight",
        "norm_bias": "embeddings LayerNorm bias",
        "shared_bias": "final_logits_bias",
    }
)


def _get_encode_output_mapping_dict(dec_layer_num):
    encode_output_kernel_pattern = [
        "{0} crossattention self key weight&&{0} crossattention self value weight".format(ele)
        for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "{0} crossattention self key bias&&{0} crossattention self value bias".format(ele)
        for ele in range(dec_layer_num)
    ]

    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]
        ),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
    }


def save_bart_proto_to_hdf5(transformer: Deltalm, f: h5py.File):
    """Convert bart protobuf to hdf5 format to support larger weight."""
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
        "middle_ffn_norm_scale",
        "middle_ffn_norm_bias",
        "middle_ffn_first_kernel",
        "middle_ffn_first_bias",
        "middle_ffn_second_kernel",
        "middle_ffn_second_bias",
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


def extract_transformer_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    max_step,
    extra_decode_length=50,
    beam_size=4,
    length_penalty: float = 0,
    topk=1,
    topp=0.75,
    lang="en",
    only_decoder=True,
    save_proto=False,
):
    transformer = Deltalm()
    # load var names
    sd = DeltaLMForConditionalGeneration.from_pretrained(model_dir).state_dict()

    # Change key in statedict so that the check_rule works as expected
    reloaded = {}
    for k in sd:
        name_split = k.split(".")
        if len(name_split) > 3 and name_split[3].isdigit() and  '.'.join(name_split[4:6]) == 'output.LayerNorm':
            new_k = '.'.join(name_split[:4] + ['last'] + name_split[4:])
            print(f'Change sd key {k} -> {new_k}')
            reloaded[new_k] = sd[k]
        elif len(name_split) > 3 and name_split[3].isdigit() and  '.'.join(name_split[4:6]) == 'output.dense':
            new_k = '.'.join(name_split[:4] + ['last'] + name_split[4:])
            print(f'Change sd key {k} -> {new_k}')
            reloaded[new_k] = sd[k]
        else:
            reloaded[k] = sd[k]

    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in reloaded:
        if k.startswith("model.encoder."):
            encoder_state_dict[k] = reloaded[k]
        if k.startswith("model.decoder."):
            decoder_state_dict[k] = reloaded[k]
        if k == "model.shared.word_embeddings.weight":
            encoder_state_dict[k] = reloaded[k]
            decoder_state_dict[k] = reloaded[k]
        if k == "final_logits_bias":
            decoder_state_dict[k] = reloaded[k]

    dec_var_name_list = list(decoder_state_dict.keys())
    enc_var_name_list = list(encoder_state_dict.keys())
    # print('dec_var_name_list:', dec_var_name_list)
    # print('enc_var_name_list:', enc_var_name_list)
    # fill each encoder layer's params
    if not only_decoder:
        enc_tensor_names = {}
        for name in enc_var_name_list:
            name_split = name.split(".")
            if len(name_split) <= 3 or not name_split[3].isdigit():
                continue
            layer_id = int(name_split[3])
            enc_tensor_names.setdefault(layer_id, []).append(name)

        for layer_id in sorted(enc_tensor_names.keys()):
            fill_pb_layer(
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
        fill_pb_layer(
            dec_tensor_names[layer_id],
            decoder_state_dict,
            transformer.decoder_stack.add(),
            dec_layer_mapping_dict,
        )

    # fill src_embedding
    if not only_decoder:
        fill_pb_layer(
            enc_var_name_list,
            encoder_state_dict,
            transformer.src_embedding,
            src_emb_mapping_dict,
        )
        # bart position index starts from 2
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/configuration_bart.py#L208
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py#L821
        pos_emb_list = (
            # encoder_state_dict["model.encoder.embed_positions.weight"]
            ( encoder_state_dict["model.encoder.embeddings.position_embeddings.weight"]
                 + encoder_state_dict["model.encoder.embeddings.token_type_embeddings.weight"][0]) 
            # Although during training, only token_type==0 is used, but the weight has been modified. Since the weight is always the same, just add to position_embedding.weight
            .numpy()[
                2 : 2 + max_step, :
            ]  # because in huggingface bart, the position embedding starts from 2
            .reshape([-1])
            .tolist()
        )
        transformer.src_embedding.position_embedding[:] = pos_emb_list
        print(
            "model.encoder.embeddings.position_embeddings.weight -> src_embedding.position_embedding, shape: {}, conversion finished!".format(
                encoder_state_dict["model.encoder.embeddings.position_embeddings.weight"]
                .numpy()[2 : 2 + max_step, :]
                .shape
            )
        )
        src_tb, _ = gather_token_embedding(
            enc_var_name_list, encoder_state_dict, "shared", scale=False
        )
        print('src:emb:', _)

        transformer.src_embedding.token_embedding[:] = src_tb.flatten().tolist()

    # fill trg_embedding
    encode_output_mapping_dict = _get_encode_output_mapping_dict(len(dec_tensor_names))
    print('encode_output_mapping_dict:', encode_output_mapping_dict)
    trg_emb_mapping_dict.update(encode_output_mapping_dict)
    fill_pb_layer(
        dec_var_name_list,
        decoder_state_dict,
        transformer.trg_embedding,
        trg_emb_mapping_dict,
    )
    pos_emb_list = (
        # decoder_state_dict["model.decoder.embed_positions.weight"]
            ( decoder_state_dict["model.decoder.embeddings.position_embeddings.weight"] 
                + decoder_state_dict["model.decoder.embeddings.token_type_embeddings.weight"][0]) 
            # Although during training, only token_type==0 is used, but the weight has been modified. Since the weight is always the same, just add to position_embedding.weight
        .numpy()[2 : 2 + max_step, :]
        .reshape([-1])
        .tolist()
    )
    transformer.trg_embedding.position_embedding[:] = pos_emb_list
    print(
        "model.decoder.embeddings.position_embeddings.weight -> trg_embedding.position_embedding, shape: {}, conversion finished!".format(
            decoder_state_dict["model.decoder.embeddings.position_embeddings.weight"]
            .numpy()[:max_step, :]
            .shape
        )
    )
    # assert lang in LANG2ID
    ## Need to modify - shared doesnt work now.
    ## check if position embedding are the same as 'shared' -- yes (also check with huggingface from_pretrain utils)
    ## Deal with token type embeddings - currently all inputs has token_type_id == 0 and after training the token_type_embedding changes
    trg_tb, _ = gather_token_embedding(
        dec_var_name_list, decoder_state_dict, "shared", scale=False
    )
    print('trg:emb:', _)
    transformer.trg_embedding.token_embedding[:] = trg_tb.transpose().flatten().tolist()
    print(
        "token_embedding.weight -> trg_embedding.token_embedding, shape: {}, conversion finished!".format(
            trg_tb.transpose().shape
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

        ## middle ffn
        new_tmp_scale, new_tmp_bias = (
            decoder.middle_ffn_norm_scale[:],
            decoder.middle_ffn_norm_bias[:],
        )
        decoder.middle_ffn_norm_scale[:], decoder.middle_ffn_norm_bias[:] = (
            tmp_scale,
            tmp_bias,
        )
        print(
            "middle_ffn_norm_scale: {} -> {}\nmiddle_ffn_norm_bias: {} -> {}".format(
                new_tmp_scale[:3],
                decoder.middle_ffn_norm_scale[:3],
                new_tmp_bias[:3],
                decoder.middle_ffn_norm_bias[:3],
            )
        )
        tmp_scale, tmp_bias = new_tmp_scale[:], new_tmp_bias[:]
        ##

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
    transformer.model_conf.no_scale_embedding = True
    transformer.model_conf.use_gelu = True

    if save_proto:
        pass
        # output_file += ".pb"
        # print("Saving model to protobuf...")
        # print("Writing to {0}".format(output_file))
        # with tf.io.gfile.GFile(output_file, "wb") as fout:
        #     fout.write(transformer.SerializeToString())

        # transformer = Transformer()
        # with tf.io.gfile.GFile(output_file, "rb") as fin:
        #     transformer.ParseFromString(fin.read())
        # print(transformer.model_conf)
    else:
        output_file += ".hdf5"
        print("Saving model to hdf5...")
        print("Writing to {0}".format(output_file))
        f = h5py.File(output_file, "w")
        save_bart_proto_to_hdf5(transformer, f)
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
    # if save_proto is True, extension .pb will be added, otherwise .hdf5 is added
    model_name = "deltalm_converted_corrected"
    output_lightseq_model_name = 'lightseq_' + model_name

    input_huggingface_deltalm_model = model_name
    head_number = 12  # change this to 16 for "bart-large" model
    # in order to get score, we should use `beam_search` inference method
    generation_method = "beam_search"
    beam_size = 1
    max_step = 1024  # max step for generation, it decides GPU memory occupancy
    # maximum_generation_length = min(src_length + extra_decode_length, max_step)
    extra_decode_length = 50
    length_penalty = 0.0
    extract_transformer_weights(
        output_lightseq_model_name,
        input_huggingface_deltalm_model,
        head_num=head_number,  # layer number
        generation_method=generation_method,
        beam_size=beam_size,
        max_step=max_step,
        extra_decode_length=extra_decode_length,
        only_decoder=False,
        length_penalty=length_penalty,
        save_proto=False,
    )


