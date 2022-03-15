"""
Export LightSeq Transformer models to int8 protobuf format using post training quantization.
Refer to the `examples/training/custom` directory for more training details.
"""
import argparse
import time
import numpy as np
import torch
from transformers import BertTokenizer

from proto.quant_transformer_pb2 import QuantTransformer
from lightseq.training import (
    export_ls_config,
    export_ls_embedding_ptq,
    export_ls_encoder_ptq,
    export_ls_decoder_ptq,
    LSTransformer,
)
import lightseq.inference as lsi


# adjust this value to achieve better performance
global_act_clip_max = 16.0


def _extract_weight(state_dict):
    encoder_state_dict = {}
    decoder_state_dict = {}
    for k in state_dict:
        if k.startswith("encoder."):
            encoder_state_dict[k] = state_dict[k]
        if k.startswith("decoder."):
            decoder_state_dict[k] = state_dict[k]
    return encoder_state_dict, decoder_state_dict


def export_other_weights(ls_infer_model, state_dict):
    enc_norm_w = state_dict["encoder.layer_norm.weight"].flatten().tolist()
    enc_norm_b = state_dict["encoder.layer_norm.bias"].flatten().tolist()
    dec_norm_w = state_dict["decoder.layer_norm.weight"].flatten().tolist()
    dec_norm_b = state_dict["decoder.layer_norm.bias"].flatten().tolist()
    dec_shared_b = (
        torch.zeros(state_dict["decoder.embed_tokens.embeddings"].size(0))
        .flatten()
        .tolist()
    )
    ls_infer_model.src_embedding.norm_scale[:] = enc_norm_w
    ls_infer_model.src_embedding.norm_bias[:] = enc_norm_b
    ls_infer_model.trg_embedding.norm_scale[:] = dec_norm_w
    ls_infer_model.trg_embedding.norm_bias[:] = dec_norm_b
    ls_infer_model.trg_embedding.shared_bias[:] = dec_shared_b


def export_pb(state_dict, pb_path, pad_id, start_id, end_id, config):
    encoder_state_dict, decoder_state_dict = _extract_weight(state_dict)
    ls_infer_model = QuantTransformer()

    export_ls_embedding_ptq(
        ls_infer_model,
        encoder_state_dict,
        config.max_seq_len,
        True,
    )
    export_ls_embedding_ptq(
        ls_infer_model,
        decoder_state_dict,
        config.max_seq_len,
        is_encoder=False,
    )
    export_ls_encoder_ptq(
        ls_infer_model,
        encoder_state_dict,
        config.hidden_size,
        config.intermediate_size,
        act_clip_max=global_act_clip_max,
    )
    export_ls_decoder_ptq(
        ls_infer_model,
        decoder_state_dict,
        config.hidden_size,
        config.intermediate_size,
        config.num_decoder_layer,
        act_clip_max=global_act_clip_max,
    )
    export_other_weights(ls_infer_model, state_dict)
    export_ls_config(
        ls_infer_model,
        config.nhead,
        pad_id,
        start_id,
        end_id,
        config.num_encoder_layer,
        config.num_decoder_layer,
        beam_size=1,
    )

    with open(pb_path, "wb") as fout:
        fout.write(ls_infer_model.SerializeToString())


def create_data():
    # create Hugging Face tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.encode(
        tokenizer.special_tokens_map["pad_token"], add_special_tokens=False
    )[0]
    start_id = tokenizer.encode(
        tokenizer.special_tokens_map["cls_token"], add_special_tokens=False
    )[0]
    end_id = tokenizer.encode(
        tokenizer.special_tokens_map["sep_token"], add_special_tokens=False
    )[0]

    # source text to id
    src_text = [
        "What is the fastest library in the world?",
        "You are so pretty!",
        "What do you love me for?",
        "The sparrow outside the window hovering on the telephone pole.",
    ]
    src_tokens = tokenizer.batch_encode_plus(
        src_text, padding=True, return_tensors="pt"
    )
    src_tokens = src_tokens["input_ids"].to(torch.device("cuda:0"))
    batch_size, src_seq_len = src_tokens.size(0), src_tokens.size(1)

    # target text to id
    trg_text = [
        "I guess it must be LightSeq, because ByteDance is the fastest.",
        "Thanks very much and you are pretty too.",
        "Love your beauty, smart, virtuous and kind.",
        "You said all this is very summery.",
    ]
    trg_tokens = tokenizer.batch_encode_plus(
        trg_text, padding=True, return_tensors="pt"
    )
    trg_tokens = trg_tokens["input_ids"].to(torch.device("cuda:0"))
    trg_seq_len = trg_tokens.size(1)

    # left shift 1 token as the target output
    target = trg_tokens.clone()[:, 1:]
    trg_tokens = trg_tokens[:, :-1]

    return (
        tokenizer,
        src_text,
        src_tokens,
        trg_text,
        trg_tokens,
        target,
        pad_id,
        start_id,
        end_id,
        vocab_size,
        batch_size,
        src_seq_len,
        trg_seq_len,
    )


def create_config(vocab_size):
    transformer_config = LSTransformer.get_config(
        model="transformer-base",
        max_batch_tokens=2048,
        max_seq_len=512,
        vocab_size=vocab_size,
        padding_idx=0,
        num_encoder_layer=6,
        num_decoder_layer=6,
        fp16=True,
        local_rank=0,
    )
    return transformer_config


def ls_predict(ls_infer_model, src_tokens):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    ls_output = ls_infer_model.infer(src_tokens)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ls_output, end_time - start_time


def parse_args():
    parser = argparse.ArgumentParser(description="export LightSeq checkpoint", usage="")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="checkpoint_best.pt",
        help="path of LightSeq checkpoint",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    (
        tokenizer,
        src_text,
        src_tokens,
        trg_text,
        trg_tokens,
        target,
        pad_id,
        start_id,
        end_id,
        vocab_size,
        batch_size,
        src_seq_len,
        trg_seq_len,
    ) = create_data()

    args = parse_args()
    model_name = ".".join(args.model.split(".")[:-1])
    pb_path = f"{model_name}_ptq.pb"

    with open(args.model, "rb") as fin:
        state_dict = torch.load(fin, map_location=torch.device("cpu"))

    config = create_config(vocab_size)

    export_pb(state_dict, pb_path, pad_id, start_id, end_id, config)
    ls_infer_model = lsi.QuantTransformer(pb_path, 8)

    src_tokens_np = np.array(src_tokens.cpu())

    print("========================WARM UP========================")
    ls_predict(ls_infer_model, src_tokens_np)

    print("========================LIGHTSEQ TEST========================")
    ls_output, ls_time = ls_predict(ls_infer_model, src_tokens_np)
    ls_output = [ids[0] for ids in ls_output[0]]
    ls_predict_text = tokenizer.batch_decode(ls_output, skip_special_tokens=True)

    print(">>>>> source text")
    print("\n".join(src_text))
    print(">>>>> target text")
    print("\n".join(trg_text))
    print(">>>>> lightseq (infer) predict text")
    print("\n".join(ls_predict_text))
    print("lightseq (infer) predict time: {}ms".format(ls_time * 1000))
