"""
Export LightSeq Transformer models to protobuf/hdf5 format.
Refer to the `examples/training/custom` directory for more training details.
"""
import time
import numpy as np
import torch
from transformers import BertTokenizer

from proto.transformer_pb2 import Transformer
from lightseq.training import (
    export_ls_config,
    export_ls_embedding,
    export_ls_encoder,
    export_ls_decoder,
    LSTransformer,
)
import lightseq.inference as lsi


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
    ls_infer_model = Transformer()

    export_ls_embedding(ls_infer_model, encoder_state_dict, config.max_seq_len, True)
    export_ls_embedding(ls_infer_model, decoder_state_dict, config.max_seq_len, False)
    export_ls_encoder(
        ls_infer_model,
        encoder_state_dict,
        config.hidden_size,
        config.intermediate_size,
    )
    export_ls_decoder(
        ls_infer_model,
        decoder_state_dict,
        config.hidden_size,
        config.intermediate_size,
        config.num_decoder_layer,
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


def create_model(vocab_size):
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
    model = LSTransformer(transformer_config)
    model.to(dtype=torch.half, device=torch.device("cuda:0"))
    return model


def ls_train_predict(ls_train_model, src_tokens, trg_tokens, batch_size):
    """
    NOTE: We do not use beam search here for implementation simplicity.
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    encoder_out, encoder_padding_mask = ls_train_model.encoder(src_tokens)
    predict_tokens = trg_tokens[:, :1]
    cache = {}
    for _ in range(trg_seq_len - 1):
        output = ls_train_model.decoder(
            predict_tokens[:, -1:], encoder_out, encoder_padding_mask, cache
        )
        output = torch.reshape(torch.argmax(output, dim=-1), (batch_size, -1))
        predict_tokens = torch.cat([predict_tokens, output], dim=-1)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return predict_tokens, end_time - start_time


def ls_predict(ls_infer_model, src_tokens):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    ls_output = ls_infer_model.infer(src_tokens)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ls_output, end_time - start_time


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

    ckpt_path = "checkpoint.pt"
    pb_path = "transformer.pb"

    with open(ckpt_path, "rb") as fin:
        state_dict = torch.load(fin, map_location=torch.device("cpu"))

    ls_train_model = create_model(vocab_size)
    ls_train_model.load_state_dict(state_dict)
    ls_train_model.eval()
    print("torch model loaded.")

    export_pb(state_dict, pb_path, pad_id, start_id, end_id, ls_train_model.config)
    ls_infer_model = lsi.Transformer(pb_path, 8)

    src_tokens_np = np.array(src_tokens.cpu())

    print("========================WARM UP========================")
    ls_train_predict(ls_train_model, src_tokens, trg_tokens, batch_size)
    ls_predict(ls_infer_model, src_tokens_np)

    print("========================TORCH TEST========================")
    predict_tokens, ls_train_time = ls_train_predict(
        ls_train_model, src_tokens, trg_tokens, batch_size
    )
    mask = torch.cumsum(torch.eq(predict_tokens, end_id).int(), dim=1)
    predict_tokens = predict_tokens.masked_fill(mask > 0, end_id)
    predict_text = tokenizer.batch_decode(predict_tokens, skip_special_tokens=True)

    print("========================LIGHTSEQ TEST========================")
    ls_output, ls_time = ls_predict(ls_infer_model, src_tokens_np)
    ls_output = [ids[0] for ids in ls_output[0]]
    ls_predict_text = tokenizer.batch_decode(ls_output, skip_special_tokens=True)

    print(">>>>> source text")
    print("\n".join(src_text))
    print(">>>>> target text")
    print("\n".join(trg_text))
    print(">>>>> lightseq (train) predict text")
    print("\n".join(predict_text))
    print(">>>>> lightseq (infer) predict text")
    print("\n".join(ls_predict_text))
    print("lightseq (train) predict time: {}ms".format(ls_train_time * 1000))
    print("lightseq (infer) predict time: {}ms".format(ls_time * 1000))
