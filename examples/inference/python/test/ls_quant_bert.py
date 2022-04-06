import time

import torch
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import lightseq.inference as lsi
from lightseq.training.ops.pytorch.quantization import qat_mode, disable_quant
from lightseq.training.ops.pytorch.torch_transformer_layers import (
    BertEmbeddingLayer,
    TransformerEncoderLayer,
)
from export.fairseq.util import parse_args


def ls_bert(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    ls_output = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ls_output, end_time - start_time


def hf_bert(model, inputs, attn_mask):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    hf_output = model(inputs.to("cuda:0"), attention_mask=attn_mask.to("cuda:0"))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return hf_output, end_time - start_time


def ls_generate(model, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_output, ls_time = ls_bert(model, inputs_id)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results (class predictions):")
    print(ls_output.argmax(axis=2).detach().cpu().numpy())


def hf_generate(model, inputs_id, attn_mask):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_output, hf_time = hf_bert(model, inputs_id, attn_mask)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results (class predictions):")
    print(hf_output.logits.argmax(axis=2).detach().cpu().numpy())


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    ls_generate(ls_model, inputs_id)
    hf_generate(hf_model, inputs_id, attn_mask)


class LightseqBertClassification:
    def __init__(self, ls_weight_path, hf_model):
        self.ls_bert = lsi.QuantBert(ls_weight_path, 8)
        self.classifier = hf_model.classifier

    def infer(self, inputs):
        last_hidden_states = self.ls_bert.infer(inputs)
        last_hidden_states = torch.Tensor(last_hidden_states).float()
        logits = self.classifier(last_hidden_states.to("cuda:0"))
        return logits


def gen_bert_emb_config(config):
    bert_emb_config = BertEmbeddingLayer.get_config(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        max_batch_tokens=4096,
        max_seq_len=config.max_position_embeddings,
        padding_idx=config.pad_token_id,
        dropout=config.hidden_dropout_prob,
        fp16=True,
        local_rank=0,
    )
    bert_emb_config.type_vocab_size = config.type_vocab_size
    bert_emb_config.layer_norm_eps = config.layer_norm_eps
    return bert_emb_config


class LSHFTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
        ls_encoder_padding_mask = encoder_padding_mask / -10000.0
        ls_encoder_padding_mask = ls_encoder_padding_mask.squeeze()
        output = super().forward(hidden_states, ls_encoder_padding_mask)
        return (output, None, None, None)


def gen_bert_enc_config(config):
    bert_enc_config = TransformerEncoderLayer.get_config(
        max_batch_tokens=4096,
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        nhead=config.num_attention_heads,
        attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
        activation_dropout_ratio=config.hidden_dropout_prob,
        hidden_dropout_ratio=config.hidden_dropout_prob,
        pre_layer_norm=False,
        fp16=True,
        local_rank=0,
        activation_fn="gelu",
    )
    return bert_enc_config


def inject_ls_layer(model, config):
    bert_emb_config = gen_bert_emb_config(config)
    model.bert.embeddings = BertEmbeddingLayer(bert_emb_config)
    model.bert.embeddings.apply(qat_mode)

    for i in range(config.num_hidden_layers):
        bert_enc_config = gen_bert_enc_config(config)
        model.bert.encoder.layer[i] = LSHFTransformerEncoderLayer(
            bert_enc_config
        ).cuda()
        model.bert.encoder.layer[i].apply(qat_mode)


def main():
    args = parse_args()
    model_name = ".".join(args.model.split(".")[:-1])
    ckpt_path = f"{model_name}.bin"

    print("initializing bert config...")
    config = BertConfig.from_pretrained(
        "bert-base-uncased", num_labels=9, finetuning_task="ner"
    )

    print("initializing bert tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("creating huggingface model...")
    hf_model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", config=config
    )
    inject_ls_layer(hf_model, config)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.to("cuda:0")
    hf_model.eval()

    print("creating lightseq model...")
    ls_model = LightseqBertClassification(args.model, hf_model)

    sentences = [
        "EU rejects German call to boycott British lamb .",
        "-- Dimitris Kontogiannis , Athens Newsroom +301 3311812-4",
        "BayerVB sets C$ 100 million six-year bond .",
        "China says time right for Taiwan talks .",
    ]

    print("====================START warmup====================")
    warmup(tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    print("tokenizing the sentences...")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    ls_generate(ls_model, inputs_id)
    hf_generate(hf_model, inputs_id, attn_mask)


if __name__ == "__main__":
    main()
