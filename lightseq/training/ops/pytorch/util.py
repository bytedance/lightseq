import torch


def copy_para(x):
    return torch.nn.Parameter(torch.empty_like(x).copy_(x))


def base_architecture(args):
    args.setdefault("hidden_size", 512)
    args.setdefault("intermediate_size", 2048)
    args.setdefault("nhead", 8)
    args.setdefault("attn_prob_dropout_ratio", 0.0)
    args.setdefault("activation_dropout_ratio", 0.0)
    args.setdefault("hidden_dropout_ratio", 0.1)
    args.setdefault("pre_layer_norm", True)


def transformer_base(args):
    base_architecture(args)


def transformer_big(args):
    args.setdefault("hidden_size", 1024)
    args.setdefault("intermediate_size", 4096)
    args.setdefault("nhead", 16)
    args.setdefault("attn_prob_dropout_ratio", 0.1)
    args.setdefault("activation_dropout_ratio", 0.1)
    base_architecture(args)


def bert_base(args):
    args.setdefault("hidden_size", 768)
    args.setdefault("intermediate_size", 3072)
    args.setdefault("nhead", 12)
    args.setdefault("attn_prob_dropout_ratio", 0.1)
    args.setdefault("activation_dropout_ratio", 0.1)
    args.setdefault("pre_layer_norm", False)
    base_architecture(args)


def bert_big(args):
    args.setdefault("pre_layer_norm", False)
    transformer_big(args)


MODEL_ARCH = {
    "transformer-base": transformer_base,
    "transformer-big": transformer_big,
    "bert-base": bert_base,
    "bert-big": bert_big,
}
