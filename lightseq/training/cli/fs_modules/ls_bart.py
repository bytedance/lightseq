"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.models.bart import BARTHubInterface, BARTModel
from .ls_transformer import LSTransformerModel


logger = logging.getLogger(__name__)


@register_model("ls_bart")
class LSBARTModel(LSTransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        super(LSBARTModel, LSBARTModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[src_tokens.eq(eos), :].view(
                x.size(0), -1, x.size(-1)
            )[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        sample_break_mode="eos",
        **kwargs,
    ):
        from fairseq import hub_utils

        if "arch" in kwargs and not kwargs["arch"].startswith("ls_"):
            kwargs["args"] = "ls_" + kwargs["args"]
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def upgrade_state_dict_named(self, state_dict, name):
        self.upgrade_fairseq_state_dict_named(state_dict)
        if "encoder.version" not in state_dict:
            return

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # convert encoder and decoder from fairseq to lightseq
        def get_weight(key):
            return state_dict.pop(key).view(-1)

        enc_state_list = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ]
        dec_state_list = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "encoder_attn.q_proj.weight",
            "encoder_attn.q_proj.bias",
            "encoder_attn.out_proj.weight",
            "encoder_attn.out_proj.bias",
            "encoder_attn_layer_norm.weight",
            "encoder_attn_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ]

        # enc_state_list, dec_state_list = encoder_list, decoder_list
        for lid in range(self.args.encoder_layers):
            prefix = f"encoder.layers.{lid}."
            para = torch.cat([get_weight(prefix + k) for k in enc_state_list])
            state_dict[prefix + "para"] = para

        kv = {
            ".weight": [],
            ".bias": [],
        }
        for lid in range(self.args.decoder_layers):
            prefix = f"decoder.layers.{lid}.encoder_attn."
            for suffix in [".weight", ".bias"]:
                kv[suffix].append(get_weight(prefix + "k_proj" + suffix))
                kv[suffix].append(get_weight(prefix + "v_proj" + suffix))

        for lid in range(1, self.args.decoder_layers):
            prefix = f"decoder.layers.{lid}."
            para = torch.cat([get_weight(prefix + k) for k in dec_state_list])
            state_dict[prefix + "para"] = para

        dec_layer0_para = []
        for k in dec_state_list:
            dec_layer0_para.append(get_weight("decoder.layers.0." + k))
        dec_layer0_para.extend(kv[".weight"])
        dec_layer0_para.extend(kv[".bias"])
        state_dict["decoder.layers.0.para"] = torch.cat(dec_layer0_para)

        embed_map_dict = {
            "encoder.embed_tokens.weight": "encoder.embed_tokens.emb_lookup.weight",
            "encoder.embed_positions.weight": "encoder.embed_tokens.embed_positions.weight",
            "encoder.layernorm_embedding.weight": "encoder.embed_tokens.layernorm_embedding.weight",
            "encoder.layernorm_embedding.bias": "encoder.embed_tokens.layernorm_embedding.bias",
            "decoder.embed_tokens.weight": "decoder.embed_tokens.emb_lookup.weight",
            "decoder.embed_positions.weight": "decoder.embed_tokens.embed_positions.weight",
            "decoder.layernorm_embedding.weight": "decoder.embed_tokens.layernorm_embedding.weight",
            "decoder.layernorm_embedding.bias": "decoder.embed_tokens.layernorm_embedding.bias",
        }

        def rename_key(old, new):
            val = state_dict.pop(old)
            state_dict[new] = val

        for k, v in embed_map_dict.items():
            rename_key(k, v)

        # remove ignore keys
        ignore_keys = [
            "encoder.version",
            "decoder.version",
            "model.encoder.version",
            "model.decoder.version",
            "_float_tensor",
        ]
        for k in ignore_keys:
            state_dict.pop(k, None)

        # add new state
        embedding_weight = state_dict["encoder.embed_tokens.emb_lookup.weight"]
        state_dict["encoder.embed_tokens.embeddings"] = embedding_weight
        state_dict["encoder.embed_tokens.emb_quant._amax"] = torch.tensor(1.0)
        state_dict["decoder.embed_tokens.embeddings"] = embedding_weight
        state_dict["decoder.embed_tokens.emb_quant._amax"] = torch.tensor(1.0)
        if "decoder.output_projection.weight" not in state_dict:
            state_dict["decoder.output_projection.weight"] = embedding_weight

    def upgrade_fairseq_state_dict_named(self, state_dict):
        for lid in range(self.args.encoder_layers):
            name = f"encoder.layers.{lid}.self_attn"
            self.upgrade_attn_state_dict_named(state_dict, name)
            name = f"encoder.layers.{lid}"
            layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layer_norms.{}.{}".format(name, old, m)
                    if k in state_dict:
                        state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                        del state_dict[k]

        for i in range(self.args.decoder_layers):
            name = f"decoder.layers.{i}.self_attn"
            self.upgrade_attn_state_dict_named(state_dict, name)
            name = f"decoder.layers.{i}.encoder_attn"
            self.upgrade_attn_state_dict_named(state_dict, name)
            name = f"decoder.layers.{i}"
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

    def upgrade_attn_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def set_beam_size(self, beam):
        """Set beam size for efficient beamable enc-dec attention."""
        beamable = False
        for layer in self.decoder.layers:
            if layer.encoder_attn is not None:
                if hasattr(layer.encoder_attn, "set_beam_size"):
                    layer.encoder_attn.set_beam_size(beam)
                    beamable = True
        if beamable:
            self.encoder.reorder_encoder_out = self.encoder._reorder_encoder_out


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("ls_bart", "ls_bart_large")
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)


@register_model_architecture("ls_bart", "ls_bart_base")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)


@register_model_architecture("ls_bart", "ls_mbart_large")
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("ls_bart", "ls_mbart_base")
def mbart_base_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_base_architecture(args)


@register_model_architecture("ls_bart", "ls_mbart_base_wmt20")
def mbart_base_wmt20_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    mbart_base_architecture(args)
