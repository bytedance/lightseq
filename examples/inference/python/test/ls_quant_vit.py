import time
import requests

from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
import lightseq.inference as lsi
from lightseq.training.ops.pytorch.quantization import qat_mode
from lightseq.training.ops.pytorch.torch_transformer_layers import (
    TransformerEncoderLayer,
)
from export.util import parse_args


def ls_vit(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    ls_output = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ls_output, end_time - start_time


def hf_vit(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    hf_output = model(inputs.cuda())
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return hf_output, end_time - start_time


def ls_generate(model, inputs):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_output, ls_time = ls_vit(model, inputs)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results (class predictions):")
    print(ls_output.argmax(axis=1).detach().cpu().numpy())


def hf_generate(model, inputs):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_output, hf_time = hf_vit(model, inputs)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results (class predictions):")
    print(hf_output.logits.argmax(axis=1).detach().cpu().numpy())


def one_infer(inputs, ls_model, hf_model):
    ls_generate(ls_model, inputs)
    # hf_generate(hf_model, inputs)


def gen_vit_config(config):
    num_patches = (config.image_size // config.patch_size) ** 2 + 1
    max_batch_size = 16
    vit_config = TransformerEncoderLayer.get_config(
        max_batch_tokens=num_patches * max_batch_size,
        max_seq_len=num_patches,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        nhead=config.num_attention_heads,
        attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
        activation_dropout_ratio=config.hidden_dropout_prob,
        hidden_dropout_ratio=config.hidden_dropout_prob,
        pre_layer_norm=True,
        fp16=True,
        local_rank=0,
        activation_fn="gelu",
    )
    return vit_config


class LSVITTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSVITTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, *args, **kwargs):
        ls_encoder_padding_mask = torch.zeros(hidden_states.size()[:-1]).to(
            hidden_states.device
        )
        output = super().forward(hidden_states, ls_encoder_padding_mask)
        return (output,)


def inject_ls_enc_layer(model, config):
    for i in range(config.num_hidden_layers):
        vit_config = gen_vit_config(config)
        model.vit.encoder.layer[i] = LSVITTransformerEncoderLayer(vit_config).cuda()
        model.vit.encoder.layer[i].apply(qat_mode)


class LightseqVitClassification:
    def __init__(self, ls_weight_path, hf_model):
        self.ls_vit = lsi.QuantVit(ls_weight_path, 8)
        self.classifier = hf_model.classifier

    def infer(self, inputs):
        last_hidden_states = self.ls_vit.infer(inputs)
        last_hidden_states = torch.Tensor(last_hidden_states).float().cuda()
        logits = self.classifier(last_hidden_states[:, 0, :])
        return logits


def main():
    args = parse_args()
    model_name = ".".join(args.model.split(".")[:-1])
    ckpt_path = f"{model_name}.bin"

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs["pixel_values"]

    print("initializing vit config...")
    config = ViTConfig.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=3,
        finetuning_task="image-classification",
    )

    print("creating huggingface model...")
    hf_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        config=config,
    ).cuda()
    inject_ls_enc_layer(hf_model, config)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.to("cuda:0")
    hf_model.eval()

    print("creating lightseq model...")
    ls_model = LightseqVitClassification(args.model, hf_model)

    print("====================START warmup====================")
    one_infer(inputs, ls_model, hf_model)
    print("====================END warmup====================")

    one_infer(inputs, ls_model, hf_model)


if __name__ == "__main__":
    main()
