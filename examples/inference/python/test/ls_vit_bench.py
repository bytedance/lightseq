import time
import torch
import lightseq.inference as lsi
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel
from PIL import Image
import requests


def ls_vit(model, inputs):
    for _ in range(10):
        ls_output = model.infer(inputs)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(20):
        ls_output = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ls_output, (end_time - start_time) / 20


def hf_vit(model, inputs):
    for _ in range(10):
        hf_output = model(inputs.cuda())
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(20):
        hf_output = model(inputs.cuda())
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return hf_output, (end_time - start_time) / 20


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
    print(hf_output.last_hidden_state.size())
    # print(hf_output.logits.argmax(axis=1).detach().cpu().numpy())


def one_infer(inputs, ls_model, hf_model):
    # ls_generate(ls_model, inputs)
    hf_generate(hf_model, inputs)


class LightseqVitClassification:
    def __init__(self, ls_weight_path, hf_model):
        self.ls_vit = lsi.Vit(ls_weight_path, 8)
        self.classifier = hf_model.classifier

    def infer(self, inputs):
        last_hidden_states = self.ls_vit.infer(inputs)
        last_hidden_states = torch.Tensor(last_hidden_states).float().cuda()
        logits = self.classifier(last_hidden_states[:, 0, :])
        return logits


def main():

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open("/opt/tiger/lightseq/examples/inference/python/000000039769.jpg")
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs["pixel_values"]

    print("creating huggingface model...")
    hf_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").cuda()
    # hf_model.half()
    hf_model.eval()

    print("creating lightseq model...")
    # ls_model = LightseqVitClassification("lightseq_vit.hdf5", hf_model)
    ls_model = None

    # inputs = inputs.half()
    print("====================START warmup====================")
    one_infer(inputs, ls_model, hf_model)
    print("====================END warmup====================")
    batch_size = 1
    inputs = inputs.repeat((batch_size, 1, 1, 1))
    print("input size:", inputs.size())
    one_infer(inputs, ls_model, hf_model)


if __name__ == "__main__":
    main()
