from dataclasses import dataclass
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, AutoConfig
from lightseq.training import LSGptEncoderLayer


@dataclass
class TrainingArguments:
    fp16: bool = True
    local_rank: int = -1


def test_gpt_layer():
    # text = "Replace me by any text you'd like."
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # encoded_input = tokenizer(text, return_tensors="pt")
    torch.random.manual_seed(1234)
    test_input = torch.empty(4, 64, 768).normal_().cuda()
    training_args = TrainingArguments()
    model = GPT2Model.from_pretrained("gpt2")
    config = AutoConfig.from_pretrained("gpt2")
    layer = model.h[0].cuda().train(False)
    base_output = layer(test_input)[0]
    ls_layer = LSGptEncoderLayer.from_huggingface(layer, training_args, config).train(
        False
    )
    ls_output = ls_layer(test_input)[0]
    np.testing.assert_allclose(
        base_output.detach().cpu().numpy(),
        ls_output.detach().cpu().numpy(),
        rtol=1e-2,
        atol=2e-1,
    )


if __name__ == "__main__":
    test_gpt_layer()
