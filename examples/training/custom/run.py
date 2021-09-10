import torch

from transformers import BertTokenizer
from lightseq.training import LSTransformer, LSCrossEntropyLayer, LSAdam


def create_data():
    # create Hugging Face tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    vocab_size = tokenizer.vocab_size
    sep_id = tokenizer.encode(
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
        sep_id,
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


def create_criterion():
    ce_config = LSCrossEntropyLayer.get_config(
        max_batch_tokens=2048,
        padding_idx=0,
        epsilon=0.0,
        fp16=True,
        local_rank=0,
    )
    loss_fn = LSCrossEntropyLayer(ce_config)
    loss_fn.to(dtype=torch.half, device=torch.device("cuda:0"))
    return loss_fn


if __name__ == "__main__":
    (
        tokenizer,
        src_text,
        src_tokens,
        trg_text,
        trg_tokens,
        target,
        sep_id,
        vocab_size,
        batch_size,
        src_seq_len,
        trg_seq_len,
    ) = create_data()
    model = create_model(vocab_size)
    loss_fn = create_criterion()
    opt = LSAdam(model.parameters(), lr=1e-5)

    print("========================TRAIN========================")
    model.train()
    for epoch in range(2000):
        output = model(src_tokens, trg_tokens)
        loss, _ = loss_fn(output, target)
        if epoch % 200 == 0:
            print("epoch {:03d}: {:.3f}".format(epoch, loss.item()))
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), "checkpoint.pt")
    print("model saved.")

    print("========================TEST========================")
    model.eval()
    # obtain encoder output and mask
    encoder_out, encoder_padding_mask = model.encoder(src_tokens)
    # use the first token as initial target input
    predict_tokens = trg_tokens[:, :1]
    cache = {}
    for _ in range(trg_seq_len - 1):
        # use cache to accelerate the inference
        output = model.decoder(
            predict_tokens[:, -1:], encoder_out, encoder_padding_mask, cache
        )
        # predict the next token
        output = torch.reshape(torch.argmax(output, dim=-1), (batch_size, -1))
        # concatenate the next token with previous tokens
        predict_tokens = torch.cat([predict_tokens, output], dim=-1)
    # pad all tokens after [SEP]
    mask = torch.cumsum(torch.eq(predict_tokens, sep_id).int(), dim=1)
    predict_tokens = predict_tokens.masked_fill(mask > 0, sep_id)
    # predict id to text
    predict_text = tokenizer.batch_decode(predict_tokens, skip_special_tokens=True)
    print(">>>>> source text")
    print("\n".join(src_text))
    print(">>>>> target text")
    print("\n".join(trg_text))
    print(">>>>> predict text")
    print("\n".join(predict_text))
