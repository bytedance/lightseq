import torch

from transformers import BertTokenizer
from lightseq.training import LSTransformer, LSCrossEntropyLayer, LSAdam


def create_data():
    # create Hugging Face tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    vocab_size = tokenizer.vocab_size

    # source text to id
    src_text = "Which company do you work for?"
    src_tokens = tokenizer.encode(src_text, return_tensors="pt")
    src_tokens = src_tokens.to(torch.device("cuda:0"))
    batch_size, src_seq_len = src_tokens.size(0), src_tokens.size(1)

    # target text to id
    trg_text = "I guess it must be LightSeq, because ByteDance is the fastest."
    trg_tokens = tokenizer.encode(trg_text, return_tensors="pt")
    trg_tokens = trg_tokens.to(torch.device("cuda:0"))
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
    for epoch in range(1000):
        output = model(src_tokens, trg_tokens)
        loss, _ = loss_fn(output, target)
        if epoch % 100 == 0:
            print("epoch {:03d}: {:.3f}".format(epoch, loss.item()))
        loss.backward()
        opt.step()

    print("========================TEST========================")
    model.eval()
    # obtain encoder output and mask
    encoder_out, encoder_padding_mask = model.encoder(src_tokens)
    # use the first token as initial target input
    predict_tokens = trg_tokens[:, :1]
    for _ in range(trg_seq_len - 1):
        # TODO: use cache to accelerate the inference
        output = model.decoder(predict_tokens, encoder_out, encoder_padding_mask)
        # predict the next token
        output = torch.reshape(torch.argmax(output, dim=-1), (batch_size, -1))
        # concatenate the next token with previous tokens
        predict_tokens = torch.cat([predict_tokens, output[:, -1:]], dim=-1)
    predict_tokens = torch.squeeze(predict_tokens)
    # predict id to text
    predict_text = tokenizer.decode(predict_tokens, skip_special_tokens=True)
    print("source:\n", src_text)
    print("target:\n", trg_text)
    print("predict:\n", predict_text)
