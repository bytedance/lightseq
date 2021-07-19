import torch

from lightseq.training import LSTransformer, LSCrossEntropyLayer, LSAdam

vocab_size, padding_idx = 1000, 0
batch_size, src_seq_len, trg_seq_len = 6, 10, 15

def create_data():
    src_tokens = torch.randint(padding_idx, vocab_size, (batch_size, src_seq_len), dtype=torch.long, device=torch.device("cuda:0"))
    trg_tokens = torch.randint(padding_idx, vocab_size, (batch_size, trg_seq_len), dtype=torch.long, device=torch.device("cuda:0"))
    target = trg_tokens.clone()[:, 1:]
    eos = torch.zeros((batch_size, 1), dtype=torch.long, device=torch.device("cuda:0"))
    target = torch.cat([target, eos], dim=-1)
    return src_tokens, trg_tokens, target

def create_model():
    transformer_config = LSTransformer.get_config(
        model="transformer-base",
        max_batch_tokens=4096,
        max_seq_len=256,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        num_encoder_layer=6,
        num_decoder_layer=6,
        fp16=True,
        local_rank=0
    )
    model = LSTransformer(transformer_config)
    model.to(dtype=torch.half, device=torch.device("cuda:0"))
    return model

def create_criterion():
    ce_config = LSCrossEntropyLayer.get_config(
        max_batch_tokens=4096,
        padding_idx=padding_idx,
        epsilon=0.0,
        fp16=True,
        local_rank=0
    )
    loss_fn = LSCrossEntropyLayer(ce_config)
    loss_fn.to(dtype=torch.half, device=torch.device("cuda:0"))
    return loss_fn

if __name__ == "__main__":
    src_tokens, trg_tokens, target = create_data()
    model = create_model()
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

    print("========================TEST========================")
    model.eval()
    encoder_out, encoder_padding_mask = model.encoder(src_tokens)
    predict_tokens = trg_tokens[:, :1]
    for _ in range(trg_seq_len):
        output = model.decoder(predict_tokens, encoder_out, encoder_padding_mask)
        output = torch.reshape(torch.argmax(output, dim=-1), (batch_size, -1))
        predict_tokens = torch.cat([predict_tokens, output[:, -1:]], dim=-1)
    predict_tokens = predict_tokens[:, 1:]
    print("target:\n", target)
    print("predict_tokens:\n", predict_tokens)