import multiprocessing as mp
import random

import torch

from tests.util import (
    TestDecorator,
    split_custom_layer_grad,
    copy_grad_from_paras,
)
from tests.gen_test_layers import (
    gen_enc_layer,
    gen_dec_layer,
    gen_emb_layer,
    gen_ce_layer,
)


kt = TestDecorator()

config = kt.generate_config()
kt.dtypes = [torch.half if config.fp16 else torch.float]

custom_enc_layers, fairseq_enc_layers = gen_enc_layer(config)
custom_dec_layers, fairseq_dec_layers = gen_dec_layer(config)
custom_emb_layer, fairseq_emb_layer = gen_emb_layer(config)
custom_ce_layer, fairseq_ce_layer = gen_ce_layer(config)


@kt.case(rtol=1e-3, atol=1e-2)
def test_encoder_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    hidden_size = config.hidden_size

    print(
        f"(batch_size, seq_len, hidden_size): ({batch_size}, {seq_len}, {hidden_size})"
    )
    hidden_states = kt.rand((batch_size, seq_len, hidden_size))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)

    def custom():
        res = hidden_states.clone()
        for layer in custom_enc_layers:
            res = layer(res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.transpose(0, 1).contiguous().clone()
        for layer in fairseq_enc_layers:
            res = layer(res, self_attn_padding_mask)
        return [
            res.transpose(0, 1).contiguous().detach(),
        ]

    return custom, baseline


@kt.case(rtol=1e-2, atol=1e-2)
def test_encoder_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    hidden_size = config.hidden_size
    print(
        f"(batch_size, seq_len, hidden_size): ({batch_size}, {seq_len}, {hidden_size})"
    )

    shs = hidden_size * hidden_size
    hidden_states = kt.rand((batch_size, seq_len, hidden_size))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)

    # custom fw
    custom_enc_layers.zero_grad()
    res = hidden_states.clone()
    for layer in custom_enc_layers:
        res = layer(res, self_attn_padding_mask)
    custom_loss = (res / 1000).sum()

    # fairseq fw
    fairseq_enc_layers.zero_grad()
    res = hidden_states.transpose(0, 1).clone()
    for layer in fairseq_enc_layers:
        res = layer(res, self_attn_padding_mask)
    fairseq_loss = (res / 1000).sum()

    def custom():
        custom_enc_layers.zero_grad()
        custom_loss.backward(retain_graph=True)

        grad_list = []
        for i in range(config.num_layers - 1, -1, -1):
            """
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            """
            grads = split_custom_layer_grad(custom_enc_layers[i])
            grad_list.extend(
                [
                    grads[8],
                    grads[9],
                    grads[6],
                    grads[7],
                    grads[10],
                    grads[11],
                    grads[2],
                    grads[3],
                    grads[0][:shs],
                    grads[1][:hidden_size],
                    grads[0][shs : shs * 2],
                    grads[1][hidden_size : hidden_size * 2],
                    grads[0][shs * 2 : shs * 3],
                    grads[1][hidden_size * 2 : hidden_size * 3],
                    grads[4],
                    grads[5],
                ]
            )
        return grad_list

    def baseline():
        fairseq_enc_layers.zero_grad()
        fairseq_loss.backward(retain_graph=True)

        grad_list = []
        for i in range(config.num_layers - 1, -1, -1):
            curl = fairseq_enc_layers[i]
            cur_grads = copy_grad_from_paras(
                [
                    curl.fc2.weight,
                    curl.fc2.bias,
                    curl.fc1.weight,
                    curl.fc1.bias,
                    curl.final_layer_norm.weight,
                    curl.final_layer_norm.bias,
                    curl.self_attn.out_proj.weight,
                    curl.self_attn.out_proj.bias,
                    curl.self_attn.q_proj.weight,
                    curl.self_attn.q_proj.bias,
                    curl.self_attn.k_proj.weight,
                    curl.self_attn.k_proj.bias,
                    curl.self_attn.v_proj.weight,
                    curl.self_attn.v_proj.bias,
                    curl.self_attn_layer_norm.weight,
                    curl.self_attn_layer_norm.bias,
                ]
            )
            grad_list.extend(cur_grads)
        return grad_list

    return custom, baseline


@kt.case(rtol=1e-3, atol=1e-2)
def test_decoder_layer_forward():
    batch_size, enc_seq_len = kt.bs_sl()
    _, dec_seq_len = kt.bs_sl(batch_size)
    hidden_size = config.hidden_size

    print(
        f"(batch_size, enc_seq_len, dec_seq_len, hidden_size): "
        "({batch_size}, {enc_seq_len}, {dec_seq_len}, {hidden_size})"
    )

    hidden_states = kt.rand((batch_size, dec_seq_len, hidden_size))
    encoder_out = kt.rand((enc_seq_len, batch_size, hidden_size))
    incremental_state = None
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    self_attn_mask = kt.dec_self_attn_mask(dec_seq_len) * -1e8

    def custom():
        res = hidden_states.clone()
        for layer in custom_dec_layers:
            res, _, _ = layer(
                res,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.transpose(0, 1).clone()
        for layer in fairseq_dec_layers:
            res, _, _ = layer(
                res,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )
        return [
            res.transpose(0, 1).contiguous().detach(),
        ]

    return custom, baseline


@kt.case(rtol=1e-2, atol=1e-2)
def test_decoder_layer_backward():
    batch_size, enc_seq_len = kt.bs_sl()
    _, dec_seq_len = kt.bs_sl(batch_size)
    hidden_size = config.hidden_size
    print(
        f"(batch_size, enc_seq_len, dec_seq_len, hidden_size):({batch_size}, {enc_seq_len}, {dec_seq_len}, {hidden_size})"
    )

    shs = hidden_size * hidden_size
    hidden_states = kt.rand((batch_size, dec_seq_len, hidden_size))
    encoder_out = kt.rand((enc_seq_len, batch_size, hidden_size))
    incremental_state = None
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    self_attn_mask = kt.dec_self_attn_mask(dec_seq_len) * -1e8

    def custom():
        custom_dec_layers.zero_grad()
        cus_res = hidden_states.clone()
        cus_encoder_out = encoder_out.clone()
        for layer in custom_dec_layers:
            cus_res, _, _ = layer(
                cus_res,
                encoder_out=cus_encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
        custom_loss = (cus_res / 1000).sum()
        custom_loss.backward()
        grad_list = []
        for i in range(config.num_layers - 1, -1, -1):
            """
            0 attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            6 encdec_attn_qw, encdec_attn_qb, encdec_attn_ow, encdec_attn_ob, encdec_attn_nw, encdec_attn_nb,
            12 inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            18 encdec_attn_kvw, encdec_attn_kvb,
            """
            grads = split_custom_layer_grad(custom_dec_layers[i])
            grad_list.extend(
                [
                    grads[14],
                    grads[15],
                    grads[12],
                    grads[13],
                    grads[16],
                    grads[17],
                    grads[2],
                    grads[3],
                    grads[0][:shs],
                    grads[1][:hidden_size],
                    grads[0][shs : shs * 2],
                    grads[1][hidden_size : hidden_size * 2],
                    grads[0][shs * 2 : shs * 3],
                    grads[1][hidden_size * 2 : hidden_size * 3],
                    grads[4],
                    grads[5],
                    # encdec grad
                    grads[6],
                    grads[7],
                    grads[8],
                    grads[9],
                    grads[10],
                    grads[11],
                ]
            )
            if i == 0:
                grad_list.extend(
                    [
                        # encdec kv grad
                        grads[18][:shs],
                        grads[19][:hidden_size],
                        grads[18][shs : shs * 2],
                        grads[19][hidden_size : hidden_size * 2],
                    ]
                )
        return grad_list

    def baseline():
        fairseq_dec_layers.zero_grad()
        base_res = hidden_states.transpose(0, 1).clone()
        base_encoder_out = encoder_out.clone()
        for layer in fairseq_dec_layers:
            base_res, _, _ = layer(
                base_res,
                encoder_out=base_encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )
        fairseq_loss = (base_res / 1000).sum()
        fairseq_loss.backward()

        grad_list = []
        for i in range(config.num_layers - 1, -1, -1):
            curl = fairseq_dec_layers[i]
            cur_grads = copy_grad_from_paras(
                [
                    curl.fc2.weight,
                    curl.fc2.bias,
                    curl.fc1.weight,
                    curl.fc1.bias,
                    curl.final_layer_norm.weight,
                    curl.final_layer_norm.bias,
                    curl.self_attn.out_proj.weight,
                    curl.self_attn.out_proj.bias,
                    curl.self_attn.q_proj.weight,
                    curl.self_attn.q_proj.bias,
                    curl.self_attn.k_proj.weight,
                    curl.self_attn.k_proj.bias,
                    curl.self_attn.v_proj.weight,
                    curl.self_attn.v_proj.bias,
                    curl.self_attn_layer_norm.weight,
                    curl.self_attn_layer_norm.bias,
                    curl.encodec_attn.q_proj.weight,
                    curl.encodec_attn.q_proj.bias,
                    curl.encodec_attn.out_proj.weight,
                    curl.encodec_attn.out_proj.bias,
                    curl.encodec_attn_layer_norm.weight,
                    curl.encodec_attn_layer_norm.bias,
                ]
            )
            grad_list.extend(cur_grads)
            if i == 0:
                cur_grads = copy_grad_from_paras(
                    [
                        curl.encodec_attn.k_proj.weight,
                        curl.encodec_attn.k_proj.bias,
                        curl.encodec_attn.v_proj.weight,
                        curl.encodec_attn.v_proj.bias,
                    ]
                )
                grad_list.extend(cur_grads)
        return grad_list

    return custom, baseline


@kt.case(rtol=1e-3, atol=1e-2)
def test_decoder_layer_forward_inference():
    batch_size, enc_seq_len = kt.bs_sl()
    hidden_size = config.hidden_size
    print(
        f"(batch_size, enc_seq_len, hidden_size): "
        "({batch_size}, {enc_seq_len}, {hidden_size})"
    )

    # beam_size = random.randint(2, 5)
    # print(f"(batch_size, enc_seq_len, beam_size): ({batch_size}, {enc_seq_len}, {beam_size})")
    # ls_encoder_out = kt.rand((batch_size, enc_seq_len, hidden_size))
    # fs_encoder_out = ls_encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).reshape(-1, enc_seq_len, hidden_size)
    # ls_enc_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    # fs_enc_mask = ls_enc_mask.unsqueeze(1).repeat(1, beam_size, 1).reshape(-1, enc_seq_len)

    encoder_out = kt.rand((enc_seq_len, batch_size, hidden_size))
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)

    hidden_states_list = []
    max_step = 10
    for i in range(max_step):
        # hidden_states = kt.rand((batch_size*beam_size, 1, hidden_size))
        hidden_states = kt.rand((batch_size, 1, hidden_size))
        hidden_states_list.append(hidden_states)

    def custom():
        incremental_state = {}
        res_list = []
        for i in range(max_step):
            res = hidden_states_list[i].clone()
            for i in range(config.num_layers):
                res, _, _ = custom_dec_layers[i](
                    res,
                    # encoder_out=ls_encoder_out.transpose(0, 1),
                    # encoder_padding_mask=ls_enc_mask,
                    encoder_out=encoder_out,
                    encoder_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                )
            res_list.append(res)
        return [x.contiguous().detach() for x in res_list]

    def baseline():
        incremental_state = {}
        res_list = []
        for i in range(max_step):
            res = hidden_states_list[i].transpose(0, 1).clone()
            for i in range(config.num_layers):
                res, _, _ = fairseq_dec_layers[i](
                    res,
                    encoder_out=encoder_out,
                    encoder_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                )
            res_list.append(res)
        return [x.transpose(0, 1).contiguous().detach() for x in res_list]

    return custom, baseline


@kt.case(rtol=1e-3, atol=1e-3)
def test_embedding_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.int)
    # TODO: can not generate PAD in the middle of the sentences.
    input = kt.randint(config.padding_idx + 1, config.vocab_size, (batch_size, seq_len))
    input = input * (1 - padding_mask) + config.padding_idx * padding_mask

    def custom():
        res = custom_emb_layer(input)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        x = fairseq_emb_layer(input)
        return [
            x.contiguous().detach(),
        ]

    return custom, baseline


@kt.case(rtol=1e-3, atol=1e-3)
def test_embedding_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.int)
    input = kt.randint(config.padding_idx + 1, config.vocab_size, (batch_size, seq_len))
    input = input * (1 - padding_mask) + config.padding_idx * padding_mask
    loss_data = torch.randn(1, dtype=kt.dtype).sum()

    custom_emb_layer.zero_grad()
    custom_input = input.clone()
    res = custom_emb_layer(custom_input)
    custom_loss = (res / 1000).sum()
    custom_loss.data.copy_(loss_data)

    fairseq_emb_layer.zero_grad()
    fs_input = input.clone()
    res = fairseq_emb_layer(fs_input)
    fs_loss = (res / 1000).sum()
    fs_loss.data.copy_(loss_data)

    def custom():
        custom_emb_layer.zero_grad()
        custom_loss.backward(retain_graph=True)

        return [
            custom_emb_layer.embeddings.grad.contiguous().detach(),
        ]

    def baseline():
        fairseq_emb_layer.zero_grad()
        fs_loss.backward(retain_graph=True)

        return [
            fairseq_emb_layer.embeddings.weight.grad.contiguous().detach(),
        ]

    return custom, baseline


@kt.case()
def test_cross_entropy_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    vocab_size = random.randint(1000, 42000)
    print(f"(batch_size, seq_len, vocab_size): ({batch_size}, {seq_len}, {vocab_size})")

    inputs = kt.rand((batch_size, seq_len, vocab_size))
    targets = kt.randint(0, vocab_size, (batch_size, seq_len))
    targets_32 = targets.to(torch.int32)

    def custom():
        loss, cus_nll_loss = custom_ce_layer(inputs, targets_32)
        loss = loss.to(inputs)
        cus_nll_loss = cus_nll_loss.to(inputs)
        return [
            loss.contiguous().detach(),
            cus_nll_loss.contiguous().detach(),
        ]

    def baseline():
        loss, base_nll_loss = fairseq_ce_layer(inputs, targets)
        return [
            loss.contiguous().detach(),
            base_nll_loss.contiguous().detach(),
        ]

    return custom, baseline


@kt.case()
def test_cross_entropy_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    vocab_size = random.randint(1000, 42000)
    print(f"(batch_size, seq_len, vocab_size): ({batch_size}, {seq_len}, {vocab_size})")

    base_inputs = kt.rand((batch_size, seq_len, vocab_size)).requires_grad_()
    cus_inputs = base_inputs.clone().detach().requires_grad_()
    targets = kt.randint(0, vocab_size, (batch_size, seq_len))
    targets_32 = targets.to(torch.int32)

    custom_ce_layer.zero_grad()
    custom_loss, _ = custom_ce_layer(cus_inputs, targets_32)

    fairseq_ce_layer.zero_grad()
    base_loss, _ = fairseq_ce_layer(base_inputs, targets)

    def custom():
        custom_ce_layer.zero_grad()
        custom_loss.backward(retain_graph=True)

        return [
            cus_inputs.grad.contiguous().detach(),
        ]

    def baseline():
        fairseq_ce_layer.zero_grad()
        base_loss.backward(retain_graph=True)

        return [
            base_inputs.grad.contiguous().detach(),
        ]

    return custom, baseline


def main(epoch):
    print(">>>>>>>>>>>>>>>>>>>>>>Test epoch: {}>>>>>>>>>>>>>>>>>>>>>>".format(epoch))
    kt.run(
        [
            "test_encoder_layer_forward",
            "test_encoder_layer_backward",
            "test_decoder_layer_forward",
            "test_decoder_layer_backward",
            # "test_decoder_layer_forward_inference",
            "test_embedding_layer_forward",
            "test_embedding_layer_backward",
            "test_cross_entropy_layer_forward",
            "test_cross_entropy_layer_backward",
        ]
    )


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    for i in range(50):
        p = ctx.Process(target=main, args=(i,))
        p.start()
        p.join()
