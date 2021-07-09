import os
import torch
import sys
import lightseq.inference as lsi
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    FSMTForConditionalGeneration,
    FSMTTokenizer,
    DataCollatorForSeq2Seq,
)


def _export_model_weight(src_lang, tgt_lang):
    # if save_proto is True, extension .pb will be added, otherwise .hdf5 is added
    output_lightseq_model_name = f"/tmp/lightseq_fsmt_wmt19{src_lang}{tgt_lang}"
    input_huggingface_fsmt_model = f"facebook/wmt19-{src_lang}-{tgt_lang}"
    print(
        f"exporting model {input_huggingface_fsmt_model} to {output_lightseq_model_name}"
    )

    # hacky way to import hs_fsmt_export without break
    # `import lightseq.inference` from installed package
    import sys
    import pathlib

    project_root_path = pathlib.Path(__file__).parent.parent.parent
    target_folder = project_root_path / "examples" / "inference" / "python"
    sys.path.insert(0, str(target_folder))
    from hf_fsmt_export import extract_fsmt_weights

    extract_fsmt_weights(
        output_lightseq_model_name,
        input_huggingface_fsmt_model,
        head_num=16,
        # in order to get score, we should use `beam_search` inference method
        generation_method="beam_search",
        beam_size=4,
        max_step=256,
        # maximum_generation_length = min(src_length + extra_decode_length, max_step)
        extra_decode_length=256,
        length_penalty=1.0,
        save_proto=False,
    )


def _get_wmt19_eval_dataloader(tokenizer, src_lang, tgt_lang, name="de-en"):
    raw_datasets = load_dataset("wmt19", name)
    # load dataset: en -> de translation
    eval_dataset = raw_datasets["validation"]
    column_names = eval_dataset.column_names

    def preprocess_function(examples):
        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(inputs, padding=True, pad_to_multiple_of=8)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, padding=True, pad_to_multiple_of=8)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=16)
    return eval_dataloader


def _calculate_bleu(
    eval_dataloader, tokenizer, src_lang, tgt_lang, output_to_file=False
):
    print(f"calculating bleu for translation: {src_lang} -> {tgt_lang}")

    # load model
    hf_modelname = f"facebook/wmt19-{src_lang}-{tgt_lang}"
    print(f"loading huggingface model: {hf_modelname}")
    hf_model = FSMTForConditionalGeneration.from_pretrained(hf_modelname).cuda()

    ls_modelfile = f"/tmp/lightseq_fsmt_wmt19{src_lang}{tgt_lang}.hdf5"
    if not os.path.exists(ls_modelfile):
        print(f"model weight {ls_modelfile} not found. exporting model weight...")
        _export_model_weight(src_lang, tgt_lang)
    print(f"loading lightseq model: {ls_modelfile}")
    ls_model = lsi.Transformer(ls_modelfile, 16)  # 2nd argument is max_batch_size

    # initialize metric
    ls_metric = load_metric("sacrebleu")
    hf_metric = load_metric("sacrebleu")

    # initialize output file
    if output_to_file:
        hf_out = open("translation-hf.txt", "w")
        ls_out = open("translation-ls.txt", "w")
        ref_out = open("translation-reference.txt", "w")

    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # save reference output here
            if output_to_file:
                ref_out.write("\n".join([label.strip() for label in decoded_labels]))
            decoded_labels = [[label.strip()] for label in decoded_labels]

            # huggingface generation
            hf_generated_tokens = hf_model.generate(
                input_ids.cuda(),
                max_length=256,
                num_beams=4,
            )
            hf_decoded_preds = tokenizer.batch_decode(
                hf_generated_tokens, skip_special_tokens=True
            )
            hf_decoded_preds = [pred.strip() for pred in hf_decoded_preds]
            if output_to_file:
                hf_out.write("\n".join(hf_decoded_preds))
            hf_metric.add_batch(predictions=hf_decoded_preds, references=decoded_labels)

            # lightseq generation
            ls_generated_tokens = ls_model.infer(
                input_ids,
            )
            ls_generated_tokens = [ids[0] for ids in ls_generated_tokens[0]]
            ls_decoded_preds = tokenizer.batch_decode(
                ls_generated_tokens, skip_special_tokens=True
            )
            ls_decoded_preds = [pred.strip() for pred in ls_decoded_preds]
            if output_to_file:
                ls_out.write("\n".join(ls_decoded_preds))
            ls_metric.add_batch(predictions=ls_decoded_preds, references=decoded_labels)

    if output_to_file:
        hf_out.close()
        ls_out.close()
        ref_out.close()

    # calculate bleu
    ls_eval_metric = ls_metric.compute()
    hf_eval_metric = hf_metric.compute()
    results = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "lightseq bleu": ls_eval_metric["score"],
        "huggingface bleu": hf_eval_metric["score"],
    }
    return results


BLEU_DIFF_THRESHOLD = 0.3


def _check_result(results: dict, src_lang, tgt_lang):
    assert results["src_lang"] == src_lang
    assert results["tgt_lang"] == tgt_lang
    bleu_diff = results["huggingface bleu"] - results["lightseq bleu"]
    print(
        f"bleu_diff for {src_lang} compared to huggingface -> {tgt_lang}: {bleu_diff}"
    )
    assert bleu_diff <= BLEU_DIFF_THRESHOLD


def test_en_de_bleu():
    src_lang, tgt_lang = ("en", "de")
    tokenizer = FSMTTokenizer.from_pretrained(f"facebook/wmt19-{src_lang}-{tgt_lang}")
    eval_dataloader = _get_wmt19_eval_dataloader(tokenizer, src_lang, tgt_lang, "de-en")

    results = _calculate_bleu(
        eval_dataloader, tokenizer, src_lang, tgt_lang, output_to_file=False
    )
    _check_result(results, src_lang, tgt_lang)


def test_de_en_bleu():
    src_lang, tgt_lang = ("de", "en")
    tokenizer = FSMTTokenizer.from_pretrained(f"facebook/wmt19-{src_lang}-{tgt_lang}")
    eval_dataloader = _get_wmt19_eval_dataloader(tokenizer, src_lang, tgt_lang, "de-en")

    results = _calculate_bleu(
        eval_dataloader, tokenizer, src_lang, tgt_lang, output_to_file=False
    )
    _check_result(results, src_lang, tgt_lang)
