
import torch
import lightseq.inference as lsi

from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, DataCollatorForSeq2Seq


src_lang = "en"
tgt_lang = "de"
raw_datasets = load_dataset("wmt19", "de-en")
mname = "facebook/wmt19-en-de"

tokenizer = FSMTTokenizer.from_pretrained(mname)

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
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=32)

# initialize metric
ls_metric = load_metric("sacrebleu")
hf_metric = load_metric("sacrebleu")

def run_ls():
    ls_model = lsi.Transformer("lightseq_fsmt_wmt19ende.hdf5", 32) # 2nd argument is max_batch_size
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels = [[label.strip()] for label in decoded_labels]

            # lightseq generation
            ls_generated_tokens = ls_model.infer(
                input_ids,
            )
            ls_generated_tokens = [ids[0] for ids in ls_generated_tokens[0]]
            ls_decoded_preds = tokenizer.batch_decode(ls_generated_tokens, skip_special_tokens=True)
            ls_decoded_preds = [pred.strip() for pred in ls_decoded_preds]
            ls_out.write("\n".join(ls_decoded_preds))
            ls_metric.add_batch(predictions=ls_decoded_preds, references=decoded_labels)

def run_hf():
    # load models 
    hf_model = FSMTForConditionalGeneration.from_pretrained(mname).cuda()

    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # save reference output here
            ref_out.write("\n".join([label.strip() for label in decoded_labels]))
            decoded_labels = [[label.strip()] for label in decoded_labels]

            # huggingface generation
            hf_generated_tokens = hf_model.generate(
                input_ids.cuda(),
                # attention_mask=batch["attention_mask"],
                max_length=256,
                num_beams=8,
            )
            hf_decoded_preds = tokenizer.batch_decode(hf_generated_tokens, skip_special_tokens=True)
            hf_decoded_preds = [pred.strip() for pred in hf_decoded_preds]
            hf_out.write("\n".join(hf_decoded_preds))
            hf_metric.add_batch(predictions=hf_decoded_preds, references=decoded_labels)


hf_out = open("translation-hf.txt", "w")
ls_out = open("translation-ls.txt", "w")
ref_out = open("translation-reference.txt", "w")

print(f"running huggingface translation...")
run_hf()
print(f"running lightseq translation...")
run_ls()

ls_eval_metric = ls_metric.compute()
hf_eval_metric = hf_metric.compute()
print({
    "lightseq bleu": ls_eval_metric["score"],
    "huggingface bleu": hf_eval_metric["score"]
})

hf_out.close()
ls_out.close()
ref_out.close()