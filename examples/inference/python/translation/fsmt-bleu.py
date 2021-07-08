
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, DataCollatorForSeq2Seq


# Get the language codes for input/target.
src_lang = "en"
tgt_lang = "de"
raw_datasets = load_dataset("wmt19", "de-en")
mname = "facebook/wmt19-en-de"

tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname).cuda()

eval_dataset = raw_datasets["validation"]
column_names = eval_dataset.column_names

# en -> de translation
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

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

# # DataLoaders creation:
# # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
# # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
# # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    pad_to_multiple_of=8,
)

eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=1)
metric = load_metric("sacrebleu")

for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
    with torch.no_grad():
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        generated_tokens = model.generate(
            input_ids,
            # attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=4,
        )
        # if not args.pad_to_max_length:
        #     # If we did not pad to max length, we need to pad the labels too
        #     labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
        # generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        # labels = accelerator.gather(labels).cpu().numpy()
        # if args.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
eval_metric = metric.compute()
print({"bleu": eval_metric["score"]})
