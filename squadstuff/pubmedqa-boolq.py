from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, \
    default_data_collator, Trainer

# dataset = load_dataset("pubmed_qa", "pqa_artificial")
dataset = load_dataset("pubmed_qa", "pqa_labeled")
# metric = load_metric('pubmed_qa')
# %%
max_length = 384  # The maximum length of a feature (question and context)
doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
model_checkpoint = "blizrys/biobert-v1.1-finetuned-pubmedqa"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pad_on_right = tokenizer.padding_side == "right"
assert isinstance(tokenizer, PreTrainedTokenizerFast)


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    # examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    question = examples["question"]
    context = [' '.join(x['contexts']) for x in examples["context"]]
    tokenized_examples = tokenizer(
        question if pad_on_right else context,
        context if pad_on_right else context,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        # stride=doc_stride,
        # return_overflowing_tokens=True,
        # return_offsets_mapping=True,
        padding="max_length",
    )

    tokenized_examples["label"] = [1 if x == 'yes' else 0 for x in examples["final_decision"]]

    return tokenized_examples


tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
batch_size = 16
args = TrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

#%%
data_collator = default_data_collator
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


#%%
trainer.train()
#%%
validation_features = dataset["train"].map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset["train"].column_names
)
raw_predictions = trainer.predict(validation_features)