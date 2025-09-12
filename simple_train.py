from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset
from mamba import MambaConfig, Mamba

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "This message is hello world",
    "Goodbye world until the sun rises",
    "Smashing the mics in the bar",
    "custom transformer model works",
    "trainer rocks",
    "pytorch is fun"
]

def encode(ex):
    out = tok(ex["text"], truncation=True, padding=False, max_length=64)
    return out
ds = Dataset.from_dict({"text": texts}).map(encode)
collator = DataCollatorWithPadding(tokenizer=tok)

config = MambaConfig(vocab_size=tok.vocab_size, d_model=128, num_hidden_layers=2)
model = Mamba(config)

args = TrainingArguments(
    output_dir="out-custom",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1000,
    eval_strategy="epoch",
    save_strategy="no",  #TODO: epoch
    logging_steps=1,
    fp16=False,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds.select(range(4)),
    eval_dataset=ds.select(range(4, len(ds))),
    tokenizer=tok,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("out-custom") 
