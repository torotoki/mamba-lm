from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset
from mamba import MambaConfig, Mamba
import multiprocessing as mp

num_proc = max(1, mp.cpu_count() - 1)

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

ds = load_dataset("roneneldan/TinyStories").map(
    encode, batched=1000, num_proc=num_proc,
)

collator = DataCollatorWithPadding(tokenizer=tok)

config = MambaConfig(vocab_size=tok.vocab_size, d_model=128, num_hidden_layers=4)
model = Mamba(config)

print("=============== MODEL INFO ===============")
print(model)
print("#parameter:", sum(p.numel() for p in model.parameters()))

model = model.cuda()

args = TrainingArguments(
    output_dir="out-custom",
    torch_compile=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",  #TODO: epoch
    logging_steps=50,
    fp16=False,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['validation'],
    tokenizer=tok,
    data_collator=collator,
)

trainer.train()
