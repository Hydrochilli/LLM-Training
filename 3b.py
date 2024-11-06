

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_name ="ibm-granite/granite-3b-code-base-2k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Assuming you have structured data in JSONL format with 'prompt' and 'completion' fields
dataset = load_dataset("json", data_files="path_to_your_data.jsonl")
train_dataset = dataset["train"].map(lambda x: tokenizer(x["prompt"], padding=True, truncation=True, return_tensors="pt"))

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

