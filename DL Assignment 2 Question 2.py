# Install required libraries
!pip install transformers datasets tqdm
!pip install torch torchvision torchaudio

# Import necessary packages
import os
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments


os.environ["WANDB_DISABLED"] = "true"

import kagglehub
download_path = kagglehub.dataset_download("paultimothymooney/poetry")
print("Dataset downloaded to:", download_path)


# Read and collect lyrics
all_lyrics = []
for filename in os.listdir(download_path):
    path = os.path.join(download_path, filename)
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read().strip()
        if content:
            all_lyrics.append(content)

print(f"Loaded {len(all_lyrics)} song lyric files.")



# Prepare dataset
lyrics_dataset = Dataset.from_dict({"text": all_lyrics})
lyrics_split = lyrics_dataset.train_test_split(test_size=0.1)


# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

model = GPT2LMHeadModel.from_pretrained(model_name)


# Tokenize function
def prepare_data(batch):
    encoded = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

tokenized_data = lyrics_split.map(prepare_data, batched=True, remove_columns=["text"])


# Training configuration
config = TrainingArguments(
    output_dir="./output_gpt2_lyrics",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"  # explicitly disable wandb/other tracking
)

# Setup trainer
trainer = Trainer(
    model=model,
    args=config,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"]
)



# Start fine-tuning
trainer.train()

# Save fine-tuned model and tokenizer
model.save_pretrained("./lyrics-gpt2")
tokenizer.save_pretrained("./lyrics-gpt2")


seed_line = "Eighteen years eighteen years"
input_tokens = tokenizer(seed_line, return_tensors="pt").to(model.device)

generated_output = model.generate(
    **input_tokens,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    early_stopping=True
)

print("\n Generated Lyrics \n")
print(tokenizer.decode(generated_output[0], skip_special_tokens=True))