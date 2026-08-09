from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import os
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_TF32"] = "1"
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

import math

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def load_pretrain_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)  # <-- added
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)  # <-- added
    return tokenizer, model

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"])

def fine_tune():
    # ── Load & split dataset ──────────────────────────────────────────────────
    raw = load_dataset("text", data_files={"train": "/s/chromatin/o/nobackup/Saira/Microbiome_Project/data/all_16S_sequences.txt"})

    # Split off 10% test, then 10% of remaining as validation (~9% of total)
    train_test = raw["train"].train_test_split(test_size=0.10, seed=42)
    train_val  = train_test["train"].train_test_split(test_size=0.10, seed=42)

    dataset = DatasetDict({
        "train":      train_val["train"],
        "validation": train_val["test"],
        "test":       train_test["test"],
    })

    print(f"Train size:      {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size:       {len(dataset['test'])}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenizer, model = load_pretrain_model()

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="/s/chromatin/o/nobackup/Saira/Microbiome_Project/dnabert-finetuned-16s-2",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        # Evaluate & save once per epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        save_safetensors=False,
        # Keep the best checkpoint based on validation loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    sample = tokenized_dataset["train"][0]
    print("Input IDs length:", len(sample["input_ids"]))
    print("Attention mask length:", len(sample["attention_mask"]))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],   # <-- validation set
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # optional
    )

    base_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(f"Base perplexity: {math.exp(base_results['eval_loss']):.4f}")
    return


    #trainer.train()

    # ── Evaluate on test set after training ───────────────────────────────────
    print("\n── Test set evaluation ──")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(test_results)

    # Derive perplexity from test loss
    perplexity = math.exp(test_results["eval_loss"])
    print(f"Test Perplexity: {perplexity:.4f}")

    # Save final model
    trainer.save_model("/s/chromatin/o/nobackup/Saira/Microbiome_Project/dnabert-finetuned-16s-2")
    tokenizer.save_pretrained("/s/chromatin/o/nobackup/Saira/Microbiome_Project/dnabert-finetuned-16s-2")


def run_inference():
    checkpoint_path = "dnabert-finetuned-16s-2/checkpoint-30975"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    seq = "ACGTAGCTAGCTGACTGATCGATCG"

    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=250)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Input IDs shape:", inputs["input_ids"].shape)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    print("K-mers:", tokens)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    print(len(outputs))
    print(outputs[0].shape)
    print(outputs[1].shape)


def test_():
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    seq = "TACAGAGGGTGCAAGCGTTGTTCGGAATCATTGGGCGTAAAGGGCGCGTAGGCGGTTTATCAAGTCGAATGTGAAAGCCCAGGGCTCAACCTTGGAAGTGCATCCGAAACTGGTAGACTAGAATCTCGGAGAGGGTGGTGGAATTCCCAGTGTAGAGGTGAAATTCGTAGATATTGGGAGGAACACCGGTGGCGAAGGCGACCACCTGGACAGAGATTGACGCTGAGGCGCGAGAGCGTGGGGAGCAAACAGG"
    seq = seq.upper()

    encoded = tokenizer(seq, return_tensors='pt')
    input_ids = encoded['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    non_pad = sum(t.item() != tokenizer.pad_token_id for t in input_ids)

    print("Input length:", len(seq))
    print("Token count:", len(input_ids))
    print("Non-padding tokens:", non_pad)
    print("First few tokens:", tokens)


if __name__ == "__main__":
    fine_tune()
    # run_inference()
    # test_()