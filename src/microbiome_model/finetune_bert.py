from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_TF32"] = "1"
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

def load_pretrain_model():

    model_name = "zhihan1996/DNABERT-2-117M"  # 6-mer version
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    return tokenizer, model

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"])

def fine_tune():
    dataset = load_dataset("text", data_files={"train": "../data/seqs_sheds.txt"})
    tokenizer, model = load_pretrain_model()
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
    output_dir="./dnabert-finetuned-16s-no-pad",
    per_device_train_batch_size=16,
    num_train_epochs=6,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=False,      # <--- turn off automatic mixed precision
    save_safetensors=False
    )
    sample = tokenized_dataset["train"][0]
    print("Input IDs shape:", len(sample["input_ids"]))
    print("Attention mask shape:", len(sample["attention_mask"]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

def run_inference():
    
    checkpoint_path = "dnabert-finetuned-16s/checkpoint-30975"  # adjust this if full path is needed

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint_path,trust_remote_code=True)
    
    # config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    # model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    model.eval()  # set to inference mode

    # Optional: move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Example input sequence
    seq = "ACGTAGCTAGCTGACTGATCGATCG"

    # Tokenize and send to device
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=250)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Input IDs shape:", inputs["input_ids"].shape)
    print(inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    print("K-mers:", tokens)
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Get mean-pooled embedding (optionally, you can use [CLS] token)
    #embedding = outputs.last_hidden_state.mean(dim=1)  # shape: [1, hidden_size]

    print(len(outputs))
    print(outputs[0].shape)  # print first 10 logits for the first token
    print(outputs[1].shape)
    #print(outputs.hidden_states[:,0,:])  # shape: [batch_size, sequence_length, hidden_size]

def test_():
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)


    seq = "TACAGAGGGTGCAAGCGTTGTTCGGAATCATTGGGCGTAAAGGGCGCGTAGGCGGTTTATCAAGTCGAATGTGAAAGCCCAGGGCTCAACCTTGGAAGTGCATCCGAAACTGGTAGACTAGAATCTCGGAGAGGGTGGTGGAATTCCCAGTGTAGAGGTGAAATTCGTAGATATTGGGAGGAACACCGGTGGCGAAGGCGACCACCTGGACAGAGATTGACGCTGAGGCGCGAGAGCGTGGGGAGCAAACAGG"

    # Ensure uppercase
    seq = seq.upper()

    # Tokenize WITHOUT max_length or padding
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
    ##run_inference()
    #test_()
