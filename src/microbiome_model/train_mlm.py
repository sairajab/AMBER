import os
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BertConfig
)
from Bio import SeqIO
from tokenizers.pre_tokenizers import Whitespace, Split
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel

# --- 1. Load ASV Sequences from FASTA ---
def load_fasta_sequences(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).upper()
        sequences.append({"id": record.id, "sequence": seq})
    return sequences

from tokenizers.processors import TemplateProcessing

def build_char_tokenizer(sequences, save_path):
    tokens = sorted(list(set("".join([seq["sequence"] for seq in sequences]))))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + tokens
    vocab_file = os.path.join(save_path, "vocab.txt")
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}

    os.makedirs(save_path, exist_ok=True)
    with open(vocab_file, "w") as f:
        f.write("\n".join(vocab))

    tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split("", behavior="isolated")

    # ✅ Add post-processor to insert special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", vocab_dict["[CLS]"]),
            ("[SEP]", vocab_dict["[SEP]"]),
        ],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    return fast_tokenizer


# --- 3. Custom Dataset ---
class ASVDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.dataset = hf_dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq = self.dataset[idx]["sequence"]
        #print(seq)
        encoded = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
             add_special_tokens=True
        )
        #print(encoded, encoded["input_ids"].shape)
        return {key: val.squeeze(0) for key, val in encoded.items()}

# --- 4. Load Data ---
fasta_path = "../data/seqs_sheds.fasta"
sequences = load_fasta_sequences(fasta_path)

import random
random.seed(42)  # Any number you like
random.shuffle(sequences)

# Split into train/test
train_data = sequences[:int(0.9 * len(sequences))]
val_data = sequences[int(0.9 * len(sequences)):]

# Convert to HuggingFace Dataset
hf_train = HFDataset.from_list(train_data)
hf_val = HFDataset.from_list(val_data)

# --- 5. Tokenizer & Dataset ---
tokenizer = build_char_tokenizer(sequences, save_path="./asv_tokenizer_250")
print(tokenizer("ATCG", add_special_tokens=True))  # Should return input_ids like: [CLS] A T C G [SEP]

# Should return input_ids like: [CLS] A T C G [SEP]

train_dataset = ASVDataset(hf_train, tokenizer, max_length=250)
val_dataset = ASVDataset(hf_val, tokenizer, max_length=250)


# --- 6. Define MLM Model ---
config = BertConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=250,
    hidden_size=256,
    num_attention_heads=4,
    num_hidden_layers=4,
    type_vocab_size=1,
)

model = BertForMaskedLM(config)
# --- 7. Training Setup ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

sample = train_dataset[0]
collated = data_collator([sample])
print("Input IDs:", collated["input_ids"])
print("Masked IDs:", collated["labels"])


training_args = TrainingArguments(
    output_dir="./asv_bert_mlm_250",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",      # ← requires newer version
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=15,
    weight_decay=0.01,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# # --- 8. Train ---
trainer.train()


#--- 9. Save ---
trainer.save_model("./asv_bert_mlm_250")
tokenizer.save_pretrained("./asv_tokenizer_250")


# from transformers import AutoModelForMaskedLM, AutoTokenizer

# model_path = "asv_bert_mlm_250/checkpoint-41820"  # Replace with your actual checkpoint folder
# model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained(model_path,  local_files_only=True)
# model.eval()

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
# )

# # Evaluate again
# metrics = trainer.evaluate()
# print(metrics)

# Output: {'eval_loss': 0.453, 'eval_runtime': ..., 'eval_samples_per_second': ..., ...}

