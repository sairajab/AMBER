import torch
import os
import random
import pandas as pd
import numpy as np
from biom import load_table
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from itertools import product
from unifrac import unweighted
import torch.nn.functional as F
from biom.util import biom_open
from torch.utils.data import Sampler
from sklearn.model_selection import train_test_split


from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.data.embedding_loader import compute_and_save_embeddings
from microbiome_model.data.optimized_embedding_loader import MemoryMappedEmbeddingLoader as EmbeddingLoader 


class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, n_bins=5):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_bins = n_bins

        # Create quantile bins
        self.binned_labels = pd.qcut(self.labels, q=n_bins, labels=False, duplicates='drop')
        self.indices = np.arange(len(labels))

        # Group sample indices by bin
        self.bins = {i: self.indices[self.binned_labels == i].tolist() for i in range(n_bins)}

        # Shuffle inside each bin (no fixed seed)
        for bin_indices in self.bins.values():
            random.shuffle(bin_indices)

        # Collect all indices while maintaining stratification
        self.all_indices = []
        bins_copy = {k: v.copy() for k, v in self.bins.items()}
        while any(bins_copy.values()):
            for bin_idx in range(n_bins):
                if bins_copy[bin_idx]:
                    self.all_indices.append(bins_copy[bin_idx].pop())

        # Pad the list to make it divisible by batch_size (optional but safe)
        remainder = len(self.all_indices) % self.batch_size
        if remainder > 0:
            pad = self.batch_size - remainder
            self.all_indices += self.all_indices[:pad]  # repeat from start

    def __iter__(self):
        return iter(self.all_indices)

    def __len__(self):
        return len(self.all_indices)



def one_body_out(sample_ids , donor_id = "D12"):
    
    test_ids = []
    train_samples = []

    
    for id in sample_ids:
        if donor_id in id:
            test_ids.append(id)
        else:
            train_samples.append(id) 
    
    #train_ids, val_ids = train_test_split(train_samples, test_size=0.1)
    
    return train_samples, test_ids
    
    


len_mers = 5
bases = ['A', 'C', 'G', 'T']
kmers = [''.join(p) for p in product(bases, repeat=len_mers)]
kmer_to_index = {kmer: idx+2 for idx, kmer in enumerate(kmers)}
base_to_index = {base: idx+2 for idx, base in enumerate(bases)}
CLS_IDX = 1
PAD_TOKEN_IDX = 0
def seq_to_kmer_indices(seq, k=5):
    return [kmer_to_index[seq[i:i+k]] for i in range(len(seq) - k + 1) if seq[i:i+k] in kmer_to_index]

def seq_to_base_indices(seq):
    return [base_to_index[base] for base in seq if base in base_to_index]

def pad_or_truncate(seq, max_len=288):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return torch.cat([seq, torch.full((max_len - len(seq),), PAD_TOKEN_IDX, dtype=torch.long)])

def one_hot_encode(sequence, max_len=288):
    """
    One-hot encode a nucleotide sequence.
    
    Args:
        sequence (str): Nucleotide sequence (A, C, G, T)
        max_len (int): Maximum length of the sequence
    
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (max_len, 4)
    """
    one_hot = torch.zeros(max_len, 4, dtype=torch.float32)
    for i, base in enumerate(sequence):
        if i < max_len:
            if base == 'A':
                one_hot[i, 0] = 1.0
            elif base == 'C':
                one_hot[i, 1] = 1.0
            elif base == 'G':
                one_hot[i, 2] = 1.0
            elif base == 'T':
                one_hot[i, 3] = 1.0
    return one_hot



def tokenize_sequences(sequences, tokenizer, max_len=512):
    tokenized_seqs = {}
    seq = "TACAGAGGGTGCAAGCGTTGTTCGGAATCATTGGGCGTAAAGGGCGCGTAGGCGGTTTATCAAGTCGAATGTGAAAGCCCAGGGCTCAACCTTGGAAGTGCATCCGAAACTGGTAGACTAGAATCTCGGAGAGGGTGGTGGAATTCCCAGTGTAGAGGTGAAATTCGTAGATATTGGGAGGAACACCGGTGGCGAAGGCGACCACCTGGACAGAGATTGACGCTGAGGCGCGAGAGCGTGGGGAGCAAACAGG"

    encoded = tokenizer(seq, 
                        truncation=True,
                        return_tensors='pt')

    input_ids = encoded['input_ids'][0].tolist()

    print("Number of non-padding tokens:", sum(t != tokenizer.pad_token_id for t in input_ids))
    print("Length of input_ids:", len(input_ids))
    print("Tokenizer model max length:", tokenizer.model_max_length)


    
    for header, seq in sequences.items():
        # Tokenize sequence into k-mers and convert to token IDs
        # `return_tensors` omitted since we're storing token IDs
        tokenized = tokenizer(seq, padding='max_length', truncation=True, max_length=max_len)
        tokenized_seqs[header] = tokenized['input_ids']
        print(len(seq))
        print(seq)
        print(tokenizer.pad_token_id)  # Should print 3
        # Count tokens not equal to padding id (3)
        count_non_padding = sum(t != 3 for t in tokenized['input_ids'])
        print("Number of non-padding tokens:", count_non_padding)
        print(tokenized['input_ids'], len(tokenized['input_ids']))
        break

    return tokenized_seqs

def one_hot_encode(input_ids, vocab_size):
    # input_ids: [batch_size, seq_len]
    return F.one_hot(input_ids, num_classes=vocab_size).float()

def self_trained_mlm(embedding_path, sequences, device):
    
    from transformers import AutoTokenizer, AutoModel

    model_path = "asv_bert_mlm_250/checkpoint-41820"  # e.g., "outputs/checkpoint-27880"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()  # Disable dropout etc.
    compute_and_save_embeddings(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        output_path=embedding_path,
        device=device,
        max_length=250  # Adjust as needed
    )

    
def dna_bert(embedding_path, sequences, device):
    
            config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
            tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            dnabert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
            dnabert_model = dnabert_model.to(device)

            
            compute_and_save_embeddings(
                    sequences=sequences,
                    tokenizer=tokenizer,
                    model=dnabert_model,
                    output_path=embedding_path,
                    device=device
            )
            
            # Free up memory
            #del dnabert_model
            torch.cuda.empty_cache()

def finetuned_dna_bert(embedding_path, sequences, device):
            checkpoint_path = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/initial_exps_microbiome/src/dnabert-finetuned-16s-no-pad/checkpoint-37170"

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            finetuned_model = AutoModel.from_pretrained(checkpoint_path,trust_remote_code=True)
            dnabert_model = finetuned_model.to(device)
            compute_and_save_embeddings(
                sequences=sequences,
                tokenizer=tokenizer,
                model=dnabert_model,
                output_path=embedding_path,
                device=device
            )
            
            # Free up memory
            #del dnabert_model
            torch.cuda.empty_cache()



class DataProcessor:
    def __init__(self, args):
        """
        Class to load and process microbiome/sequence data.
        
        Args:
            biom_file (str): Path to BIOM file with abundances
            sequence_file (str): Path to FASTA file (or IDs only)
            target_file (str): Path to sample targets file
            embedding_dir (str): Directory to store embeddings
            unifrac_tree (str, optional): Path to UniFrac tree
            train_encoder (bool): Whether to process for encoder training
            kmer_embedding (bool): Whether to process with k-mer embeddings
        """
        self.config = args


        # Loaded data
        self.table = None
        self.sequences = {}
        self.sample_targets = None
        self.distances = None
        self.unifrac_tree = None
        self.seqs_processed = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.indoor_samples = None
        self.outdoor_samples = None
        self.sample_seasons = None
        self.pmival_test_data = None
        self.sampleID_map_donorID = {}

    def load_biom(self):
        table = load_table(self.config.biom_file)
        self.table = table
        return table

    def load_sequences(self):
        """
        Here we’re just filling sequences with observation IDs.
        Replace with SeqIO.parse(self.sequence_file, "fasta") if needed.
        """
        if self.table is None:
            raise ValueError("BIOM table not loaded yet. Call load_biom() first.")

        s_ids = self.table.ids(axis="observation")
        self.sequences = {s_id: s_id for s_id in s_ids}
        return self.sequences

    def compute_unifrac(self):
        if self.unifrac_tree is None:
            return None

        rand = np.random.random(1)[0]
        temp_path = f"/tmp/temp{rand}.biom"
        with biom_open(temp_path, "w") as f:
            self.table.to_hdf5(f, "aam")
        self.distances = unweighted(temp_path, self.config.tree_path)
        os.remove(temp_path)
        return self.distances

    def load_targets(self):

        ids = self.table.ids()
        targets_df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        
        targets_df["sample_name"] = targets_df["sample_name"].astype(str)
        filtered = targets_df[targets_df["sample_name"].isin(ids)]
        filtered = filtered[filtered["dataset_type"] == "train"].copy()
        print("Number of samples in targets after filtering:", len(filtered))
        self.sample_targets = dict(
            zip(filtered["sample_name"], filtered["add_0c"] / 100)
        )

        self.sampleID_map_donorID = dict(zip(filtered["sample_name"], filtered["host_subject_id"]))

        return self.sample_targets
    
    def load_metadata(self, column="season", dataset_type="train"):
        """
        Load per-sample metadata (one-hot encoded) for a given dataset split.

        dataset_type: "train" or "test" or "all"
        """
        if self.table is None:
            raise ValueError("BIOM table not loaded yet. Call load_biom() first.")

        ids = self.table.ids()
        metadata_df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        metadata_df["sample_name"] = metadata_df["sample_name"].astype(str)
        self.sampleID_map_donorID = dict(zip(metadata_df["sample_name"], metadata_df["host_subject_id"]))

        filtered = metadata_df[metadata_df["sample_name"].isin(ids)].copy()

        if dataset_type != "all":
            filtered = filtered[filtered["dataset_type"] == dataset_type].copy()

        print("Number of samples in metadata after filtering:", len(filtered))

        if column == "season":
            _map = {
                "spring": [1, 0, 0, 0],
                "fall":   [0, 1, 0, 0],
                "summer": [0, 0, 1, 0],
                "winter": [0, 0, 0, 1],
            }
        elif column == "bi_month_name":
            _map = {
                "Jan-Feb": [1, 0, 0, 0, 0, 0],
                "Mar-Apr": [0, 1, 0, 0, 0, 0],
                "May-Jun": [0, 0, 1, 0, 0, 0],
                "Jul-Aug": [0, 0, 0, 1, 0, 0],
                "Sep-Oct": [0, 0, 0, 0, 1, 0],
                "Nov-Dec": [0, 0, 0, 0, 0, 1],
            }
        else:
            raise ValueError(f"Unknown metadata column: {column}")

        # Handle missing/unexpected values safely
        if filtered[column].isnull().any():
            missing = filtered[filtered[column].isnull()]["sample_name"].tolist()[:10]
            raise ValueError(f"Metadata column '{column}' has missing values for samples like: {missing}")

        unknown_vals = set(filtered[column].unique()) - set(_map.keys())
        if unknown_vals:
            raise ValueError(f"Metadata column '{column}' contains unknown values: {sorted(list(unknown_vals))}")

        meta_array = np.array(filtered[column].map(_map).tolist())
        self.sample_seasons = dict(zip(filtered["sample_name"], meta_array))
        return self.sample_seasons


    def load_targets_multitask(self):

        ids = self.table.ids()
        targets_df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        ## for some reasons sheds table has 13810 at the beginning of the sample ids
        ## concat all sample ids with 13810 for sheds data only
        #targets_df["SampleID"] = "13810." + targets_df["SampleID"].astype(str)
        filtered = targets_df[targets_df["sample_name"].isin(ids)]
        filtered = filtered[filtered["dataset_type"] == "train"].copy()
        print("Number of samples in targets after filtering:", len(filtered))
        ## create bool for indoor/outdoor        
        indoor_samples = filtered[filtered["env"] == "indoor"]["sample_name"].tolist()
        outdoor_samples = filtered[filtered["env"] != "indoor"]["sample_name"].tolist()

        print("Number of indoor samples:", len(indoor_samples))
        print("Number of outdoor samples:", len(outdoor_samples))

        self.sample_targets = {}
        self.sample_targets = dict(
            zip(filtered["sample_name"], filtered["add_0c"].astype(float) / 100)
        )
        self.indoor_samples = indoor_samples
        self.outdoor_samples = outdoor_samples
        return self.sample_targets
    
    def process_sequences(self):
        """
        Process sequences into embeddings/k-mers/encoder inputs depending on mode.
        """
        if self.config.kmer_embeddings:
            seqs_kmers = {}
            max_len = 289
            for header, seq in self.sequences.items():
                indices = [CLS_IDX] + seq_to_kmer_indices(seq)
                seqs_kmers[header] = pad_or_truncate(
                    torch.tensor(indices, dtype=torch.long), max_len=max_len
                )
            return seqs_kmers

        elif self.config.train_encoder:
            seqs_kmers = {}
            max_len = 150
            for header, seq in self.sequences.items():
                indices = seq_to_base_indices(seq)
                seqs_kmers[header] = pad_or_truncate(
                    torch.tensor(indices, dtype=torch.long), max_len=max_len
                )
            return seqs_kmers

        else:

            if not os.path.exists(self.config.embedding_file):
                print("Computing embeddings...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                finetuned_dna_bert(self.config.embedding_file, self.sequences, device)
            return self.config.embedding_file
        
    def match_samples(self):
        if self.table is None or self.sample_targets is None:
            raise ValueError("Table or sample targets not loaded.")

        sample_ids = set(self.table.ids())
        target_ids = set(self.sample_targets.keys())
        common_ids = sample_ids.intersection(target_ids)

        if not common_ids:
            raise ValueError("No matching sample IDs between table and targets.")

        # Filter table and targets to only include common IDs
        self.table = self.table.filter(common_ids, axis="sample", inplace=False)
        self.sample_targets = {k: v for k, v in self.sample_targets.items() if k in common_ids}

        print(f"Number of samples after matching: {len(common_ids)}")
        return self.table, self.sample_targets

    def run(self, multitask=False, column="season"):
        """Full pipeline."""
        self.load_biom()
        self.load_sequences()
        self.load_metadata(column=column)
        if self.config.tree_path:
            self.compute_unifrac()
        if multitask:
            self.load_targets_multitask()
        else:
            self.load_targets()
        self.match_samples()
        self.seqs_processed = self.process_sequences()
    
    def load_data(self, multitask=False, column="season"):

        self.run(multitask=multitask, column=column)
        sample_ids = list(self.sample_targets.keys())

        train_samples, test_samples = one_body_out(sample_ids, donor_id = self.config.heldout)
        # Create train/val target dictionaries
        train_targets = {sample_id: self.sample_targets[sample_id] for sample_id in train_samples}
        test_targets = {sample_id: self.sample_targets[sample_id] for sample_id in test_samples}
        self.train_data = (train_samples, train_targets)
        self.test_data = (test_samples, test_targets)
    

    def donor_aware_split(self, train_samples, val_ratio = 0.1):
        donor_to_samples = {}
        for s in train_samples:
            donor_id = self.sampleID_map_donorID.get(s, "Unknown")
            donor_to_samples.setdefault(donor_id, []).append(s)

        donors = list(donor_to_samples.keys())
        random.shuffle(donors)

        n_val_donors = max(1, int(len(donors) * val_ratio))
        ## indoor and outdoor stratified split by donors
        n_indoors = sum(1 for d in donors if any(s in self.indoor_samples for s in donor_to_samples[d]))
        n_outdoors = sum(1 for d in donors if any(s in self.outdoor_samples for s in donor_to_samples[d]))
        min_indoor = max(1, int(n_indoors * val_ratio))
        min_outdoor = max(1, int(n_outdoors * val_ratio))
        
        # Ensure we have at least one indoor and one outdoor donor in validation set
        indoor_donors = [d for d in donors if any(s in self.indoor_samples for s in donor_to_samples[d])]
        outdoor_donors = [d for d in donors if any(s in self.outdoor_samples for s in donor_to_samples[d])]
        
        random.shuffle(indoor_donors)
        random.shuffle(outdoor_donors)
        # Select indoor and outdoor donors for validation set
        val_indoor_donors = indoor_donors[:min_indoor]
        val_outdoor_donors = outdoor_donors[:min_outdoor]
        
        # Remaining donors (excluding selected validation donors)
        remaining_donors = [d for d in donors if d not in val_indoor_donors and d not in val_outdoor_donors]
        
        # If we have more remaining donors than needed, select randomly
        n_remaining_val_donors = n_val_donors - len(val_indoor_donors) - len(val_outdoor_donors)
        if n_remaining_val_donors > 0:
            val_remaining_donors = random.sample(remaining_donors, min(n_remaining_val_donors, len(remaining_donors)))
            val_donors = set(val_indoor_donors + val_outdoor_donors + val_remaining_donors)
        else:
            val_donors = set(val_indoor_donors + val_outdoor_donors)

        train_donors = set(donors) - val_donors

        val_samples = [s for d in val_donors for s in donor_to_samples[d]]
        train_samples = [s for d in train_donors for s in donor_to_samples[d]]

        print(f"Total donors: {len(donors)}, Val donors: {len(val_donors)}, Train donors: {len(train_donors)}"
             f" (Indoor val donors: {len(val_indoor_donors)}, Outdoor val donors: {len(val_outdoor_donors)})")
        print(f"indoor train donors: {[d for d in train_donors if any(s in self.indoor_samples for s in donor_to_samples[d])]}, outdoor train donors: {[d for d in train_donors if any(s in self.outdoor_samples for s in donor_to_samples[d])]}")
        print(f"Split {len(train_samples)} train samples and {len(val_samples)} val samples by donors.")
        return train_samples, val_samples
        
    def stratified_donor_split(self, train_samples, val_ratio=0.15, random_state=42):
        """
        Split train_samples into train/val at the donor (body) level,
        stratified by body-level attributes (season + env).
        
        Ensures no donor appears in both train and val sets.
        Singleton strata are merged into the nearest existing stratum.
        Val size is automatically bumped up if needed to cover all strata.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # Build donor -> samples mapping
        donor_to_samples = {}
        for s in train_samples:
            donor_id = self.sampleID_map_donorID.get(s, "Unknown")
            donor_to_samples.setdefault(donor_id, []).append(s)

        # Build body-level profile from metadata
        metadata_df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        metadata_df["sample_name"] = metadata_df["sample_name"].astype(str)
        
        # Get first occurrence of each attribute per donor
        donor_meta = metadata_df[metadata_df["sample_name"].isin(train_samples)].copy()
        body_profile = donor_meta.groupby("host_subject_id").agg(
            season=("season", "first"),
            facility=("facility", "first"),
            env=("env", "first"),
        ).reset_index()
        
        # Only keep donors that appear in our train_samples
        donors_in_split = list(donor_to_samples.keys())
        body_profile = body_profile[body_profile["host_subject_id"].isin(donors_in_split)].copy()
        
        # Use season + env as stratification key (coarser than season+facility+env)
        # This keeps strata count manageable for small val splits
        body_profile["strat_key"] = (
            body_profile["season"].fillna("unk") + "_" +
            body_profile["env"].fillna("unk")
        )
        
        # Merge singleton strata into the nearest existing stratum
        counts = body_profile["strat_key"].value_counts()
        good_keys = set(counts[counts >= 2].index)

        for idx in body_profile.index:
            if body_profile.loc[idx, "strat_key"] not in good_keys:
                row_season = str(body_profile.loc[idx, "season"])
                row_env = str(body_profile.loc[idx, "env"])
                
                # Try to find a good stratum matching season first
                candidates = [k for k in good_keys if k.startswith(row_season)]
                if not candidates:
                    candidates = [k for k in good_keys if row_env in k]
                if not candidates:
                    candidates = [counts.idxmax()]
                
                best = max(candidates, key=lambda k: counts.get(k, 0))
                body_profile.loc[idx, "strat_key"] = best

                counts = body_profile["strat_key"].value_counts()
                good_keys = set(counts[counts >= 2].index)

        donor_ids = body_profile["host_subject_id"].values
        strat_labels = body_profile["strat_key"].values
        
        # Ensure val set is large enough to have >= 1 donor per stratum
        n_strata = len(set(strat_labels))
        n_val_requested = int(len(donor_ids) * val_ratio)
        n_val = max(n_strata, n_val_requested)
        actual_val_ratio = n_val / len(donor_ids)

        print(f"\nStratified donor split summary:")
        print(f"  Total donors: {len(donor_ids)}, Strata: {n_strata}")
        print(f"  Requested val_ratio: {val_ratio}, Effective val donors: {n_val} ({actual_val_ratio:.2f})")
        print(f"  Strata distribution: {dict(pd.Series(strat_labels).value_counts())}")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=actual_val_ratio, random_state=random_state)
        
        for train_idx, val_idx in sss.split(donor_ids, strat_labels):
            train_donors = set(donor_ids[train_idx])
            val_donors = set(donor_ids[val_idx])

        # Map back to samples
        new_train_samples = [s for d in train_donors for s in donor_to_samples[d]]
        val_samples = [s for d in val_donors for s in donor_to_samples[d]]

        # Print diagnostics
        val_profile = body_profile[body_profile["host_subject_id"].isin(val_donors)]
        train_profile = body_profile[body_profile["host_subject_id"].isin(train_donors)]
        print(f"  Train donors: {len(train_donors)}, Val donors: {len(val_donors)}")
        print(f"  Train samples: {len(new_train_samples)}, Val samples: {len(val_samples)}")
        print(f"  Train seasons: {dict(train_profile['season'].value_counts())}")
        print(f"  Val seasons: {dict(val_profile['season'].value_counts())}")
        print(f"  Train env: {dict(train_profile['env'].value_counts())}")
        print(f"  Val env: {dict(val_profile['env'].value_counts())}")
        
        return new_train_samples, val_samples
    
    def sample_data(self, epoch):
    # Create datasets

        train_samples, train_targets = self.train_data
        
        if self.val_data is None and epoch ==0:
            #train_samples, val_samples = train_test_split(train_samples, test_size=0.1 , random_state=42)
            #train_samples, val_samples = self.donor_aware_split(train_samples, val_ratio=0.1)
            train_samples, val_samples = self.stratified_donor_split(train_samples, val_ratio=0.1)
            train_targets = {sample_id: self.sample_targets[sample_id] for sample_id in train_samples}
            val_targets = {sample_id: self.sample_targets[sample_id] for sample_id in val_samples}
            self.val_data = (val_samples, val_targets)
            self.train_data = (train_samples, train_targets)
            print("Validation data created with", len(val_samples), "samples.")   
            donors = set()         
            for s in train_samples:
                donor_id = self.sampleID_map_donorID.get(s, "Unknown")
                donors.add(donor_id)
            print("Unique donors in validation set:", donors, len(donors))
        else:
            val_samples, val_targets = self.val_data


        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of validation samples: {len(self.val_data[0])}")
        print(f"Number of test samples: {len(self.test_data[0])}")
        
        train_table = self.table.filter(train_samples , inplace = False)
        val_table = self.table.filter(val_samples , inplace = False)

        if not self.config.kmer_embeddings and self.config.embedding_file is None:
                train_dataset = MicrobiomeSparseDataset(
                    biom_table=train_table,
                    one_hot_seqs=self.seqs_processed,
                    sample_targets=train_targets, 
                    sort_asvs=self.config.sort_asvs,
                    random_vec=False, 
                    seed = epoch 
                )
                val_dataset = MicrobiomeSparseDataset(
                    biom_table=val_table,
                    one_hot_seqs=self.seqs_processed,
                    sample_targets=val_targets, 
                    sort_asvs=self.config.sort_asvs,
                    random_vec=False,
                    seed = 9999
                )
        elif not self.config.kmer_embeddings and self.config.embedding_file is not None:
            # Use the embedding path to create the dataset
            embedding_loader = EmbeddingLoader(self.config.embedding_file)
            train_dataset = MicrobiomeSparseDataset(
                    biom_table=train_table,
                    sample_targets=train_targets,
                   sort_asvs=self.config.sort_asvs,
                    embedding_loader=embedding_loader,
                    random_vec=False,
                    seed = epoch,
                    env = (self.indoor_samples, self.outdoor_samples) ,
                    sample_metadata=self.sample_seasons
                )
                
            val_dataset = MicrobiomeSparseDataset(
                        biom_table=val_table,
                        sample_targets=val_targets,
                        sort_asvs=self.config.sort_asvs,
                        embedding_loader=embedding_loader,
                        random_vec=False,
                        seed = 9999 ,
                        env = (self.indoor_samples, self.outdoor_samples),
                        sample_metadata=self.sample_seasons
                    )
        elif self.config.kmer_embeddings and self.config.embedding_file is None:
            # Use the kmer_seqs to create the dataset
            train_dataset = MicrobiomeSparseDataset(
                    biom_table=train_table,
                    kmer_seqs=self.seqs_processed,
                    sort_asvs=self.config.sort_asvs,
                    sample_targets=train_targets, random_vec=False
                )
            val_dataset = MicrobiomeSparseDataset(
                        biom_table=val_table,
                        sort_asvs=self.config.sort_asvs,
                        kmer_seqs=self.seqs_processed,
                        sample_targets=val_targets, random_vec=False,
                        seed=9999
                    )
        
        train_y = train_dataset.get_targets()
        val_y = val_dataset.get_targets()
        train_targets = [train_y[k] for k in train_dataset.sample_ids]
        val_targets = [val_y[k] for k in val_dataset.sample_ids]

        return train_dataset, val_dataset

    def sample_test_data(self, random_vector=False):

        test_samples, test_targets = self.test_data
        test_table = self.table.filter(test_samples, inplace=False)


        if not self.config.kmer_embeddings and self.config.embedding_file is None:
                test_dataset = MicrobiomeSparseDataset(
                    biom_table=test_table,
                    one_hot_seqs=self.seqs_processed,
                    sort_asvs=self.config.sort_asvs,
                    sample_targets=test_targets, 
                    random_vec=random_vector, 
                    seed = None,
                    subsample=False
                )
        elif not self.config.kmer_embeddings and self.config.embedding_file is not None:
            # Use the embedding path to create the dataset
            embedding_loader = EmbeddingLoader(self.config.embedding_file)
            test_dataset = MicrobiomeSparseDataset(
                    biom_table=test_table,
                    sample_targets=test_targets,
                    sort_asvs=self.config.sort_asvs,
                    embedding_loader=embedding_loader,
                    random_vec=random_vector,
                    seed = None,
                    env = (self.indoor_samples, self.outdoor_samples),
                    sample_metadata=self.sample_seasons,
                    subsample=False
                )
        elif self.config.kmer_embeddings and self.config.embedding_file is None:
            # Use the kmer_seqs to create the dataset
            test_dataset = MicrobiomeSparseDataset(
                    biom_table=test_table,
                    kmer_seqs=self.seqs_processed,
                    sort_asvs=self.config.sort_asvs,
                    sample_targets=test_targets, random_vec=random_vector,
                    subsample=False
                )
        else:
            raise ValueError("Unknown configuration for test data sampling.")


        return test_dataset
    
    def load_test_data_pmival(self, column="season"):
        """
        Load PMI-val test split (dataset_type == 'test') including metadata and processed sequences/embeddings.
        Sets self.pmival_test_data = (test_samples, test_targets)
        """
        # Load table + sequences/embeddings the same way as train pipeline
        self.load_biom()
        self.load_sequences()

        # Ensure embeddings / one-hot seqs are ready (needed by dataset constructors)
        self.seqs_processed = self.process_sequences()

        # Read metadata + targets for TEST split
        ids = set(self.table.ids())

        df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        df["sample_name"] = df["sample_name"].astype(str)

        filtered = df[(df["sample_name"].isin(ids)) & (df["dataset_type"] == "test")].copy()
        print("Number of samples in pmival TEST after filtering:", len(filtered))

        # Targets
        test_targets = dict(zip(filtered["sample_name"], filtered["add_0c"].astype(float) / 100.0))
        test_samples = list(test_targets.keys())

        if len(test_samples) == 0:
            raise ValueError("No pmival test samples found after filtering. Check sample_name and dataset_type='test'.")

        # Metadata one-hot for the same test samples
        # Build seasons dict only on test split
        self.table = self.table.filter(test_samples, axis="sample", inplace=False)
        self.sample_seasons = {}  # reset to avoid stale train metadata
        self.load_metadata(column=column, dataset_type="test")

        # Optional: indoor/outdoor lists for test (if you use env in dataset)
        if "env" in filtered.columns:
            self.indoor_samples = filtered[filtered["env"] == "indoor"]["sample_name"].tolist()
            self.outdoor_samples = filtered[filtered["env"] != "indoor"]["sample_name"].tolist()
        else:
            self.indoor_samples, self.outdoor_samples = None, None

        self.pmival_test_data = (test_samples, test_targets)
        print(f"Number of pmival test samples: {len(test_samples)}")
        return self.pmival_test_data

    
    
    def sample_pmival_test_data(self, random_vector=False):
        test_samples, test_targets = self.pmival_test_data
        test_table = self.table.filter(test_samples, inplace=False)

        if not self.config.kmer_embeddings and self.config.embedding_file is None:
            test_dataset = MicrobiomeSparseDataset(
                biom_table=test_table,
                one_hot_seqs=self.seqs_processed,
                sample_targets=test_targets,
                random_vec=random_vector,
                seed=None
            )

        elif not self.config.kmer_embeddings and self.config.embedding_file is not None:
            embedding_loader = EmbeddingLoader(self.config.embedding_file)
            test_dataset = MicrobiomeSparseDataset(
                biom_table=test_table,
                sample_targets=test_targets,
                embedding_loader=embedding_loader,
                random_vec=random_vector,
                seed=None,
                env=(self.indoor_samples, self.outdoor_samples),
                sample_metadata=self.sample_seasons
            )

        elif self.config.kmer_embeddings and self.config.embedding_file is None:
            test_dataset = MicrobiomeSparseDataset(
                biom_table=test_table,
                kmer_seqs=self.seqs_processed,
                sample_targets=test_targets,
                random_vec=random_vector
            )
        else:
            raise ValueError("Unknown configuration for pmival test data sampling.")

        return test_dataset

    




class Arguments:
    def __init__(self, biom_file, metadata_file, embedding_file, tree_path = None,  heldout = "D12", embedding="DNABERT", normalize=False, sort_asvs=False):
        
        self.normalize = normalize
        self.heldout = heldout
        self.embedding = embedding
        self.sort_asvs = sort_asvs
        self.biom_file = biom_file
        self.metadata_file = metadata_file
        self.embedding_file = embedding_file
        self.kmer_embeddings  = False
        self.train_encoder = False
        self.tree_path = tree_path
        if self.embedding == "kmers":
            self.kmer_embeddings = True
        elif self.embedding == "train_encoder":
            self.train_encoder = True
            
        
def shuffle_indices(samples_dict):
    items = list(samples_dict.items())  
    random.shuffle(items)  
    return dict(items)


     


