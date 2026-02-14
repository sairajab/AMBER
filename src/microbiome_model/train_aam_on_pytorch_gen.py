import tensorflow as tf
import os
import datetime
from aam.models.utils import cos_decay_with_warmup
from aam.models import SequenceRegressor, TaxonomyEncoder, UniFracEncoder
from dataset import load_data, sample_data
import numpy as np
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from skbio import DistanceMatrix
from unifrac import unweighted
from typing import Iterable
from aam.callbacks import MeanAbsoluteError
from aam.callbacks import (
    ConfusionMatrx,
    SaveModel,
    _confusion_matrix,
    _mean_absolute_error,
)
class CVModel:
    def __init__(
        self, model: tf.keras.Model, train_data, val_data, output_dir, fold_label
    ):
        self.model: tf.keras.Model = model
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir
        self.fold_label = fold_label
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(
            output_dir,
            f"logs/fold-{self.fold_label}-{self.time_stamp}",
        )
        self.log_dir = os.path.join(
            output_dir, f"logs/fold-{self.fold_label}-{self.time_stamp}"
        )

    def fit_fold(
        self,
        loss,
        epochs,
        model_save_path,
        metric="mae",
        patience=10,
        early_stop_warmup=50,
        callbacks=[],
        lr=1e-4,
        warmup_steps=10000,
    ):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Import data monitoring callbacks
        from aam.data_monitoring_callbacks import DataStatsCallback, LossComponentsCallback
        
        optimizer = tf.keras.optimizers.AdamW(
            cos_decay_with_warmup(lr, warmup_steps), beta_2=0.95
        )
        model_saver = SaveModel(model_save_path, 10, f"val_{metric}")
        
        # Add data monitoring callbacks
        #data_stats_callback = DataStatsCallback(self.train_data["dataset"], log_frequency=25, name="Training")
        #loss_components_callback = LossComponentsCallback(log_frequency=10)
                
        core_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                "val_loss", patience=patience, start_from_epoch=early_stop_warmup
            ),
            model_saver,
        ]
        self.model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)
                
        self.model.fit(
            self.train_data["dataset"],
            validation_data=self.val_data["dataset"],
            callbacks=[*callbacks, *core_callbacks],
            epochs=epochs
        )
        self.model.set_weights(model_saver.best_weights)
        self.metric_value = self.model.evaluate_metric(self.val_data["dataset"], metric)

    def save(self, path, save_format="keras"):
        self.model.save(path, save_format=save_format)

    def predict(self, dataset):
        return self.model.predict(dataset)
    
    

class GeneratorDataset:
    def __init__(self, tree_path, abundance_table, train_data, val_data, kmer_seqs=None, embedding_path=None, one_hot_seqs=None, random_vector=False, batch_size=4 ):
        self.abundance_table = abundance_table
        self.train_data = train_data
        self.val_data = val_data
        self.kmer_seqs = kmer_seqs
        self.embedding_path = embedding_path
        self.one_hot_seqs = one_hot_seqs
        self.random_vector = random_vector
        self.shift = 0
        self.scale = 100
        self.batch_size = batch_size
        self.tree_path = tree_path
        self.encoder_target = unweighted("../filtered_table.biom", self.tree_path)

    def torch_generator(self, validation=False):

        train_loader, val_loader = sample_data(self.abundance_table, self.train_data, self.val_data, kmer_seqs=self.kmer_seqs, embedding_path=self.embedding_path, one_hot_seqs=self.one_hot_seqs, random_vector=self.random_vector, batch_size=self.batch_size)

        for batch in (val_loader if validation else train_loader):
            embeddings = batch['embeddings']
            abundances = batch['abundances']
            targets = batch['outdoor_add_0']
            yield ((embeddings.numpy(), abundances.numpy()),(targets.numpy(), self._encoder_output(self.encoder_target, batch['SampleID'])))

    def get_data(self, validation=False):
        
        #gen = self.torch_generator(validation=validation)
        
        output_signature = ((
            tf.TensorSpec(shape=(None, None, 150), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32)),(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32))
        )

        tf_dataset = tf.data.Dataset.from_generator(
            lambda: self.torch_generator(validation=validation),
            output_signature=output_signature
        )
        
        dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        data_obj = {
            "dataset": dataset,
            "shift": self.shift,
            "scale": self.scale
        }
        return data_obj

    def _encoder_output(
        self,
        encoder_target: DistanceMatrix,
        sample_ids: Iterable[str]
    ) -> np.ndarray[float]:
        return encoder_target.filter(sample_ids).data

# class UniFracGenerator(GeneratorDataset):
#     def __init__(self, tree_path: str, **kwargs):
#         super().__init__(**kwargs)
#         self.tree_path = tree_path
#         if self.batch_size % 2 != 0:
#             raise Exception("Batch size must be multiple of 2")
#         self.encoder_target = self._create_encoder_target()
#         self.encoder_dtype = np.float32
#         self.encoder_output_type = tf.TensorSpec(
#             shape=[self.batch_size, self.batch_size], dtype=tf.float32
#         )

#     def _create_encoder_target(self) -> DistanceMatrix:
#         if not hasattr(self, "tree_path"):
#             return None

#         random = np.random.random(1)[0]
#         # temp_path = f"/tmp/temp{random}.biom"
#         # with biom_open(temp_path, "w") as f:
#         #     table.to_hdf5(f, "aam")
#         temp_path = f"/tmp/temp{random}.biom"
#         distances = unweighted("../filtered_table.biom", self.tree_path)
#         #os.remove(temp_path)
#         return distances


    
    
if __name__ == "__main__":
    
    
        heldout = "D15"
        tree_path = '../data/deblur-gg2-aligned-150bp-table.nwk'
        abundance_table, train_data , test_data, one_hot_seqs, distances = load_data(heldout = heldout, train_encoder=True)
        gen = GeneratorDataset(tree_path ,  abundance_table=abundance_table,train_data=train_data, val_data=test_data,one_hot_seqs=one_hot_seqs)
        
        train_data = gen.get_data()
        val_data = gen.get_data(validation=True)
        p_asv_limit = 2048
        p_embedding_dim = 128
        p_attention_heads = 4
        p_attention_layers = 4
        p_intermediate_size = 1024
        p_intermediate_activation = "gelu"
        p_dropout = 0.1
        p_no_freeze_base_weights = False
        p_penalty = 0.1
        p_nuc_penalty = 1.0
        p_max_bp = 150
        base_model = "unifrac"
        loss = tf.keras.losses.MeanSquaredError(reduction="none")
        model = SequenceRegressor(
                token_limit=p_asv_limit,
                embedding_dim=p_embedding_dim,
                attention_heads=p_attention_heads,
                attention_layers=p_attention_layers,
                intermediate_size=p_intermediate_size,
                intermediate_activation=p_intermediate_activation,
                shift=train_data["shift"],
                scale=train_data["scale"],
                dropout_rate=p_dropout,
                base_model=base_model,
                freeze_base=p_no_freeze_base_weights,
                num_tax_levels=5,
                penalty=p_penalty,
                nuc_penalty=p_nuc_penalty,
                max_bp=p_max_bp,
            )
        p_warmup_steps = 100000
        p_epochs = 100
        p_patience = 30
        p_early_stop_warmup = 50
        p_report_back = True
        p_lr = 1e-4

        out_dir = "aam_pytorch_gen_results"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        figure_path = f"{out_dir}/figures"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss)
        model_save_path = f"{out_dir}/{heldout}_unifrac_model.h5"
        cv_model = CVModel(model, train_data, val_data, out_dir, heldout)
        cv_model.fit_fold(
                loss,
                p_epochs,
                os.path.join(out_dir, f"model_run.keras"),
                metric="mae",
                patience=p_patience,
                early_stop_warmup=p_early_stop_warmup,
                callbacks=[
                    MeanAbsoluteError(
                        monitor="val_mae",
                        dataset=val_data["dataset"],
                        output_dir=os.path.join(figure_path, f"model_run.png"),
                        report_back=p_report_back,
                    )
                ],
                lr=p_lr,
                warmup_steps=p_warmup_steps,
            )
            
            