import lightning as L
from data_handler import CubesDataModule
from lightning_module import ReconstructionCASSI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

import datetime


data_dir = "./datasets_reconstruction/mst_datasets/cave_1024_28_train/" # Folder where the train dataset is
data_dir_test = "./datasets_reconstruction/mst_datasets/cave_1024_28_test/" # Folder where the test dataset is

torch.set_float32_matmul_precision('high')

datamodule = CubesDataModule(data_dir, data_dir_test, batch_size=1, crop_size = 512, num_workers=10)

model_name = "padut"  # Out of dgsmp, mst, dwmt, padut, duf, dauhst
name = f"training_wrongmis_reconstruction_amici_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
optics_model_file_path = "../system_specs/system_amiciwrongmis.yml"

log_dir = 'tb_logs'

train = True
run_on_cpu = False # Set to True if you prefer to run it on cpu

oversample = 4
nb_rays = 20

rays_file = "./rays/rays_amiciwrongmis.pt"
valid_file = "./rays/rays_valid_amiciwrongmis.pt"

mask_path = "./mask.pt"

logger = TensorBoardLogger(log_dir, name=name)

early_stop_callback = EarlyStopping(
                            monitor='val_loss',  # Metric to monitor
                            patience=500,        # Number of epochs to wait for improvement
                            verbose=True,
                            mode='min'          # 'min' for metrics where lower is better, 'max' for vice versa
                            )

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',      # Metric to monitor
    dirpath=f'checkpoints/{name}/',  # Directory path for saving checkpoints
    filename=f'best-checkpoint_amiciwrongmis_{model_name}',  # Checkpoint file name
    save_top_k=1,            # Save the top k models
    mode='min',              # 'min' for metrics where lower is better, 'max' for vice versa
    save_last=True           # Additionally, save the last checkpoint to a file named 'last.ckpt'
)

reconstruction_module = ReconstructionCASSI(net_model_name=model_name,
                                            optics_model_file_path=optics_model_file_path,
                                            rays_file=rays_file,
                                            valid_file=valid_file,
                                            oversample=oversample,
                                            nb_rays = nb_rays,
                                            mask_path = mask_path,
                                            log_dir=log_dir+'/'+ name,
                                            reconstruction_checkpoint=None)

max_epoch = 400

if (not run_on_cpu) and (torch.cuda.is_available()):
    trainer = L.Trainer( logger=logger,
                            accelerator="gpu",
                            max_epochs=max_epoch,
                            log_every_n_steps=1,
                            callbacks=[early_stop_callback, checkpoint_callback],
                            precision = '16-mixed',
                            accumulate_grad_batches=4)
else:
    trainer = L.Trainer( logger=logger,
                            accelerator="cpu",
                            max_epochs=max_epoch,
                            log_every_n_steps=1,
                            callbacks=[early_stop_callback, checkpoint_callback])


trainer.fit(reconstruction_module, datamodule)
