import lightning as L
from data_handler import CubesDataModule
from lightning_module import ReconstructionCASSI
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

import datetime


data_dir = "/data1/lpaillet/do_networks/datasets_reconstruction/mst_datasets/cave_1024_28_train/" # Folder where the train dataset is
data_dir_test = "/data1/lpaillet/do_networks/datasets_reconstruction/mst_datasets/TSA_simu_data/Truth/" # Folder where the test dataset is

torch.set_float32_matmul_precision('high')

datamodule = CubesDataModule(data_dir, data_dir_test, batch_size=1, crop_size = 512, num_workers=10)

system_name = "singlemis"  # Out of amici, amicimis, single, singlemis
model_name = "padut"  # Out of dgsmp, mst, dwmt, padut, duf, dauhst
name = f"testing_wrongmis_reconstruction_{system_name}_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
optics_model_file_path = f"/data1/lpaillet/do_networks/system_{system_name}.yml"

log_dir = 'tb_logs'

train = False
run_on_cpu = False # Set to True if you prefer to run it on cpu

reconstruction_checkpoint = f"/data1/lpaillet/do_networks/checkpoints/checkpoints_{model_name}/best-checkpoint_amiciwrongmis_{model_name}.ckpt"

oversample = 4
nb_rays = 20

rays_file = f"/data1/lpaillet/do_networks/rays_{system_name}.pt"
valid_file = f"/data1/lpaillet/do_networks/rays_valid_{system_name}.pt"

mask_path = "/data1/lpaillet/do_networks/mask.pt"

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
    filename=f'best-checkpoint_wrongmis_{system_name}_{model_name}',  # Checkpoint file name
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
                                            reconstruction_checkpoint=reconstruction_checkpoint)

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
                            precision = '16-mixed',
                            callbacks=[early_stop_callback, checkpoint_callback])

reconstruction_module.eval()
trainer.predict(reconstruction_module, datamodule)
