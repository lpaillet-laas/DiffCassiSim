# DiffCassiSim: Differentiable Ray-Traced CASSI Simulator and Distortions Analysis 
Code for paper The Marginal Importance of Distortions and Alignment in CASSI systems

## Installation

To install DiffCassiSim, follow the steps below:

1. Clone the repository from Github:

```bash
git clone https://github.com/lpaillet-laas/DiffCassiSim.git
cd DiffCassiSim
```

2. Create a dedicated Python environment using Miniconda. If you don't have Miniconda installed, you can find the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

```bash
# Create a new Python environment using the environment file
conda env create -f environment.yml

# Activate the environment
conda activate diffcassisim
```

## Download datasets

3. Download the standard datasets from this [link](https://partage.laas.fr/index.php/s/geUrFeV1tI32pCr), then unzip and paste the `mst_datasets` folder in the corresponding folder at `DiffCassiSim/processing_reconstruction/datasets_reconstruction/`:
```bash
|--DiffCassiSim
    |--processing_reconstruction
    	|--datasets_reconstruction
    	    |--mst_datasets
            	|--cave_1024_28_test
                    |--scene2.mat
                                            ：  
                    |--scene191.mat
            	|--cave_1024_28_train
                    |--scene1.mat
                	        ：  
                    |--scene205.mat
                |--TSA_simu_data
                    |--scene01.mat
                    	        ：
                    |--scene10.mat                    
```

## CASSI Simulator

The main file containing all the methods and ways to modelize and use a CASSI simulated system are found under the `CASSI_class.py` file.

You can find several examples on how to use it within the `simulator_examples` folder, as well as code to regenerate the figures from the article in `simulator_examples/reproduce_figures.py`.

## Train the networks from scratch

If you wish to train the reconstruction networks :
1. Open the processing_reconstruction folder:
```bash
cd processing_reconstruction
```

2. For the main results: run the `training_reconstruction.py` script:
```bash
python training_simca_reconstruction.py
```
Note: You can choose the system and algorithm you want to use, as well as other parameters (number of rays, oversampling factor, number of epoch) in this file.

The checkpoints will be saved in the ```checkpoints``` folder.

3. For the ablation study: run the `training_simple_reconstruction.py` for the mapping ablation and either `training_wrong_reconstruction.py` or `training_wrongmis_reconstruction.py` for the rendering ablation
```bash
python training_simple_reconstruction.py
python training_wrong_reconstruction.py
python training_wrongmis_reconstruction.py
```
The checkpoints will be saved in the ```checkpoints``` folder.

## Testing the networks

To test the networks:

1. Change the value of the following variables with the path of the checkpoints you want to use:
```bash
testing_reconstruction.py > reconstruction_checkpoint
testing_simple_reconstruction.py > reconstruction_checkpoint
testing_wrong_reconstruction.py > reconstruction_checkpoint
testing_wrongmis_reconstruction.py > reconstruction_checkpoint
```
```reconstruction_checkpoint``` is the path to the checkpoint generated in the ```checkpoints``` folder.

1. Run the test scripts:
```bash
python testing_reconstruction.py
python testing_simple_reconstruction.py
python testing_wrong_reconstruction.py
python testing_wrongmis_reconstruction.py
```
The results will be saved in the ```results``` folder.
