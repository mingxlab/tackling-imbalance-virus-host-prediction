
# Tackling Sequence Data Sparseness and Imbalance in Deep Learning for Virus-Host Prediction



## Install

We use linux and anaconda environment with RTX 2080Ti as the model training platform.

1. Download and install [Anaconda](https://www.anaconda.com/products/distribution#Downloads)

2. Create Conda environment

   ```
   conda create -n TaL python=3.8
   ```

3. Activate the environment

   ```
   conda activate TaL
   ```

4. Download reference packages

   ```
   pip install -r requirements.txt
   ```

   

## Use

Train your own model through our project requires the following two steps.

### Datasets Preparation

We provide a data file ``overall_database.csv`` that was used in the experiments performed in our paper. You can continue to use this file, or use your own data file, as long as it is in the same dataset format as ours. Run `python make_dataset.py`, and train, val, test files for sequence and  label will be generated in the default archive.

The `dataDeal` directory contains the algorithms used to dealing with sequence length imbalance. It will be specified and called in the model training and testing step. 

### Train model

`cd train  ` 

Run python file `python train_model.py -[Command option] `

| Command            | Function                                                     | Option                                            |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| -i, --input_path   | Enter the  archive path with dataset files, which  contains train, val, test files of sequences and  labels [required] | Your path                                         |
| -o, --output_path  | Path where the output will be saved [required]                     | Your path                                         |
| -m, --module       | Choose the sequence truncation and balancing method for model training [required]              | 1: Repeat with Gap<br /> 2. ASW<br />3. Fixed-Cut |
| -e, --epochs       | Specify the maximum number of epochs used for training the model         | Example: 100                                      |
| -a, --architecture | Select the network architecture for training                     | 1: CNN<br />2. BiLSTM+CNN                         |
| -l, --subseqlength      | Specify the subsequence length as input unit                              | Default: 250                                      |
| -s, --step      | Specify the stride of the sliding windown                              | Default: 200                                      |
| -S, --sample       | select data balance method (undersampling, oversampling)      | Example: u                                        |



### Test model

`cd train  ` 

Run python file `python test.py -[Command option] `.  It will load checkpoints of "best loss model" and "best acc model" which have been saved in the Train model step.

Commands and options are the same as Train model.
