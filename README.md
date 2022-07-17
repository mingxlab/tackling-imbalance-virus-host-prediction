
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

### Train model

`cd train  ` 

Run python file `python train_model.py -[Command option] `

| Command            | Function                                                     | Option                                            |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| -i, --input_path   | either the  archive path with dataset files, which  archive need contains train, val, test files for sequence and  label [required] | Your path                                         |
| -o, --output_path  | path where to save the output [required]                     | Your path                                         |
| -m, --module       | choose data deal module for training [required]              | 1: Repeat with Gap<br /> 2. ASW<br />3. Fixed-Cut |
| -e, --epochs       | maximum number of epochs used for training the model         | Example: 100                                      |
| -a, --architecture | select network architecture for training                     | 1: CNN<br />2. BiLSTM+CNN                         |
| -l, --length       | specify the subsequence length                               | Example: 250                                      |



### Test model

`cd train  ` 

Run python file `python test.py -[Command option] `, then it will load checkpoints of "best loss model" and "best acc model" which had been saved in the previous process of training the model.

Commands and options are the same as Train model.
