#### LeNup: Learning Nucleosome positioning from DNA sequences with improved convolutional neural networks

LeNup is a tool to train a convolutional neural network in order to predict nucleosome positioning.



---------------------------------------------------------------------------------------------------
### Dependency


LeNup runs in Ubuntu in our lab, a linux like system. LeNup is based on Torch7. Therefore, Torch7 (http://torch.ch/) is supposed to be preinstalled for running this package. 

LeNup Users also need to install some support packages or tools in their computer, including Python, numpy, h5py, pandas, nn, optim, cudnn and cutorch.

If you want to train LeNup by GPU, CUDA and cudnn must be installed. CUDA and cudnn should be ready before you prepare to install Torch7.

---------------------------------------------------------------------------------------------------
### Creating hdf5 format dataset

Suppose that you have a positive sample file, e.g. positive.txt and a negative sample file, e.g. negative.txt with the plain text format, you can concat them to one sequence file, e.g. seq_out.txt, and generate the label file, e.g.label.txt by running command as follows,
```
python /path/PrecreateDataset.py /path/positive.txt /path/negative.txt /path/seq_out.txt /path/label.txt
```
"path" in the command in the context is the absolute path of the files. 

After that, users need to transform the sequence file, e.g. seq_out.txt to hdf5 format, e.g. dataset_out.h5 by running the command as follows,
```
python /path/CreateDataset.py /path/seq_out.txt /path/label.txt /path/dataset_out.h5
```
The default cross-validation is 20-fold. Users may change it in createdataset.py.

---------------------------------------------------------------------------------------------------
### Training LeNup

With the hdf5 file, you can use train.lua to train their model by the command:
```
th /path/Train.lua [option...] /path/dataset_out.h5
```

There are also many options you can choose.
```
-cuda
```
Training by GPU is much faster than by CPU. If you decide to train the model by GPU, you should choose -cuda option.

```
-cudnn
```
If you decide to train the model by GPU and use cudnn, you should choose -cudnn option.

```
-max_epochs num
```
This option set the maximum of the epoch to num. When the maximum is reached, the training process will stop.

```
-stagnant_t num
```
When we train a model, the accuracy may not be improved in many epoches. Here users set how many epoches which they can tolerate in the training. Over this number, the calculation will stop.

```
-job /path/params.txt
```
The hyperparameters, such as learning rate and momentum, will be set through -job. The recommended hyperparameters are given in params.txt, and you can change the value of these hyperparameters in this file.

```
-save /path/ABC(any string selected by users)
```
The best model on the test dataset will be output to the pointed path. The default output path is "/home/user directory". The best model name is ABC_best.th.

A command instance looks like,
```
th /path/Train.lua -cuda -max_epochs num1 -stagnant_t num2 -save ABC -job /path/params.txt /path/dataset_out.h5
```

---------------------------------------------------------------------------------------------------
### Creating hdf5 Format Prediction Dataset

In order to create test or prediction datasets, We, firstly, prepare a sequence file with the plain text format. There is one DNA sequence with a length of 147bp in one raw in the sequence file. After that, We transform the sequence file from plain text format to hdf5 format. The commands for creating the dataset as follows:

```
python /path/CreatePredictionDataset.py /path/sequence.txt  /path/dataset_out.h5
```



---------------------------------------------------------------------------------------------------
### LeNup Prediction  


Once LeNup training finishes, the model file, for instance, ABC_best.th, is the output. Users can use this file to predict nucleosome positioning of DNA fragments with 147bp in length included in an input file, e.g. dataset_out.h5. We use out_file.txt to save the prediction results. Therefore, the computer command for the prediction is as follows,
```
th /path/Predicting.lua [option...] /path/ABC_best.th /path/dataset_out.h5 /path/out_file.txt
```
Where, ABC_best.th is the LeNup model file; dataset_out.h5 is the input file including sequences to be classified.

There are also many options you can choose.
```
-batch num
```
num is a number of sequences in one batch.

```
-cuda
```
Predicting by GPU is much faster than by CPU.

```
-cudnn
```
Cudnn is an effective option for the computational acceleration of a neural network. 


For instance, the predicting/test command for LeNup is as follows,
```
th /path/Predicting.lua -batch 64 -cuda -cudnn /path/ABC_best.th /path/dataset_out.h5  /path/out_file.txt
```

The output is the classification probability produced by the sigmoid function for each DNA sequence with a length of 147bp. Users may choose 0.5 as the threshold for the positive and negative classification, and execute the following command to get the final prediction.

```
python /path/Finalcalssresult.py /path/out_file.txt /path/finalout_file.txt
```

---------------------------------------------------------------------------------------------------
### Acknowledgement
A part of Python's source code transforming DNA sequence to one-hot data is from https://github.com/davek44/Basset.
