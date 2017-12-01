#### Nucleusome positioning using convolutional neural networks

LeNup is a tool to train a convolutional neural network in order to predict nucleosome positioning.



---------------------------------------------------------------------------------------------------
### Dependency


LeNup runs in Ubuntu in our lab, a linux like system. 
LeNup is based on Torch7. Therefore, Torch7 (http://torch.ch/) is needed for this package. LeNup Users also need to 

install orther suport packages or tools in their computer, including Python, numpy, h5py, pandas, nn, optim, cudnn, 

cutorch.

If you want to train the model by GPU, CUDA and cudnn in also needed. CUDA and cudnn should be ready before you 

prepare to install Torch7.

---------------------------------------------------------------------------------------------------
### Creating hdf5 format dataset

Suppose you positive sample file, e.g. positive.txe and negative sample file, e.g. negative.txt with the plain text 

format, you can concat them to one sequence file, e.g. seq_out.txt, and generate the label file, e.g.label.txt by 

running command as follows,
'''
python /path/precreatedataset.py /path/positive.txt /path/negative.txt /path/seq_out.txt /path/label.txt

'''
"path" in the command in the context is the absolute path of the files.

After that, users need to transform the sequence file, e.g. seq_out.txt to hdf5 format, e.g.dataset_out.h5 by 

running the command as follows, 

'''
python /path/createdataset.py /path/seq_out.txt /path/label.txt /path/dataset_out.h5
'''
The default cross-validation is 20-fold. Users may change it in createdataset.py.

---------------------------------------------------------------------------------------------------
### Training LeNup

With the hdf5 file, you can use train.lua to train their model by the command:
'''
th /path/train.lua [option...] /path/dataset_out.h5
'''

There are also many options you can choose.
'''
-cuda
'''
Training by GPU is much faster than by CPU. If you decide to train the model by GPU, you should choose -cuda 

option.

'''
-cudnn
'''
If you decide to train the model by GPU and use cudnn, you should choose -cudnn option.

'''
-max_epochs num
'''
This option will set a max_epochs in training. When the maximum epoch is reached, the training process will stop.

'''
-stagnant_t num
'''
This option will give a limit in training that if the result does not promote, the training will stop.

'''
-job [path for hyperparameters]
'''
The hyperparameters such as learning rate, momentum  will be set through -job. The recommended hyperparameters are 

given in params.txt, and you can change the value of these hyperparameters in params.txt.

'''
-save [the path for the best model output]
'''
The best model on the test dataset will be output to the pointed path. The default output path is "/home/user"

A command instance looks like,
'''
th /path/train.lua -cuda -max_epochs xxx -stagnant_t yyy -job /path/params.txt /path/dataset.h5
'''

---------------------------------------------------------------------------------------------------
### Creating predictive dataset

In order to creating predictive dataset, you should, firstly, prepare a sequence file. Sequence file should include 

DNA sequence data.
You should use the following two commands to create the dataset to be classified.

'''
python precreatetestdataset.py /path/sequsece.txt  /path/seq_out.txt
'''

'''
python /path/createtestdataset.py /path/seq_out.txt /path/dataset_out.h5
'''


---------------------------------------------------------------------------------------------------
### LeNup Prediction

Once LeNup training finishes, the model file, model_file.t7, is output. Users can use this file to predict 

nucleosome positioning of DNA fragments with length 147 bp including in an input file, e.g. data_file.h5. Assuming 

the sequence of DNA fragments in the input file has already tranformed to hdf5 format. out_file.txt saves the 

prediction results.

Therefore, the computer command for the prediction is as follows, 
'''
th /path/predicting.lua [option...] /path/model_file.t7 /path/data_file.h5 /path/out_file.txt
'''
where, model_file.t7 is a LeNup model file; data_file.h5 is the input file, including sequences, which will be 

classified.

There are also many options you can choose.
'''
-batch num
'''
num is a number of sequences in one batch.

'''
-cuda
'''
Predicting by GPU is much faster than by CPU.

'''
-cudnn
'''
cudnn is an effective option for the computational acceleration of a neural network. 

For instance, the predicting/test command for LeNup is as follows,

'''
th /path/predicting.lua -batch 64 -cuda -cudnn /path/model_file.t7 /path/data_file.h5 /path/out_file.txt
'''

The output is the classification probability produced by the sigmoid function for each DNA sequence with length 147 

bp. User may choose 0.5 as the boundary of positive and negative classification, and execute the following command 

to get the final prediction.

'''
python /path/finalcalssresult.py /path/out_file.txt /path/finalout_file.txt
''' 

---------------------------------------------------------------------------------------------------
### Acknowledgement
A part of Python's source code transforming DNA sequence to one-hot data is from https://github.com/davek44/Basset.
