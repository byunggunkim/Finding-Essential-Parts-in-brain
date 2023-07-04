# Finding-Essential-Parts-in-brain
This is the code for "Finding Essential Parts of the Brain in rs-fMRI can Improve ADHD Diagnosis using Deep Learning" in Arxiv (https://arxiv.org/abs/2108.10137)


## 1. prepare 

This repository has only code related with paper. If you want to run this code, you need to prepare a few thing.


### 1.1 dependency
* Tensorflow == 2.4
* scikit-learn >= 0.24.1
* matpotlib >= 3.3.4


### 1.2 data download

This is not contain fMRI data. If you want to download the data, 
see http://preprocessed-connectomes-project.org/adhd200/download.html

You can find NIAK dataset in https://www.nitrc.org/frs/?group_id=383


### 1.3 data preprocessing

If you download the data from section 1.2, you need to preprocessing the data.
We preprocessed data using AAL 116 atlas with SPM12 


## 2. code 

* main.py : model training code with hyperparameter setting
* models.py : SCCNN type's models (e.g., SCCNN-RNN, ASCRNN, ASDRNN, ASSRNN)
* layers.py : SCCNN block and attention layer contained
* dataset.py : make dataset from preprcessed data
* result.py : summerize the results
 
