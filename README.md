# iclr2023-meta-learning-eeg-mi-classification
Official implementation of 'Meta-Learning for Subject Adaptation in Low-Data Environments for EEG-Based Motor Imagery Brain-Computer Interfaces' (ICLR 2023). Our meta-learning approach is compared to transfer learning in terms of accuracy after few-shot fine-tuning on new subjects for EEG-based motor imagery classification.

## Dataset
The BCI Competition IV 2a dataset is used for the experiments. It is retrieved by the Mother of all BCI Benchmark (MOABB).

Download_Dataset.ipynb contains the code for downloading the BCI Competition IV 2a dataset and saving the trials in separate numpy files for easy access.
The ```bciiv2a_train``` folder contains the files for each trial, named s```{subject}```_c```{class}```_t```{trial}```.npy, where ```{subject}``` is from 1 to 9, ```{class}``` is the class index (0 to 3) corresponding to each of the motor imagery classes, and ```{trial}``` is the index of the trial (0 to 143).

## Experiments
EEGNet_subj3.py contains the code for few shot performance using subject 3 as the test subject, and saves the output in Output_subj3.txt, which is included in Output Files/Few Shot.
The experiment can be repeated for the other subjects by changing the subject number in the lines 15 and 487.