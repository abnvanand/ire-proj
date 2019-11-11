# Steps

## Step1 Getting the dataset
This is a SemEval19 task and you can find the raw data [here](https://zenodo.org/record/1489920). 

## Step2: Preprocessing 
The dataset is in xml format and needs some preprocessing.  
The script file `preprocess.py` is used to convert xml to csv file.
```bash
python preprocess.py -h

python preprocess.py -a data/articles-validation-bypublisher-20181122.xml \
            -g data/ground-truth-validation-bypublisher-20181122.xml 
            -o processedData/articles-validation-bypublisher.csv 
            -b True
```

>> You can skip the first 2 steps and directly get the preprocessed data from here:-
>  https://drive.google.com/open?id=1VurD3qWGiGugZfIO5lkGQCT2YG9OaDcm


>> # Make sure to put `glove.6B.300d.txt` in your processed dataset directory


## Step3: Training
```shell script
python lstm_multitask.py -h

python lstm_multitask.py -d processedData \
    --trainingfile articles-training-bypublisher.csv \  
    --testfile articles-validation-bypublisher.csv
```
This will train and save lstm models in current directory.

