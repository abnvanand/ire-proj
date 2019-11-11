# Steps to make a prediction
```shell script
python ishyperpartisan.py
```

# Steps To Build the project

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

python lstm_multitask.py  \
    --trainingfile processedData/articles-training-bypublisher.csv \
    --glovefile processedData/glove.6B.300d.txt
```
This will train and save lstm models in current directory.  
This will also generate a tokenizer dump `tokenizer.p` file needed to token 
the test set while performing prediction.

>> Pretrained models can be downloaded [here](https://drive.google.com/open?id=1JO-7nsA8Cx_47KyN82zDeC65ymTPTWGc)

## Step4: Predict
```shell script
python predict.py \
  --inputfile processedData/articles-training-bypublisher.csv \
  --modelsdir models \
  --tokenizerfile tokenizer.p
```

