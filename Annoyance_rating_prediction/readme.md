# Run the model

```1) Unzip the Dataset.7z.001 ~ Dataset.7z.011 under the application folder```

```2) Unzip the pretrained_models.7z.001 ~ pretrained_models.7z.012 under the application folder```

```3) Enter the application folder: cd application```

## Start to infer and evaluate the model

### 1) The proposed CNN
```python 
python Proposed_CNN.py  
----------------------------------------------------------------------------------------
Loading data time: 0.613 s
Split development data to 2200 training and 245 validation data and 445 test data.
Number of 445 audios in testing
MSE :  1.1043212312528814
MAE :  0.8320875923499633
```

### 2) DNN
```python 
python DNN.py  
----------------------------------------------------------------------------------------
Loading data time: 0.395 s
Split development data to 2200 training and 245 validation data and 445 test data.
Number of 445 audios in testing
MSE :  1.7334889331245231
MAE :  1.0105259865535778
```
 
### 3) Simple_CNN
```python 
python Simple_CNN.py  
----------------------------------------------------------------------------------------
Loading data time: 0.346 s 
Split development data to 2200 training and 245 validation data and 445 test data. 
Number of 445 audios in test
MSE :  1.6749440582583413
MAE :  0.9967529329128479
```

### 4) CNN-Transformer
```python 
python CNN_Transformer.py  
----------------------------------------------------------------------------------------
Loading data time: 0.297 s
Split development data to 2200 training and 245 validation data and 445 test data. 
Number of 445 audios in test
MSE :  1.4451657348827343
MAE :  0.9664605942522541
```
 
