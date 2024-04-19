# PHBS_MLF_2024_Project: Credit Risk Prediction
## Data Description:
<img width="989" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/cd215a8c-ad22-4195-8710-84ccb77df8f7">

y: ModelChoice_Default_Flag (0: no risk; 1: have risk)\\
segment: Site, Industry(3 dummy var), Age_of_Company_in_Month

## Basic Framework
### Thought: 
**Firstly use all groups' infomation, build modular models.**
<img width="680" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/63ffa85f-6dce-4354-8078-27fb2543c96f">

**Then for specific segment, use present modular models' prediction results, to build a specific model for this segment.**
<img width="830" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/a06bcc84-d5d7-49fe-8627-a84b70e09b10">

### Data Processing
1. turn 'Age_of_Company_in_Month' into 'Age_of_Company_in_Year' and divide it up into 'Age_Category'
   <img width="269" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/eb609232-f1a9-47ae-935a-c42d339df4e5">
   
2. Use Site and Industry as Segment variables, age is saved to use in modular integration for segments.
   
   <img width="151" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/41aaf10f-c9dd-438c-ac0f-83824b7ce53a">

4. Stratified random sample by ['ModelChoice_Default_Flag', 'Segment2'], get X_train, X_test, y_train, y_test

### Baseline model
1. Do one-hot encoding
2. Use all variables as input for XGBoost model, and use **optuna** to find optimal parameters
3. Do 5-fold cross validation
4. Baseline effect on testset:
   
   <img width="434" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/6a4c7564-54c6-4931-a29b-8ccb0ef1b7a5">

### Model1 : XGBoost + Logistic
1. Do RandomOverSampler because imbalance.
2. Also use XGBoost and optuna and 5-fold cv to find optimal model for each groups of modular variables, obtaining 3 prediction probability as label 1 for each rows.
   <img width="996" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d21f1005-7286-440a-9084-7701a465c676">
   
4. For each segment, Use ['Age_of_Company_in_Years', 'modular0', 'modular1', 'modular2'] as input, firstly use **knn-imputer** to fill Nan inside each segment, then build logistic model.
5. Model effect on testset:
   
   <img width="432" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f6f03e19-988e-4187-a4ad-375634a138dd">

### Model2 : Deal with outliers and do PCA inside each modular variables.
1. There exists great corelation inside modular variables, partly because of existence of outliers.

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/3dd77052-3cee-4726-b52d-77b55908d137)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d2149c4b-622d-4934-8232-cadbfa338bd8)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f3208435-ae36-45ff-9003-0c6fe440fc73)



## Autoencoder 
### Autoencoder Network Structure
Here is the structure of a typical Autoencoder (Source: MLF_Finance_Research.pdf):
<img src=".\Figures\AE结构.png"/>

- Unsupervised NN, fully connected.
- Encoder + decoder, input dimension = output dimension.
- Objective: minimize the difference between input and output.
- Bottleneck: hidden layer with fewer dimensions.

### Autoencoder for Feature Engineering and Anomaly Detection
- Dimension reduction and nonlinear PCA.
- Anomaly detection, imbalanced sample. (Non default samples and default samples)
- Utilizing overfitting. Low $X-\hat{X}$ in normal sample and higher in abnormal sample.
- Therefore, $X-\hat{X}$ can be used as features to distinguish normal and abnormal data.


### $X-\hat{X}$ in our dataset with different modules
Using autoencoder in different feature modules, we have L1Loss:

| Module             | Non default | Default data | % Diff |
|--------------------|-------------|--------------|--------|
| Financial          | 0.3142      | 0.3027       | -3.6%  |
| Internal Behaviour | 0.3212      | 0.4104       | 27.8%  |
| Bureau             | 0.1638      | 0.2125       | 29.7%  |


   

