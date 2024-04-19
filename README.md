PHBS_MLF_2024_Project: Credit Risk Prediction
# Data Description:
<img width="989" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/cd215a8c-ad22-4195-8710-84ccb77df8f7">

y: ModelChoice_Default_Flag (0: no risk; 1: have risk)
segment: Site, Industry(3 dummy var), Age_of_Company_in_Month

# Basic Framework
## Thought: 
**Firstly use all groups' infomation, build modular models.**
<img width="680" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/63ffa85f-6dce-4354-8078-27fb2543c96f">

**Then for specific segment, use present modular models' prediction results, to build a specific model for this segment.**
<img width="830" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/a06bcc84-d5d7-49fe-8627-a84b70e09b10">

## Data Processing
1. turn 'Age_of_Company_in_Month' into 'Age_of_Company_in_Year' and divide it up into 'Age_Category'
   <img width="269" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/eb609232-f1a9-47ae-935a-c42d339df4e5">
   
2. Use Site and Industry as Segment variables, age is saved to use in modular integration for segments.
3. Stratified random sample by ['ModelChoice_Default_Flag', 'Segment2'], get X_train, X_test, y_train, y_test

## Baseline model
1. Do one-hot encoding
2. Use all variables as input for XGBoost model, and use **optuna** to find optimal parameters
3. Do 5-fold cross validation
4. Baseline effect on testset:
   <img width="434" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/6a4c7564-54c6-4931-a29b-8ccb0ef1b7a5">

## Model1 : XGBoost + Logistic
1. Do RandomOverSampler because imbalance.
2. Also use XGBoost and optuna and 5-fold cv to find optimal model for each groups of modular variables, obtaining 3 prediction probability as label 1 for each rows.
   <img width="996" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d21f1005-7286-440a-9084-7701a465c676">
   
4. For each segment, Use ['Age_of_Company_in_Years', 'modular0', 'modular1', 'modular2'] as input, firstly use **knn-imputer** to fill Nan inside each segment, then build logistic model.
5. Model effect on testset:
   <img width="432" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f6f03e19-988e-4187-a4ad-375634a138dd">

## Model2 : Deal with outliers and do PCA inside each modular variables.
