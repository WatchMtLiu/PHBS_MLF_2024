PHBS_MLF_2024_Project: Credit Risk Prediction
## Data Description:
<img width="989" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/cd215a8c-ad22-4195-8710-84ccb77df8f7">

1. y: ModelChoice_Default_Flag (0: no risk; 1: have risk)
2. segment: Site, Industry(3 dummy var), Age_of_Company_in_Month
3. X: 3 modulars(financial_variables, internal_behavior_variables, bureau_variables)

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
4. Baseline effect on testset: Test ROC AUC = 0.7512
   
   <img width="432" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/9941842d-bf33-408a-8745-5a088615feda">

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/382c5eea-9e9f-4691-bf7e-a93430051635)

### Model1 : XGBoost + Logistic
1. Do RandomOverSampler because imbalance.
2. Also use XGBoost and optuna and 5-fold cv to find optimal model for each groups of modular variables, obtaining 3 prediction probability as label 1 for each rows. Insider this XGBoost, we set imbalance-weight.
   <img width="996" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d21f1005-7286-440a-9084-7701a465c676">
   
4. For each segment, Use ['Age_of_Company_in_Years', 'modular0', 'modular1', 'modular2'] as input, firstly use **knn-imputer** to fill Nan inside each segment, then build logistic model.
5. Model effect on testset:
   
   <img width="432" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f6f03e19-988e-4187-a4ad-375634a138dd">

### Model2 : Deal with outliers and do PCA inside each modular variables.
1. There exists great corelation inside modular variables, partly because of existence of outliers.

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/3dd77052-3cee-4726-b52d-77b55908d137)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d2149c4b-622d-4934-8232-cadbfa338bd8)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f3208435-ae36-45ff-9003-0c6fe440fc73)

2. So we deal with outliers for trainset firstly, replace outliers with boundary values
3. Then for each group of moduler variables, we do min-max scaling and use knn-imputer to fill missing values.
4. Inside each group of moduler variables, we do pca and reserve some important pcas.
   pca_components = {'financial_variables': 10, 'internal_behavior_variables': 15, 'bureau_variables': 25}

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/6dc557ea-405e-4d25-8616-6be3ce17b0ea)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f4df7ac2-8f93-499a-ba3c-49b9fdf7638d)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/949e83f3-2f99-4f84-8a6a-76aad3d72361)

5. do other thing same as model2.
6. Model effect: still running.
   

## Autoencoder 
 ### Autoencoder Network Structure
Here is the structure of a typical Autoencoder (Source: MLF_Finance_Research.pdf):

<img width="667" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/dce3dad2-4d49-411b-bf98-798fcef15878">

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

The correlation of $X-\hat{X}$ is as below:

![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/1f70d840-e69c-4b62-a55f-06c8f64cd3d6)
![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/eed51366-0a37-4aeb-9a26-0d83bd7ce340)
![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d50cb5ce-8c6c-4463-923d-0ad35278c350)


### Possible Alternatives

Another possible model for anomaly detection is GAN (Generative Adversarial Network).

<img width="1211" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d03afdc1-7a4c-4b53-9179-f362a0d1e6f6">
<img width="644" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/8a4e11e9-0899-4a8b-b58d-4b155b7dac95">


- Generator and discriminator. 
- Generator generates fake sample from random noise.
- Discriminator classify the real sample and the generated sample.
- A strong discriminator can be used as a classifier of normal and abnormal sample.


## To-do
1. Under-smapling + Over-sampling
2. Use method based on objective function
3. Maybe use other model which can deal with imbalance problem better.
4. Maybe bagging and replacement sampling with label 1(risk) samples each time.
