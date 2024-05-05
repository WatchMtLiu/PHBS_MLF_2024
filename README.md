Credit Risk Prediction

## Team Member:
| Name               | Student ID      | 
|--------------------|-----------------|
| Zerun Zhu          | 2201212450      | 
| Fanyuan Ma         | 2301212364      | 


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
   
2. Use Site and Industry as Segment variables, **age of Company** is saved to use in modular integration for segments.
   
   <img width="151" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/41aaf10f-c9dd-438c-ac0f-83824b7ce53a">

4. **Stratified random sample** by ['ModelChoice_Default_Flag', 'Segment2'], get X_train, X_test, y_train, y_test

### Baseline model
1. Do one-hot encoding.
2. Use all variables as input for XGBoost model, and use **optuna** to find optimal parameters.
3. Do 5-fold cross validation.
4. Baseline effect on testset: Test ROC AUC = 0.7744
   
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/b2d3b83b-4fe3-421c-9437-9203cd8f2686)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/e5c82424-0300-4cc2-8d6a-34401d36697d)


### Model1 : XGBoost + Logistic
1. Do **RandomOverSample** because of imbalance problem.
2. Also use XGBoost and optuna and 5-fold cv to find optimal model for each groups of modular variables, obtaining 3 prediction probability as label 1 for each rows. Inside this XGBoost, we set **imbalance-weight**.
   <img width="996" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d21f1005-7286-440a-9084-7701a465c676">
   
4. For each segment, Use ['Age_of_Company_in_Years', 'modular0', 'modular1', 'modular2'] as input, firstly use **knn-imputer** to fill Nan inside each segment, then build **logistic** model. Pay attention to record knn-imputer models to do same dealing with testset later.
5. Model effect on testset: Test ROC AUC = 0.5246

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/4cde544c-783a-46eb-b476-c02346b7bb30)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/fec6b6a1-2e79-4540-a1d4-693d45547a07)
   


### Model2 : Deal with outliers and do PCA inside each modular variables.
1. There exists great **corelation** inside modular variables, partly because of existence of **outliers**.

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/3dd77052-3cee-4726-b52d-77b55908d137)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/d2149c4b-622d-4934-8232-cadbfa338bd8)
   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/f3208435-ae36-45ff-9003-0c6fe440fc73)

2. So we deal with outliers for trainset firstly, **replace outliers with boundary quantile values**, and record these values to do same dealing with testset later.
3. Then for each group of moduler variables, we do **min-max scaling** firstly, so that knn-imputer can treat every variable fairly. Then use knn-imputer to fill missing values.
4. Inside each group of moduler variables, we do **pca** and reserve some important pcas. Before this, we need to do normalization scaling.
   pca_components = {'financial_variables': 10, 'internal_behavior_variables': 15, 'bureau_variables': 20}

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/288f701d-a76f-4c83-9835-27f0253886af)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/e3363bd4-ad99-453e-ad44-748cb52f5228)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/aa9a96ad-1d38-44b0-98fe-78b2b8b73479)



6. do other thing same as model2. Build XGboost model for each group of module variables, and then use age information and 3 module models' prediction probability as input, build a logistic model to get final output.
7. Model effect on testset: Test ROC AUC = 0.5246

  ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/5e9916ab-658a-476b-9da9-ed455c73d473)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/886c7ff8-0e48-4308-a31f-4431d397065a)

   
### Model3: Do dimensionality reduction use autoencoder and build XGboost model.
1.  Replace outliers with boundary quantile values.
2.  Do min-max scaling and impute Nan with knn-imputer.
3.  Train a **autoencoder** model to compress data into 10-dim and then expand them to original dimensions again. Here we just use those positive training samples to build model, adn get trainning loss on all training samples. Training loss = 0.0931. Test Loss = 0.0942
4.  Just use encoder part to reduce data dimension into 10. Just use these 10-dim training data to train a XGboost model using all 10 features.
5.  Model effect on testset: Test ROC AUC = 0.50

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/b3626bd5-0cc4-4cb5-b8be-7cd262df62af)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/4edc60fd-b7e8-420f-8866-ae2400f849e1)


### Model4: Replace XGboost in model 3 with Logistic model.
I think for a 10-dim data, XGBoost is too much, so I also48se logistic model here using those 10-dim model after dimension reduction.
   Model effect on testset: Test ROC AUC = 0.5

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/c2e09ac2-2989-4472-8a63-a80816d75228)

   ![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/54ff01cf-335c-4c2f-a558-8d219e353a8e)


## Summary
I think maybe autoencoder is a strong tool. I also found related paper in credit risk prediction using it. But here its effect is not so good. The primary problem should be the dimension reducing to 10 dim. It's too low. It lost too much information. I think maybe 30-50 is more appropriate. And still, baseline XGboost model is strong and best. 



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
![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/c7d34b06-9117-4d6e-9cd3-e28d4e3b0eb9)
![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/8afde254-9a18-4224-8f9a-0cc758cf484e)
![image](https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/92200287-7481-472c-b483-5891f2f4dd8a)


### Possible Alternatives
- Generator and discriminator. 
- Generator generates fake sample from random noise.
- Discriminator classify the real sample and the generated sample.
- A strong discriminator can be used as a classifier of normal and abnormal sample.
