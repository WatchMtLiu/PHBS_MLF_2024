PHBS_MLF_2024_Project: Credit Risk Prediction
# Data Description:
<img width="989" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/cd215a8c-ad22-4195-8710-84ccb77df8f7">
y: ModelChoice_Default_Flag (0: no risk; 1: have risk)
segment: Site, Industry(3 dummy var), Age_of_Company_in_Month
# Basic Framework
## Thought: 
Firstly use all groups' infomation, build modular models.
<img width="680" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/63ffa85f-6dce-4354-8078-27fb2543c96f">
Then for specific segment, use present modular models' prediction results, to build a specific model for this segment.
<img width="830" alt="image" src="https://github.com/WatchMtLiu/PHBS_MLF_2024/assets/151809533/a06bcc84-d5d7-49fe-8627-a84b70e09b10">

## Data Processing
1. turn 'Age_of_Company_in_Month' into 'Age_of_Company_in_Year' and divide it up
2. 
