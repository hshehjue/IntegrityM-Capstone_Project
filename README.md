# Project sponsored by IntegrityM
## Information 
* **Project Type:** Capstone Project 
* **Client:** [IntegrityM](https://integritym.com/)
  - Department: IntegrityM Data Analytics Team
* **Contributors:** SeungHeon Han, Yihang Zhao, Yuwen Luo
* **Email:** seung225@gwu.edu (SeungHeon Han)

## 1. Executive Summary
* **Project Goal:**
  - Develop classification machine learning models which can effectively identify healthcare frauds, waste, and abuse of the information on patients
* **Success Criteria:**
  - The predictive model can be replicated in other places without losing much accuracy
  - The model can predict fraudulent patterns with high F1, AUC, ROC scores
* **ProjectPlan:**
  - Understand the business problem and goal with clients
  - Collect datasets, review documents
  - Wrangle with datasets and clean datasets
  - Label fraud patterns within the data frame
  - Data balancing (undersampling)
  - Data partitioning (train, test)
  - Data partitioning (train, validation)
  - Apply machine learning algorithm to the data
  - Measure model performance on a test dataset
  - Visualize the results
  
## 2. Problem Understanding
* **Background:**
  - Emerging technologies like AI and ML can help agencies like CMS to mitigate fraud, waste, and abuse, expedite the claims process, reduce errors, and lower the costs of Medicare claims management
* **Objective:**
  - Develop a predictive model prototype to identify fraudulent patterns in healthcare data
* **BusinessSuccessCriteria:**
  - The predictive model can be implemented to reduce fraud, waste, and abuse which can lead to reduced patient harm and financial costs
  
## 3. Methodology
### 1) Data Analyzed
* **Database** [Storage](https://1drv.ms/u/s!AmvqAkK8fiM5gQI7aSgR5zD90rCl?e=ePfAhe)
  - PartB [Body Data]
    - Medicare Physician & Other Practitioners - by Provider and Service
    - 2013 - 2018
    - Source: [CMS website](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service)
    - Size: 67,764,122 * 29
  - LEIE Databases [Labeling]
    - 08-2021 Updated LEIE Database
      - Source: [Office of Inspector General Website](https://oig.hhs.gov/exclusions/exclusions_list.asp)
      - Size: 74,584 * 18
    - 2020-2021 LEIE with Reinstatements
      - Source: Office of Inspector General Website
      - Size: 484*21
    - LEIE Plus
      - Source: IntegrityM
      - Size: 937*18
* **Used Features (PartB)**
    **Features Used** |  **Data Dictionary**
  -----------------|------------------
  Rndrng_NP | National Provider Identifier
  Rndrng_Prvdr_Type | Type of the Provider
  HCPCS_Cd | Healthcare Common Procedure Coding System (HCPCS) code
  Place_Of_Srvc | either a facility (F) or non-facility (O)
  Tot_Benes | Number of Medicare Part B fee-for-service beneficiaries utilizing the drug
  Tot_Srvc | Number of services provided
  Tot_Bene_Day_Srvcs | Number of Distinct Medicare Beneficiary/Per Day Services
  Avg_Sbmtd_Chrg | Average Submitted Charge Amount
  Avg_Mdcr_Pymt_Amt | Average Medicare Payment Amount
  Avg_Mdcr_Alowd_Amt | Average Medicare Allowed Amount
  Avg_Mdcr_Stdzd_Amt | Average Medicare Standardized Payment Amount

* **Data Preprocessing**
  - Data Exploration:
    - No null/invalid values across every used feature
    - Outlier: |z-score| > 3
    - Multicollinearity:
      - Tot_Bene_Day_Srvcs & Tot_Benes (r = 0.97)
      - Avg_Mdcr_Alowd_Amt & Avg_Mdcr_Stdzd_Amt - Average (r = 0.99)
      - Avg_Mdcr_Pymt_Amt & Avg_Mdcr_Stdzd_Amt (r = 0.99)
      - Avg_Mdcr_Pymt_Amt & Avg_Mdcr_Alowd_Amt (r = 1)
  - Data Cleaning & Labeling:
    - Filtering out the rows that contain HCPCS code related to prescription
      - Drug Average Sales Pricing File [DASP](https://1drv.ms/u/s!AmvqAkK8fiM5gQU15OEmrx3nNFIj?e=weDLfe)
      - Source: [CMS.gov](https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Part-B-Drugs/McrPartBDrugAvgSalesPrice/2019ASPFiles)
      - Removing the instances that have HCPCS_cd matching to the DASP file
      - 2,074,502 rows were removed from PartB
    - Sorting out the NPIs in the LEIE files matching to the fraud-related exclusion types referencing the OIG Acts
      - Total number of LEIE data: 76,005
      - Number of invalid unique NPI: 69,465
      - Number of valid unique NPI: 6,540
      - Number of valid unique NPI matching to frauds: 5,489
      - **Fraud-related exclusion types from OIG acts**
        - 1128a1: Conviction of program-related crimes
        - 1128a2: Conviction relating to patient abuse
        - 1128a3: Felony conviction relating to health care fraud
        - 1128a4: Felony conviction relating to controlled substance
        - 1128b1: Conviction relating to fraud
        - 1128b2:  Conviction relating to obstruction of an investigation or audit
        - 1128b6: Claims for excessive charges or unnecessary services and failure of certain organizations to furnish medically necessary services
        - 1128b7: Fraud, kickbacks, and other prohibited activities
        - 1128b4: License revocation or suspension
        - 1128b5: Exclusion or suspension under federal or state health care program
        - 1128b8: Entities controlled by a sanctioned individual
        - 1128b15: Individuals Controlling a Sanctioned Entity
        - 1128b16: Making false statements or misrepresentation of material facts
    - Labeling each instance of PartB (Fraud = 1/ Non-fraud = 0)
      - Fraud: 33,638
      - Non-fraud: 65,655,982
    - Undersampling the majority group
      - Decrease the number of examples in the Non-fraud group to 10 times as many examples as the minority group
      - Undersampled PartB:
        - Fraud: 33,638 (9.09%)
        - Non-fraud: 336,380 (90.91%)
    - One-Hot Encoding
      - Removed HCPCS_cd variable (due to its more than 3000 dummies)
      - Dummy variables for:
        - Rndrng_Prvdr_Type
        - Place_Of_Srvc
      - Number of features: 11 ⇒ 132
    - Data Splitting
      - Train: 80% / Test: 20%
      - Stratified random sampling
      
### 2) Analytics Techniques 
* **Validation Split:** 
  - Splitting the whole dataset into train and test sets with 7:3 ratio
* **Modeling Technique:**
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Explainable Boosting Machine (EBM) 
  - **eXtreme Gradient Boosting (XGBoost) - Final Recommendation**
* **Modeling assumptions for a logistic regression model:**
  - The random errors have a constant standard deviation 
  - The random errors follow a normal distribution
  - The data are randomly sampled from the process

## 4. Results, Conclusion, and/or Recommendations
### 1) LogisticRegression
* **Below table include all continuous variable and their coefficients with significant p-values**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/Logistic_coeff.png width=50% height=50%>

* **AUC chart of the logistic regression model**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/Logistic_ROC.png width=50% height=50%>

### 2) Decision Tree 
* **Confusion matrix**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/DT_confusion.png width=50% height=50%>

* **Visualization of decision tree after pruning**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/DT_tree.png width=50% height=50%>

### 3) RandomForest
* **Features selection on Train Set using Elastic Net Logistic GLM**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/RF_importance.png width=50% height=50%>

* **Modeled Random Forest using selected features and list test set performance**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/RF_performance.png width=50% height=50%>

### 4) ExplainableBoostingMachine(EBM)
* **Hyperparameters**
  - Random Grid Search
  - Best Hyperparameters:
    <img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/EBM_param.png width=50% height=50%>

* **Running Time:** 2531.81 sec

* **Average of the five fold evaluation:**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/EBM_performance.png width=50% height=50%>

* **Confusion Matrix with the best cut-off**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/EBM_confusion.png width=50% height=50%>

### 5) eXtremeGradientBoosting(XGBoost)
* **Hyperparameters**
  - Random Grid Search
  - Best Hyperparameters:
    <img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/XGB_param.png width=50% height=50%>
    
* **Running Time:** 340 sec

* **Average of the five fold evaluation:**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/XGB_performance.png width=50% height=50%>

* **Confusion Matrix with the best cutoff**
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/XGB_confusion.png width=50% height=50%>


### 6) Evaluation on Test Set
<img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/Evaluation.png width=50% height=50%>

* **Our recommending classification model:** *XGBoost model*

### 7) Post-Processing
* **Global Feature Importance based on Shapley Values**
  - Features with the shapley values above 90th percentiles:
    <img src=https://github.com/hshehjue/IntegrityM-Capstone_Project/blob/main/Images/XGB_importance.png width=50% height=50%>
    - 1. Tot_Srvcs
    - 2. Type_Internal Medicine
    - 3. Type_Family Practice
    - 4. Avg_Mdcr_Alowd_Amt
    - 5. Avg_Mdcr_Pymt_Amt
    - 6. Type_Diagnostic Radiology
    - 7. Tot_Bene_Day_Srvcs
    - 8. Tot_Benes
    - 9. Avg_Sbmtd_Charg
    - 10. Avg_Mdcr_Stdzd_Amt
    - 11. Type_Cardiology
    - 12. Type_Nurse Practitioner
    - 13. Type_Proiatry

* **Potential Disparate Impact on Gender**
  - Protected Group:  Female Providers
  - Referenced Group:  Male Providers




## 5. Potential Next Steps for future
* Apply string matching method (like Fuzzywuzzy package in Python) to receive more fraud labels from LEIE Plus.
* Check if the predicting fraud labels from XGBoost match the fraud labels in reality. This could help us to examine the predictability of the XGBoost model.









 
 
 
