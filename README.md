# Project sponsored by IntegrityM
### Information on the project
* **Project Type:** Capstone Project 
* **Client:** [IntegrityM](https://integritym.com/)
  - Department: IntegrityM Data Analytics Team
* **Contributors:** SeungHeon Han, Yihang Zhao, Yuwen Luo
* **Email:** seung225@gwu.edu (SeungHeon Han)

### 1. Executive Summary
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
### 2. Problem Understanding
* **Background:**
  - Emerging technologies like AI and ML can help agencies like CMS to mitigate fraud, waste, and abuse, expedite the claims process, reduce errors, and lower the costs of Medicare claims management
* **Objective:**
  - Develop a predictive model prototype to identify fraudulent patterns in healthcare data
* **BusinessSuccessCriteria:**
  - The predictive model can be implemented to reduce fraud, waste, and abuse which can lead to reduced patient harm and financial costs
### 3. Methodology
#### 1) Data Analyzed
* **Database:** [Storage](https://1drv.ms/u/s!AmvqAkK8fiM5gQI7aSgR5zD90rCl?e=ePfAhe)
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
    _  |  Train  |  Valid 
  -----|---------|---------
  Ratio | 70% | 30%
   Col | 24 | 24
   Row | 112146 | 48026
 
 
 
 
 
 
 
 
 
