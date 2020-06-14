# <center><ins>Telco Customer Churn Analysis: EDA & Classification Models<ins/><center/>
### <center>Andrew Cole<center/>


Please refer to the links below for a detailed blog post walking through the analysis:
* **Exploratory Data Analysis**: https://medium.com/@andrewcole.817/customer-churn-analysis-eda-a688c8a166ed
* **Building the Logistic Regression Model:** https://towardsdatascience.com/predicting-customer-churn-using-logistic-regression-c6076f37eaca


### <ins>Project Overview<ins/>
In the commercial world, customers are king. Understanding the customer is of the utmost importance and understanding their behavior patterns can lead to very impactful business decisions. **Customer Churn** is the rate at which a commercial customer leaves the commercial business/platform and takes their money elsewhere, and understanding the underlying customer patterns will greatly impact a business' ability to retain their customers. As a data researcher trying to break into the professional world, I thought it would be pertinent to get a better understanding of what these churn data features may look like and how they can be used to understand the customer.

In this repository I will utilize a telecommunication company's (Telco) customer dataset to perform a very detailed Exploratory Data Analysis to develop a strong understanding of any patterns or trends existing in our data. Secondly, I will process the data and build a series of binary outcome classification models that will try to effectively predict whether a customer will or will not churn from the telecommunications platform.

### <ins>The Data<ins/>
The data is sourced from Kaggle (https://www.kaggle.com/blastchar/telco-customer-churn). Our dataset contains 7043 entries representing 7043 unique customers. There are 21 columns, with 19 features (target feature = 'Churn'). The features are numeric and categorical in nature, so we will need to address these differences before modeling.

### <ins/>Included in this Repository<ins/>
* EDA.ipynb : Commented Walkthrough of the Exploratory Data Analysis process and visualizations
* decision_tree.ipynb: Decision Tree Classification Model
* KNN.ipynb: K-Nearest Neighbors Classification Model 
* decision_tree.ipynb: Decision Tree Classification Model    
* rf_bagging.ipynb: Random Forrest and BaggingClassifier Classification Models

* regression_module.py: Module with functions for execution of Logistic regression
* eda_module.py: Module with functions for execution of EDA process
    
* Telco Customer Churn Analysis.pdf: Google slides presentation (as if an insight presentation was required per deliverables)
    
* data [folder]: Folder containing all used data
* pics [folder]: Folder containing figures & screenshots for medium blog posts
    

