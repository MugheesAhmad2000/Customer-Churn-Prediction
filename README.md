# Customer-Churn-Prediction

## Data Preprocessing:
Frist loaded the data and show the demographics of data through exploring various aspects such as data.info, data.describe() , data.dtypes etc. Then looked for the missing values and the dataset does not contain missing values. Then looked for duplicate rows and found there were 22 duplicate rows, then they were dropped. Dropping the customer ID column as it is highly cardinal. And seeing the data if there are any rows having in which tenure was 0, those rows were also dropped as if the customer have not spent time on application how he can be churned or not. 

## Exploratory Data Analysis:
Handled the Categorical Values by using LabelEncoder() and converting these values into int. Then got a detailed report of data using report profile.

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/eb0d75d1-2517-423b-a6f8-1e8b80e81e9e)
![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/0f9ae609-7449-47a9-9e93-de101b9bfc3f)
![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/fcb81036-647b-453e-9218-0dcdbb69fa3d)
![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/3eff7406-30de-4ced-911c-e8526c4f09ba)

There is detailed info of other columns too in the code and show the proportional distribution of Churn which is as follow:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/d750446e-8b26-45f0-9822-b5b13bc3d1be)

Then plotted the Box plot to see if there are any outliers.

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/4c9cd302-da14-42e4-8248-da8a004359ea)
![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/29e25823-607d-4614-ba43-f0fb57f472d3)

By seeing the box plots concluded that there were no outliers. The computed Correlation heatmap to virtualize the correlation of Churn Column with other columns. The graph is as follow:

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/2f6c711b-8989-4a43-913d-e661779aff1c)
 
Following were the insights from the heatmap:

|Variables|	Values	|Description|
|:---  | :---: |  ---:|
|Gender:|	-0.01	|negligible correlation|
|SeniorCitizen:|	0.15	|weak +ve correlation|
|Partner:|	-0.15	|weak -ve correlation|
|Dependents:|	-0.16	|weak +ve correlation|
|tenure:|	-0.35	|weak +ve correlation|
|PhoneService:|	0.01	|negligible corelation|
|MultipleLines:|	0.04	|negligible corelation|
|InternetService:|	-0.05	|negligible corelation|
|OnlineSecurity:|	-0.29	|weak -ve correlation|
|OnlineBackup:|	-0.19	|weak -ve correlation|
|DeviceProtection:|	-0.18	|weak -ve correlation|
|TechSupport:|	-0.28	|weak -ve correlation|
|StreamingTV:|	-0.03	|negligible correlation|
|StreamingMovies:|	-0.04	|negligible correlation|
|Contract:|	-0.4	|strongest -ve correlation|
|PaperlessBilling:|	0.19	|weak +ve correlation|
|PaymentMethod:| 	0.11	|weak +ve correlation|
|MonthlyCharges:| 	0.19	|weak +ve correlation|
|TotalCharges:|	0.01	|negligible corelation|

So as the results it is decided that **Contract, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Dependents, MonthlyCharges, PaperlessBilling, PaymentMethod** are the most appropriate features for modeling.

Then draw the PCA_biplot to visualize the relationships and patterns among the variables The insights from following graph were:
| PC | Variable |Value |Insights|
|:---|:---:|:---:|---:|
|PC1	|tenure	|0.433686	| best|
|PC2	|MonthlyCharges | 	0.512852 	| best|
|PC3  	|InternetService	| -0.663862	|  best|
|PC4   	|Dependents  |	0.623286 	| best|
|PC5  	|StreamingTV 	| 0.384394 	| best|
|PC6  	|TotalCharges 	|-0.526037 |	 best|
|PC7    |	gender |	-0.865638 	| best|
|PC8   |	PaymentMethod |	 0.598075	|  best|
|PC9  	|OnlineBackup 	|-0.670714 	 |best|
|PC10  |	OnlineSecurity|	 -0.442524|  	best|
|PC8 	|SeniorCitizen 	 |0.527951 |	 weak|
|PC4 	|Partner 	 |0.616343 	| weak|
|PC3   | 	PhoneService	| -0.571867  |	weak|
|PC10	|MultipleLines 	|-0.357936  	|weak|
|PC1	|DeviceProtection |	 0.287569 | 	weak|
|PC1 	|TechSupport	  |0.270524  	|weak|
|PC5 	|StreamingMovies	|  0.369578 	| weak|
|PC1  |        	Contract | 	0.394960 	|weak|
|PC9 | 	PaperlessBilling	| -0.384567 |	weak|

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/f648c608-aef2-43dd-b207-9f55112b9cc3)

## Splitting the Dataset:
Selecting the features based on the results of Correlation and PCA results for training the models. The features which I choose were:
|Choosen Features|
|:------:|
|tenure	|
|MonthlyCharges|
|OnlineBackup	|
|InternetService|
|Dependents	|
|PhoneService|
|Gender	|
|PaymentMethod|
|DeviceProtection	|
|TechSupport|
|PaperlessBilling	|
|Partner|
|SeniorCitizen	|
|Contract|
|OnlineSecurity|

And split the data into **80 20 ratios** for training and testing. Then created another dataset in which the balanced distribution of data by using sampling up technique as the number of yes in Churn were less than nos.
So, I have used SMOTE algorithm for sampling up. 

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/3c1e9ad2-8d55-4e53-9d5b-6a219aa2657f)
![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/9489574c-d9e8-4050-916a-1e48ae444cfc)
 
## Models Training:
The models which I have choices are:
*	Logistic Regression
* Random Forest
*	Support Vector Classifier

I have trained and tested these models on both unbalanced and balanced data so compare the results.

*	Logistic Regression:
* * Unbalanced Dataset:
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/1e6e6910-881c-4fec-83f6-938c0dc465e1)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.827389  |
| Precision | 0.638596 |
| Recall    | 0.566978 |
| F1 Score  | 0.60066 |

* *	Balanced Dataset:
  
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/77c476bb-f0c1-4404-aa56-ddde80c7bfb3)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.753923  |
| Precision | 0.476471 |
| Recall    | 0.757009 |
| F1 Score  | 0.584838 |

*	Random Forest
* *	Unbalanced Dataset:
    
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/04fac646-d287-4a67-bb99-aef67b6d9e96)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.792439  |
| Precision | 0.55102 |
| Recall    | 0.504673 |
| F1 Score  | 0.526829 |

* *	Balanced Dataset:
    
The results of testing were:

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/ad77168d-8298-4949-b21d-77bb6cdf43fd)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.776034  |
| Precision | 0.509044 |
| Recall    | 0.613707 |
| F1 Score  | 0.556497 |

*	Support Vector Classifier
* *	Unbalanced Dataset:
    
The results of testing were:

![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/1990bc7b-eb80-407b-8d00-7b2140f7bbd0)

 | Metric    |    Score |
|:---|----:|
| Accuracy  | 0.821683  |
| Precision | 0.678392 |
| Recall    | 0.420561 |
| F1 Score  | 0.519231 |

* *	Balanced Dataset:
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/8e917c30-90e2-427b-ab10-0193e94f8c52)

 | Metric    |    Score |
|:---|----:|
| Accuracy  | 0.731098  |
| Precision | 0.44964 |
| Recall    | 0.778816 |
| F1 Score  | 0.570125 |

After comparing the results, the model which performed the best was Logistic Regression with unbalanced dataset with an accuracy of 82.73%. After than was Logistic Regression with balanced dataset with an accuracy of 82.16%. Which means that logistic performed god in both conditions with balance and unbalance dataset.

## Optimization:
For optimization I used GridSearchCV to get the set of parameters with the best performance for that specific model and specific dataset.
*	Logistic Regression for unbalanced dataset:
  
 Best Parameters: {'C': 0.5, 'penalty': 'l2', 'solver': 'liblinear'}
*	Logistic Regression for balanced dataset:
  
Best Parameters: {'C': 0.5, 'penalty': 'l2', 'solver': 'liblinear'}
*	Random forest for unbalanced dataset:
  
Best Parameters: {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 500}
*	Random forest for balanced dataset:
  
Best Parameters: {'max_depth': 20, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
*	SVC for unbalanced dataset:
  
Best Parameters: {'C': 10.0, 'gamma': 'scale', 'kernel': 'linear'}
*	SVC for balanced dataset:
  
Best Parameters: {'C': 10.0, 'gamma': 'auto', 'kernel': 'rbf'}


## Models Training after Optimization:
*	Logistic Regression:
* *	Unbalanced Dataset:
    
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/98e94bbb-e253-41f8-b195-e780e3dafbf0)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.82525  |
| Precision | 0.631034 |
| Recall    | 0.570093 |
| F1 Score  | 0.599018 |

* *	Balanced Dataset:
  
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/eefd8134-6a3e-47c8-ac6d-f6230bb42107)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.75321  |
| Precision | 0.475538|
| Recall    | 0.757009 |
| F1 Score  | 0.584135 |

*	Random Forest
* *	Unbalanced Dataset:
  
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/909a8675-0cd6-4afe-a78e-c79431084637)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.821683  |
| Precision | 0.629091|
| Recall    | 0.538941 |
| F1 Score  | 0.580537 |

* *	Balanced Dataset:
    
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/670517b6-eaf3-4f12-afcd-5c41c9d69516)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.773894  |
| Precision | 0.505181|
| Recall    | 0.607477 |
| F1 Score  | 0.551627 |

*	Support Vector Classifier
* *	Unbalanced Dataset:
    
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/b8e6014e-1dae-4756-bda1-99cb194cdaad)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.823823  |
| Precision | 0.628472|
| Recall    | 0.563863 |
| F1 Score  | 0.594417|

* *	Balanced Dataset:
    
The results of testing were:

 ![image](https://github.com/MugheesAhmad2000/Customer-Churn-Prediction/assets/61706830/0d8d1014-3f97-42c5-bca8-b32edd8b6a78)

| Metric    |    Score |
|:---|----:|
| Accuracy  | 0.740371  |
| Precision | 0.460838|
| Recall    | 0.788162 |
| F1 Score  | 0.581609|

After comparing the results, the model which performed the best was Logistic Regression with unbalanced dataset with an accuracy of 82.53%. But SVC has also increased its accuracy to 82.3%.
