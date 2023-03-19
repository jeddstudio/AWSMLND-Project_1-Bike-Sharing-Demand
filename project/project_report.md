# Report: Predict Bike Sharing Demand with AutoGluon Solution
---
#### Jedd H. C. Wong



## Initial Training
---
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
- Kaggle does not accept any negative values in the submission CSV file, I have to turn all the negative values to 0 before submitting.  
- The reason is that the model predicts how many people will use the bike, so there are no negative values in the count.
- So before committing all three times in Project, I have to do the following three steps.  
	- Use `.describe()` to check if the "min" value is less than 0. If there is a value less than 0 in this min, then there is at least one piece of data in the predictor that is less than 0.  
		- If "min" < 0, do the next step  
		- If "min" >= 0, submit directly.  
	- Find all less than 0 value.  
		- Using `negative_check = predictions.lt(0)`  
		- Total number of all negative value data  
			- `num_of_negative = negative_check.sum()`  
				- This will let us know how much data the predictor has that is negative  
- Set all negative values to zero  
	- `predictions[predictions < 0] = 0`  



### What was the top ranked model that performed?
The top-ranked model is "WeightedEnsemble_L3"



## Exploratory data analysis and feature creation
---
### What did the exploratory analysis find and how did you add additional features?
##### Convert `[season]` and `[weather]` Dtype
- In Train dataset `[season]` and `[weather]` are 0 and 1, which should be "category", but in Dtype it is "int64", so I need to convert
	- Use `.astype` for Dtype conversion


##### Create a new feature  
- Create a new feature from `[datetime]`, year, month, or hour in the dataset  
- This is a basic and common method in exploratory data.  
- It is important to note that both the train dataset and the test dataset must be created as new features, otherwise an error will occur.  
	- In the process, I found that the Dtype of `[datatime]` in the test dataset is "object".  
		- So you need to convert `.astype()` to `datetime64[ns]`  
  

### How much better did your model preform after adding additional features and why do you think that is?
- After adding features, the Validation score improved significantly from -50.645567 to -29.438956  
- I think this is due to correcting the `[season]` and `[weather]` in the dataset to the correct Dtype  
	- Because season and weather are really two important factors in deciding whether people rent bikes or not  
- Adding the year, month and hour, gives the model more information about the time of day and helps with the regression problem. 



## Hyper parameter tuning
---
### How much better did your model preform after trying different hyper parameters?
- In the first two training sessions, AutoGluon gave the following values for Stack configuration
	- `Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=20`
- Then I try to double these values
	- `num_stack_levels=2, num_bag_folds=16, num_bag_sets=40
- Kaggle score improved from 0.82773 to 0.78101


### If you were given more time with this dataset, where do you think you would spend more time?
- Try different combinations of features, for example excluding humidity or windspeed
- Try tuning the different Hyperparameter tuning 
- Try to increase the upper limit of the training time for the model, this is the most basic first step
	- Because time is the most expensive, leave this step until after the above values have been adjusted.


### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|1|1|2|1.85223|
|add_features|8|8|16|0.82773|
|hpo|20|20|40|0.78101|


### Create a line plot showing the top model score for the three (or more) training runs during the project.
![[model_train_score.png]]


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.
![[model_test_score.png]]

## Summary
---
In this project, I did a hands-on machine learning workflow, and AutoGluon was a good starting point for deciding which model to train without starting from zero, as well as providing suggestions and reference values for which models to train with. This saves a lot of time as we don't have to try each model by hand.  
It was also important to organize the data upfront, just correcting `[season]` and `[weather]` to the correct Dtype and creating a new feature for the datatime could already improve the model's performance significantly.  
Finally, the most time-consuming and difficult part is Hyperparameter tuning, which requires a good understanding of the dataset and experimenting with different combinations of features.  
The training time of the model is also an important issue, as time is the most expensive, so it is important to ensure that all the relevant settings and values are correct before starting the training.  
  
