


# Lambda-BatchTraining-HousePricePredict

## *Kick-Off*

The command to submit spark task can refer to: [command file](https://github.com/saLeox/BigData-HousePricePrediction/blob/main/src/main/resources/command.txt). 

WordCount sample refer to [here](https://github.com/saLeox/BigData-HousePricePrediction/blob/main/src/main/java/gof5/spark/WordCount.java).

MLlib sample refer to [here](https://github.com/saLeox/BigData-HousePricePrediction/blob/main/src/main/java/gof5/spark/MachineLearningApp.java).

## *Here We Go*

 - [x] Data Analysis Exploration, Cleaning & Preparation, and Feature
              Selection refer [here](https://nbviewer.jupyter.org/github/suravimandal/Team1_Data_Analytics/blob/master/Team1_Data-Pipeline.ipynb).  The result can be found in the [Github Repo @suravimandal](https://github.com/suravimandal/team1_big_data).

 - [x] Machine Learning training with *LinearRegression*, *DecisionTree*,  *RandomForests*, *GradientBoostedTrees*.  Main entrance go to [here](https://github.com/saLeox/BigData-HousePricePrediction/blob/main/src/main/java/gof5/spark/HousePricePredictML.java). 
	

	 - To achieve the reusability and low decoupling, we apply **Stratefy
	   Pattern** and **Templeate Pattern**, detail please refer to the
	   [folder](https://github.com/saLeox/BigData-HousePricePrediction/tree/main/src/main/java/gof5/spark/regression/strategy).

	 - Evaluation Metrics, implementation is [here](https://github.com/saLeox/BigData-HousePricePrediction/blob/main/src/main/java/gof5/spark/regression/strategy/Template.java): 

			Mean Squared Error (MSE)
			Root Mean Squared Error (RMSE)
			Mean Absoloute Error (MAE)
			Coefficient of Determination (R2)
			Explained Variance



	***Reference***:
	
	https://spark.apache.org/docs/2.4.7/mllib-classification-regression.html
	
	https://spark.apache.org/docs/2.4.7/ml-classification-regression.html

	https://spark.apache.org/docs/1.6.3/mllib-evaluation-metrics.html
