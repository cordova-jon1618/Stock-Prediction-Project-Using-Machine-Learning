# Stock Prediction Project Using Machine Learning
### Can Machine Learning assist with Predicting Stock Market prices?

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/85463855-0b12-41bb-b93d-053ee2be9667)

# Meet the Team:
#

## Jonathan Cordova [@cordova-jon1618] - https://github.com/cordova-jon1618
## Phani Challabotla - Assisted with this project

#

### This project for COMP 542 uses the following Machine Learning algorithms:

•	Regression Trees

•	Extra Regression Trees

•	Random Forest

•	Artificial Neural Network (Recurrent Neural Network)


# Introduction:
The problem we posted is: “Can Machine Learning assist with Predicting Stock Market prices?”. Our goal is to build a prediction Machine Learning model to predict stock closing prices using supervised learning. The problem of predicting how the stock market that often benchmarks the U.S. economy is an essential and significant question posed by many individuals, investment firms, and government institutions. As information regarding how the markets will react affects retirement funds, corporations, banks, and other entities. Billions of dollars are paid out by large investment firms to hire the best mathematicians, data scientists, and machine learning engineers, to attempt to predict and make the best financial decisions for people across the globe. Hence, our project will attempt to answer the question, “Can Machine Learning assist with Predicting Stock Market prices?”

## Getting The Data: 
The data we decided to use for our training our Machine Learning model is to use the S&P 500 index fund. The provider of the data source is the website called Yahoo! Finance. This website is often use by investors and day traders to view historical stock price data along with numerous other variables for the stock. There is a python library that was created specifically for Yahoo! Finance and pulling stock market data. This library is the ‘yfinance’ library that can be installed for Python. The S&P 500 index stock can be located by the stock ticker of ‘^GSPC’ on the Yahoo! Finance website.

## Data Preparation: 
For data preparation we used the Technical Analysis Library, this library provides functions for generating technical indicators from the stock data that is found in Yahoo! Finance. We used this library in conjunction with the Yahoo! Finance library to generate the momentum indicator, the money flow indicator, and the relative strength indicator.

## About the Data: 
When one pulls the data for the S&P 500 from Yahoo! Finance (through the yfinance python library), the data set uses ‘Dates’ as the data set’s index. Additionally, the features for the data set include Open, High, Low, Close, Volume, Dividends and Stock Splits. Because we are dealing with the S&P 500 and the stock is known as an index fund stock, there is no useful information in regard to the dividends and stock splits because these features are only relevant to individual company stock. Hence, we went ahead and removed the Dividends and Stock Splits columns (features). At first, we created the momentum, money flow, relative strength, and rate of change indicator using the TA-Lib (Technical Analysis Library). After seeing that the there was a high correlation between the Rate of Change ratio indicator and the Momentum indicator. We decided to remove the Rate of Change ratio because it had a correlation of 0.9583 to the Momentum indicator. Additionally, we added two new indicators which captured the daily price range of the stock and the daily mean. To create the Daily Range indicator, we took the Close stock price and subtracted it with the Open stock price. To create the Daily Mean indicator, we took the High stock price and added it to the Low stock price for that date and divided by 2.0 to get the daily mean. Another change made to the original data set, was to update the date range that is being represented. Hence, the date range was reduced to 32 years, starting from October 1st, 1990, through October 1st, 2022. 

## Data Visualization: 
To visualize the data, we used histogram plots to visualize the frequency of values for each feature in the full data set. Please see example below for histogram plot from presentation 1.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/8c2d0cf8-4a40-4c96-9d42-468774b2ecb0)

The describe() function was then called to view information about the full data set. These metrics include minimum, maximum, mean, standard deviation, the 25th percentile, 50th percentile (median), 75th percentile of each feature in the full data set. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/92a52d2f-8a1c-44c7-8617-c35b80e4777d)

To visualize the data set and gain insight, we used a scatter plot matrix using the Seaborn library for data visualization. This is how we determined that there was a high correlation between the Rate of Change ratio and the Momentum indicator of correlation value 0.9583. Please see below for scatter plot matrix, correlation matrix and correlation values.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/d3bd9d74-c9a5-4baa-8031-9e337289234d)

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/a42ed992-3b35-49e3-8228-7d703c8f7396)

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/4a4cc454-29a6-46ed-a63a-c2b928058636)

In addition to the above insight, the S&P 500 Time Series data was plotted with ‘Dates’ as the x-axis and the ‘Close’ prices as the y-axis. Please see below for details.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/2d8511bd-ac8b-475f-878e-7e89c89c58a5)

After separating the training set and the test set, we visualized that the data sets were separated correctly by plotting the training set as a different color to the test set, both plotted on the same time series line plot.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/8c20c107-fbe6-40a9-abdc-4cda07e882a4)

# Related Work:

There is one paper that served as inspiration for using a neural network algorithm for stock market prices. Though we are far from getting the same results as the paper, it served as a resource to determine if we were heading in the right direction in regard to building the Neural Network model. This research paper is the following:

•	Qiu, Jiayu, et al. “Forecasting Stock Prices with Long-Short Term Memory Neural Network Based on Attention Mechanism.” PloS One, U.S. National Library of Medicine, 3 Jan. 2020, <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941898/>.

# Methods:

The primary methods we used were Regression Tree (decision tree) models and Artificial Neural Network (Recurrent Neural Network) models to solve this problem. The reasons we went with these models were because historical stock prices data is a time series data set. Moreover, the recurrent neural network was chosen because this model allows us to feed stock prices as input features to make future prediction on stock prices. In regard to the Regression Tree, we also decided to try Extra Regression Tree, and Random Forest as additional models to see if these models performed better or worst with the same data set.

# Evaluation:

The metrics we chose as evaluation metrics were the Mean Squared Error and the R Squared Score. The two main reasons for choosing these metrics are (1) we have a time series data set and (2) because we are trying to predict future ‘Close’ price based on past ‘Close’ price data. So are primary algorithms being Regression Tree and Artificial Neural Networks. 

## Regression Tree Evaluation – 

The Regression Tree’s Mean Squared Error was 1,378 and the R-squared score was 0.9728. Note: There is no reason to calculate accuracy for this model because we have a better measure which is R-squared. This is important to note because the data is a time series data set and moreover our first algorithm is a regression-based algorithm, so accuracy is not an effective measure as the R-squared score is. It is also important to note that the mean squared error must be seen as a relative measure because the value depends on the range of the data set. We know that the S&P 500 stock price has a ‘Close’ maximum of around $4,796 and a ‘Close’ minimum  of around $16.

## Extra Regression Tree Evaluation – 

Additionally, because we had the data prepared already, we were curious to see if a better Regression Tree model existed to improve the performance. We decided to use an Extra Regression Tree to test this hypothesis.  We change the lines of code to become an Extra Regression Tree model instead and fine-tuned the model to use 100 estimators as parameters to add a bit of randomness into the Extra Regression Tree model to see if this would improve performance. Our Mean Squared Error improved to 531.10 and the R-squared score improved to 0.9892. In comparison to the Regression Tree model, the Extra Regression Tree performed a lot better than the initial model.

## Random Forest Evaluation – 

Moreover, for the same reason as above, we had the data already prepared to be used in a Regression Tree model, hence we decided to implement this algorithm model as well. Our hypothesis relied on the believe that by using randomized regression trees, the performance of the model might be improved. We applied the same techniques in training the model as we did with the Regression Tree model and the Extra Regression Tree model. Surprisingly, we were incorrect, the Random Forest model did not perform as well as the Extra Regression model. The Mean Squared Error was 711.08 and the R-Squared Score was 0.9855. This was better than the Regression Tree model but worst than the Extra Regression Tree model. In regard to the Regression Tree type of algorithms, the algorithm that performed best was the Extra Regression Tree algorithm.

## Artificial Neural Network (Recurrent Neural Network) Evaluation – 

The last algorithm that we decided to implement was an artificial neural network algorithm. After much research, we went with the subclass neural network algorithm called the Long Short-Term Memory Model. The decision to go with a neural network model was that originally, we had decided to go with the Support Vector Machine – Regression model, but it was only later that it was discovered that this algorithm was not ideal to work with a time series data set. After seeking guidance from the professor, the artificial neural network algorithm was recommended. Because the purpose of our project is to answer the question, “Can Machine Learning assist with Predicting Stock Market prices?”, the recurrent neural network algorithm became an ideal candidate for our project. The Sequential version of the model was used for this algorithm because we decided to only use the previous 100 days of historical ‘Close’ prices as input for the algorithm. After training our model, the mean squared error was 38,013.78 and the R-squared score was -0.06. We believe the reason for this large error and bad R-squared score is because we are trying to predict a total the price of the ‘Close’ price of the S&P500 stock. But each prediction starting with 1 day prediction of the test data is based on the 100 previous days of stock price. So, each prediction on each consecutive day is based on the predicted value of the last 100 days. Knowing this logic, we can see why the Mean Squared Error and R-squared score start to become unreliable as future predictions are based on past predicted prices that were also based on previously predicted prices. This can cause the mean squared error and r-squared score to become increasingly unreliable.

# Conclusion:

In conclusion, to answer the original question of “Can Machine Learning assist with Predicting Stock Market prices?”, we conclude that it is very difficult to predict stock market prices. In regard to the Regression Tree model, we had provided features as input to predict the target ‘Close’ price. However, in the real world, these features are not available in a real-time manner and would only be available the following day. So even if the Regression Tree performed best in our project, we had assumed that the feature data for predicting the stock prices is available in real-time. In regard to the Artificial Neural Network (recurrent neural network) model, it is more appropriate because it mimics the real world. In this model, we are only feeding the available ‘Close’ data from the past 100 days, sequentially. For this model, we do not feed data that is available on the same day that we are trying to predict stock prices for. Though the neural network did not perform well, the model’s strength depends on the fact that it reflects the real-world more realistically, as same day data is not available as input.

In our future work, we are interested in ways that we might improve both these models to reflect a more pragmatic reflection of the real world. One idea is to try to use only features from the day before for predicting stock price in the regression tree. Another future work area of research is improving the neural network model by not only feeding it the previous 100 days of ‘Close’ price data, but also feeding it additional indicators that might reflect future price estimate. There are many types of indicators in the world of stock price prediction. Our project relied strongly on technical indicators and stock price data. But one future areas of consideration for future model improvement include, fundamental indicators, which are metrics that can be obtain from companies’ financial statements (income statements, balance sheets, cash flow statements, and stockholder equity sheets).The areas were investment research and management crosses with machine learning is a fascinating subfield. In addition to stock markets, there are numerous other financial markets that can benefit from Machine Learning, such as Future Exchanges, Foreign Exchange Currency Markets, Bond Markets, and Commodity Markets.

# References: 

•	Qiu, Jiayu, et al. “Forecasting Stock Prices with Long-Short Term Memory Neural Network Based on Attention Mechanism.” PloS One, U.S. National Library of Medicine, 3 Jan. 2020, <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6941898/>.

•	Géron Aurélien. Hands-on Machine Learning with Scikit-Learn, Keras and Tensorflow: Concepts, Tools, and Techniques to Build Intelligent Systems. O'Reilly, 2023.

•	Jansen, Stefan. Hands-on Machine Learning for Algorithmic Trading: Design and Implement Investment Strategies Based on Smart Algorithms That Learn from Data Using Python. Packt Publishing Ltd., 2018.

•	Yahoo! Finance, Yahoo! https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC

•	yfinance 0.1.74 https://pypi.org/project/yfinance/0.1.74/ (Note: Please use version 0.1.74)

•	TA-Lib : Technical Analysis Library https://ta-lib.org/ (Note: Please use the command ‘conda install -c conda-forge ta-lib’ to install the technical analysis library).

# Code:

## Data Preprocessing Code:

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/bdce70c3-46e8-4326-9ce0-c0debb8a182c)

The first step is to pull the data from Yahoo! Finance. Store the data in a panda’s data frame.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/f27b7021-3648-4b05-979c-95833c4d7793)

Clean the data set by creating additional indicators, removing other indicators that will no longer be use. Filtering the time range to include only October 1st, 1990, through October 1st, 2022. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/de31735e-0195-49b7-98df-8698ed549471)

We are then left with a total of 8,063 total tuple (rows) in the final data set. We also view the data types for each feature.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/89383bec-2c08-42f8-b62a-130601d33821)

We view the total count for rows, mean, standard deviation, minimum, maximum, 25% percentile, 50% percentile, and 75% percentile of each feature 
 
## Regression Tree Code:

### Summary: 

To begin with the implementation of the Regression Tree. We took the training set features (x training set) and used the Standard Scaler function to scale the data using the fit() method to generate a metric that will be used to transform the data when we apply the transform() method. The next step was to create an instance of the Regression Tree by calling ‘DecisionTreeRegressor’ and passing the tuning parameters including ‘squared error’ for criterion, which uses mean squared error as a metric to train. We trained the model on the scaled  x-train set data and using the y-train set data as the ‘target’. In this case, the ‘target’ is the ‘Close’ price for that date. Once the model had finished training, we applied the same Standard Scaler metric that was previously calculated with the training set and use it now on the test set of data. Afterwards, we use the predict() function of the Regression Tree and pass the x-test set as a parameter to generate a y-predicted ‘target’ data set. This will then be used in our evaluation to compare the results of what was predicted against the results of the actual y-test set. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/5568a94c-4812-4a4b-9e84-c207c496ef9e)

The training data set and the test data set is separated based on the date. If the date falls between 10/01/1990 – 04/01/2022, the data belongs to the training set, otherwise if the data is between 04/01/1990 – 10/01/2022, then it belongs to the test data set. 

In this code below, we are using the StandardScaler function to bring the training data set into a range that is closer to the mean. This will make training the algorithm more effective.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/21b4e2e8-0c6e-4dbe-91be-4db0e6559485)

We use the mean squared error as the criteria for training the decision tree regression. We then train the model with the training set. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/89a11f73-015d-4a01-8aca-e75a54aa45d0)
 
After training the model, we apply the same scaler metric that was calculated from the x-training data set and apply this scaler metric to the x-testing set and use that scaled x-testing set to predict the ‘close’ price. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/bb0c05d4-7566-48ab-bdc3-ab902481414c)

We compare the y-test set and the y-predicted sets with each other to determine how the model did in its prediction. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/3ede700e-bc59-4edb-93b0-6e342dd05c04)

We take the ‘Date’ index from the test set and manually add it to the y-predicted target set because the output that was generated did not have this index. We also add the ‘Close’ column name to the y-predicted result. We go ahead and plot the results. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/0517a5f6-d228-48be-a5cc-268a83046e28)

## Neural Network Code:

### Summary: 

To begin with the implementation of the Neural Network. The training set and testing set is separated by dates, before April 1st, 2022, for training, and after April 1st, 2022 for testing. This allows us to separate the test set to cover 6 months of time series data, while the rest falls under the training set range.  The MinMaxScaler is used to bring the data range between 0.0 and 1.0 to make training more efficient. The algorithm will work by using the 100 days of ‘Close’ price data to be use as the X input to calculate the Y output (target) ‘Close’ price. Once the training data set is prepared, we will build the neural network model (long short-term memory) select parameters such as activation function that will be use. Once the model has been built, we will compile the model, and start training the model using the training set with Mean Squared Error as the loss function for this model. We will run it for five epochs for training. After training is complete, the model will be ready to take in the testing set data. To prepare the test set data, we append the last 100 days of ‘Close’ price data from the training set onto the test set because the first ‘target’ of the test set will rely on previous 100 days of ‘Close’ price data as input. We then apply the MinMaxScaler metric that was previously calculated from the training set data, and we apply this metric to the test data set using the transform() function. We separate the test data set ‘Close’ price data to be use as the X input to calculate the Y output (target). We then predict the predicted ‘target’ values using the x-test set. Finally, we evaluate the y-test set and the y-predicted set to determine how well the model did.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/a0c38096-05e6-43a9-a794-5de4335e3805)

In this code, we turn the ‘Date’ index into a field in order to separate the training set and testing set. After separating the training set to include data from 10/01/1990 through 04/01/2022 and separating the testing set from 04/01/2022 through 10/01/2022. We turn the ‘Date’ column back into the index for the data set. We go ahead and create two new dataframe objects in python and add our training set and testing set data onto them with the header ‘Close’. 

In this code below, we are using the MinMaxScaler function to bring the training data set between the range of 0.0 and 1.0. This will make training the algorithm more efficient. Note: We tried the StandardScaler function as our first method for scaling the data, however, the StandardScaler did far worst results then when we used the MinMaxScaler.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/5ccfc415-86a0-482c-aab2-ca7f2bf112cf)

In this code below, we are making sure the training set uses the previous 100 days of ‘Close’ price as input (features) for training the algorithm, this is why we see that the target (y_train) set starts at index 100.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/4ab39248-be1d-4e58-91da-703955019480)

Here we start creating the neural network algorithm model we will be using. We use the Sequential() model approach to tell that the model will have one tensor input and 1 tensor output. We are using the Long Short-Term Model (LSTM) as the type of neural network algorithm. The activation functions used is the ReLU activation function. The Dropout rate refers to the regularization to prevent overfitting.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/09685edd-9577-4088-ad5d-9c4c89e865a5)
 
We call the summary() method to preview the model we just created. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/a49104a8-63f0-4711-9cc3-651c22b271eb)

In the next step, we go ahead and compile the model and start training the training set. The parameters stated upon compilation are what optimizer will be used. In this case, we stated that ‘adam’ which means that the model will use an adaptive learning rate during training. The mean squared error will be use as a metric that the algorithm will use determine if the Mean squared error is being reduce. We decided to also track additional metrics which include the mean absolute error and the mean absolute percentage error, these can be referred to later during our evaluation phase. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/6b5000ae-efd5-49fa-a6ba-b75dbee0a183)

In the below, the last 100 days’ closing prices were appended to the test set before we can use the testing set. The reason for this is because the test set relies on the previous 100 days as input to determine the predicted closing price. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/cae5c576-b585-4dcb-adc1-b42280bb6097)
 
The min-max scaler instance that was previously determined from the training set is used to transform the test set data as well. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/4aca24d6-67db-4f01-9270-c86338b54eef)

In the below, we precede with the same steps used in the training data, we make sure that the first target predicted closing price in the test set is able to use the 100 previous days of closing price as input.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/739a47b2-67e6-48a6-8e3b-2603bafbc39f)

We use the evaluate() function to evaluation the metrics between the x_test and the y_test to make sure there are no mistakes and no discrepancies between the test set input and output.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/8b953740-3080-47b0-96bb-8c5ea02d15c9)
 
We use the predict() function and pass the x_test data set to generate what the target predicted values will be (y_predicted). 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/bfddc36a-8b8d-467c-ad6c-7a27903b6a23)

Finally, we view the metrics of the mean squared error, r-squared score, mean absolute error and the root mean squared error. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/1ec3af9e-804e-4c1e-b374-e3ed0b02c9ab)

Note: The evaluation shows that the neural network did not perform well because unlike the Regression Tree model, the Neural Network uses the previous ‘Close’ prices, whereas the Regression Tree used other features that in real life will only be available after the day as come to an end. So, we conclude that the neural network model reflects the real-world more realistically, as same day feature data is not used as input.

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/b309320c-1e4c-43fb-8f41-3fec7ec324b9)

We plot the y_predicted against the y_test set, however, before we can plot it, we create two new data frame objects, we also grab the ‘Date’ index from the original test data set because the current y_predicted and y_test target sets do not have these labels. We use the dataframe objects created to add the ‘Date’ index and rename the column as ‘Close’. Afterwards, we can finally plot the results. 

![image](https://github.com/cordova-jon1618/stock-prediction-project-using-machine-learning/assets/29684905/2707d2fd-324e-4bc3-9521-8f3422b3c393)

Bonus: Another evaluation we decided to add-on is to scale the y_test and y_predicted back to their original size and calculate evaluation metrics such as Mean Squared Error, Mean Absolute Error, R-squared score, and the Root Mean Squared Error. But these did not give any additional insight then what we already had calculated as metrics.

