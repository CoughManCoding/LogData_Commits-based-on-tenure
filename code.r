---
title: "Data Analysis on Software Logs data"
problem Statement: "create a model, a linear regression model, so as to predict, if we can find the the number of months the project has been worked upon given that the code reviews are accepted, and we assume that the number of batches, and the total commit are in someway related to tenure."
output:
  html_document:
    df_print: paged
Author: Siddharth Sharma
---
-------------------------------------------------------------Question 1 -------------------------------------------------------
Marking Required Libraries
```{r}
library(dplyr)
library(corrplot)
library(rpart)
library(entropy)
library(caret)
library(Boruta)
```

```{r}
# Loading the Data : Wikimedia.csv 
Data  <- read.csv("/Users/siddharth/Desktop/UVic/Semester 3/DS for SE/Assignment 1 /wikimedia-dataset1.csv")

# we are using as_tibble to beautify our data presentation
Data<-as_tibble(Data)

#presenting the Data
Data
```
Correlation Plots and Data Visualization: is one of the most important ways to leaarn about the feature, I have plotted other plots as well in the sections down below, However I find the most information in the correlation plots. 

in this code you can find correlation matrix as well as co relation plots of singles features with respect to other as well. We will further drop the highly correlated variables in the coming up chunks, the reason to do so is we want our model to be general and not sensitive to the "outliers" of the Highly co related feature.

```{r}
#finding the correlation between the different tenure and the rest of the features:
Data$tenure <- as.numeric(Data$tenure)
cor_tenure <- cor(Data$tenure, Data)
corrplot(cor_tenure)
#finding the correlation between the different isAccepted and the rest of the features:
cor_isAccepted<- cor(Data$isAccepted, Data)
corrplot(cor_isAccepted)
#finding the correlation between the different numPatch and the rest of the features:
cor_numPatch <- cor(Data$numPatch, Data)
corrplot(cor_numPatch)
summary(Data$numPatch)
# finding the correlation between tall the features:
corrplot(cor(Data[, 1:19]))
```

In the last plot we can see that there is a  "?" symbol, we will now explore various features.
We  try to get some insight on the data with respect to tenure, owner, dirCount, cycmplx & numPatch.

we also plot histograms of tenure and other features, so as to see the distribution of the feature the box plot, give us a very good idea of how the outliers prevail in those particular features and , only those features which were necessary for us to complete our problem statement, were kept.
```{r}
#Exploring the  tenure feature:
hist(Data$tenure, breaks = 15, col = "blue")
summary(Data$tenure)
#Exploring the Owner Feature:
hist(Data$owner)
summary(Data$owner, breaks = 50, col = "green")
#Exploring the Feature dirCount: 
hist(Data$dirCount, breaks = 50, col = "black")
summary(Data$dirCount)
boxplot(Data$dirCount)
#Exploring the Feature: 
hist(Data$cyCmplx, breaks = 20, col = "red")
summary(Data$cyCmplx)
boxplot(Data$cyCmplx)

#Exploring the Tibble Data Feature numPatch
hist(Data$numPatch, breaks = 50, col = "yellow")
summary(Data$numPatch)
boxplot(Data$numPatch)
distinctValues <- unique(Data$numPatch)
count <- length(distinctValues)
print("the count for numPtach distinct and  unique row values")
count
```
qqPlot for Further visualisation of the data,we draw comparison between different features and how are they related to each other.

```{r}
#QQ plot
totalCommit_Plt <- Data$totalCommit
isAccepted_Plt <- Data$isAccepted

qqplot(totalCommit_Plt, isAccepted_Plt, main = "Total Commit VS is Accepted", xlab = "Total Commit", ylab ="isAccepted") 

qqplot(Data$tenure, isAccepted_Plt, main = "Tenure VS is Accepted", xlab ="Tenure", ylab ="isAccepted") 

plot(Data$tenure, Data$fileCount, col = "red", pch = 5, xlab = "Tenture (in Months)", ylab = "File Count", main = "Tenure Vs File Count")

plot(Data$tenure, Data$cmtVolume, col = "black", pch = 3, xlab = "Tenture (in Months)", ylab = "comment", main = "Tenure Vs comment")

plot(Data$tenure, Data$reviewInterval, col = "red", pch = 5, xlab = "Tenture (in Months)", ylab = "Review Interval", main = "Tenure Vs Review Interval")
```

Data Summary: we use str and summary to get all the statistical details of the data.
```{r}
str(Data)
print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")
summary(Data)
```

Data Cleaning &  Feature engineering:

Now we perform data cleaning one of the most important steps after visualization and knowing about the data first of all, we check if there is any null data, and if there is we omit it.

Secondly, we still plot a correlation plot so as to see the summary of numPatch, and the reason why we are doing this is to check while we choose our features that no high correlation should be present between tenure and numPatch

Thirdly, we drop the columns/features that are not necessary : isGenderNeutral is one of those features that have no information the mean is zero the median is zero and overall was 0.

"owner", "request_id","numNewFiles","dirCount","insertions", “deletions" had a high correlation, which we do not want. Furthermore, "Gender" was one of those features which was highly unbalanced, and we did not need it, and hence we chose to drop this as well.
```{r}
#Check is null:
print(is.null(Data)) # so we  do not need to omit any NA values.
Data <- na.omit(Data)
# Check for High correlation:
print("------------------------------------------------------------------------------")
cor_numPatch <- cor(Data$numPatch, Data)
corrplot(cor_numPatch)
summary(Data$numPatch)
print("------------------------------------------------------------------------------")
# drop the uneccessary features:
summary(Data$isGenderNeutral)

# dropping the highly correlated columns & and un
#FileCount is highly correlated with to numNewFiles & dirCount
# we are keeping patchSize and dropping insertions and deletions
#Gender is highly imbalanced, and we are not going to use it, so we are dropping it.
cleanData <- Data[, !names(Data) %in% c("isGenderNeutral", "owner", "request_id","numNewFiles","dirCount","insertions", "deletions", "Gender")]
```

continued ---

We further remove the outliers from the data by using a Z score where we input the dataset and get the data without outliers for the  check if there are any other any values in the data if they were then we would have to use a compute method known as complete(), who is the father filter data with different strategies? We could have used a median,0, 1, mean, or a ny other constant value for the missing data.
Does that score usually for the most of the cases has a threshold of -3 to 3, where randomly choose the threshold of 2.
```{r}
outliers <- function(data, threshold=2) {
  Z <- abs(scale(data))
  data <- data[Z <= threshold]
  return(data)
}
outliers(cleanData)
print("------------------------------------------------------------------------------")
unique(is.na(cleanData)) # there was no outlier in the features that we were interested in, and hence we can move forward, else we would have to compute and use  complete() method in r to fill the Na values so as the dimensions of the sys does not change.
print("------------------------------------------------------------------------------")
print(cleanData)
```
from ECE 535 at UVic we know that choice of normalization it's sort of a black  art, there are a few pre-define cases in which there can be only that particular type of normalization.

In our assignment we use log transformation, as well as min-max transformations. The traits of min-max transformation is that it helps in scaling the data in the range of zero to one, whereas if there is a data that is highly imbalanced and has high degree of skewness, then we use log transformation to tone down and “flatten” the distribution

```{r}
#Normalising the data  normalization.
cleanData <- cleanData %>%
  mutate_at(vars(-one_of(c("isAccepted", "isBugFix"))), ~ (.-min(.)) / (max(.) - min(.)))
print(cleanData)
```

```{r}
# Log Scale :
is.na(cleanData$numPatch)
cleanData$numPatch[cleanData$numPatch == 0] <- 1
cleanData$numPatch <- log(cleanData$numPatch)
hist(cleanData$numPatch, breaks = 50, col = "yellow")
cleanData$patchSize <- log(cleanData$patchSize)
hist(cleanData$patchSize, breaks = 20, col = "red")
```
As asked from us about the data types that got changed while reading of the data by read.csv() method of the RStudio, we here now change the describes to the appropriate ones

There are other ways of feature engineering(here we just take the ones with very less or no corerelation with our target vraible.) in the last chunk of the section. You would see a Boruto method. In which we use Boruto which over the number of iterations give us the most important features for a target variable.

 ------------------------------------------------------- Question 2 ____ -------------------------------------------------------
 
 
```{r}
C <- sapply(Data, class)
print(C)
# Here the isAccepted and Gender etc are boolean but the data type is shown as integer.Hence we will now set the Clumns and their class data types.
# only changing in the data Types that I need in the code
Data <- Data %>% mutate(Gender = as.logical(Gender),isAccepted = as.factor(isAccepted),isBugFix = as.logical(isBugFix),patchSize = as.numeric(patchSize), numPatch = as.numeric(numPatch),tenure = as.numeric(tenure))      

C <- sapply(Data, class)
print(C)
```

Here we choose the tenure as the target label, the one that has to be predicted, and is dependent upon the independent features, the main question arises how did we choose the independent features and the answer is very simple at least, in this particular scenario, they were multiple ways that I tried the first and the basic one was trying out the correlation method, the one feature, which had very less correlation with the other what's take and the rest were dropped.  

There are other ways of feature engineering in the last chunk of the section. You would see a Boruto method. In which we use Boruto which over the number of iterations give us the most important features for a target variable.

Defined the importance of the variable. We can also use random forest algorithm, as well as finding the entropy and information gain of each of the variable respect to other, however, using the random forest was very time consuming  it took over nine hours and yet it wasn't trained this happen due to the large size of data.
 
Problem Statement formulated: 
We created a model, a linear regression model, so as to predict, if we can find the the number of months the project has been worked upon given that the code reviews are accepted, and we assume that the number of batches, and the total commit are in someway related to tenure.

Note: 
I still want to highlight that we want a model to learn from the above mentioned features however, all of these features are either uncorrelated, or has very, very low correlation.
```{r}
#Building Training linear regression Model:
# check if there is Na or Null or any infinite values, as they can still creep in
print(cleanData)
sum(is.infinite(cleanData$tenure))
unique(is.na(cleanData$tenure))

sum(is.infinite(cleanData$numPatch))
sum(is.infinite(cleanData$totalCommit))
sum(is.infinite(cleanData$numPatch))
# Building Linera regression model
Model_linear_regression <- lm( tenure ~ numPatch+ totalCommit + isAccepted , data = cleanData)

print(Model_linear_regression)
summary(Model_linear_regression)

msepredicted <- predict.lm(Model_linear_regression)
mse_Train <- mean((msepredicted - cleanData$tenure)^2)
rmse <- sqrt(mean((msepredicted - cleanData$tenure)^2))
mse_Train
rmse
```

Now, once the models created, we will now tested on test data here we choose ‘go-dataset1' as a test data. We perform the same transformations data cleaning as we did for our training, data set. Furthermore, we assume,that the data in the test data is also  from the identical, independent distribution sampling, the same as or training data.

```{r}
testData <- read.csv("/Users/siddharth/Desktop/UVic/Semester 3/DS for SE/Assignment 1 /go-dataset1.csv")

# to maintian the same dimensions we remove the features that are not needed.
testData <- testData[, !names(testData) %in% c("isGenderNeutral", "owner", "request_id","numNewFiles","dirCount","insertions", "deletions", "Gender")]
```
continued ---
```{r}
outliers <- function(data, threshold = 2) {
  z_scores <- abs(scale(data))
  outliers <- data[z_scores > threshold]
  data <- data[z_scores <= threshold]
  return(data)
}
outliers(testData)
print(testData)

#Normalising the data  normalization.
testData <- testData %>%
  mutate_at(vars(-one_of(c("isAccepted", "isBugFix"))), ~ (.-min(.)) / (max(.) - min(.)))
print(testData)

testData$numPatch[testData$numPatch == 0] <- 1
testData$numPatch <- log(testData$numPatch)

hist(testData$numPatch, breaks = 50, col = "yellow")

testData$patchSize <- log(testData$patchSize)
hist(testData$patchSize, breaks = 20, col = "red")
```

```{r}
# Measure the validity of the model against the test data:
predictedTarget <- predict(Model_linear_regression, newdata = testData)
rmse_Test<- sqrt(mean((testData$tenure - predictedTarget)^2))
rmse_Test
mse_Test <- mean((predictedTarget - cleanData$tenure)^2)
mse_Test
```
Discussion on the liner regression model.
We see than  that our rmse and mse scores for the model, is comparable and does not change drastically we see a mse score of  0.03208138 and rmse  score of 0.1791127 for our model where as,  for our Test data set we get a  rmse score of  and an mse score of : 0.243814 & 0.08186436 respectively. This is due to fact that there is no generalisation.

 -------------------------------------------------------Question 3:  -------------------------------------------------------

However, we need a benchmark or something to compare with and hence, we choose decision tree model, further compare our linear regression model of it. We train our decision tree as well as test the model generated on cleandata, which is our training data and test data, which is the same as go dataset. We get almost the similar performances, and the same root mean square error as well as mean square, error values for this model as well.
```{r}
# Fit the decision tree model, as a second model to compare our results to.

Model_Decision_Tree <- rpart(tenure ~., data = cleanData, method = "anova")
Target_Tenure <- predict(Model_Decision_Tree, cleanData)

# measures for training Data 
rmse <- sqrt(mean((Target_Tenure - cleanData$tenure)^2))
print(rmse)

mse_Train <- mean((Target_Tenure - cleanData$tenure)^2)
mse_Train
```

```{r}
# testing our decision tree model.
Target_Tenure <- predict(Model_Decision_Tree, testData)

# measures for Testing set (decision tree) 
rmse <- sqrt(mean((Target_Tenure - testData$tenure)^2))
print(rmse)
mse_Test <- mean((Target_Tenure - cleanData$tenure)^2)
mse_Test
```


EXTRA : This code will take time to run about 40 minutes for 12 iterations, I would if I have to build an Industry scale model would use boruta or random forest tor find the most important featueres for the model/ project.

Feature Importance and Selection:
```{r}
set.seed(111)
featureImp = Boruta(tenure ~., data = cleanData, doTrace = 3, maxRuns = 12)
print(featureImp)
```

plotting
```{r}
plot(featureImp, las = 2,cex.axis =0.1 )

```
------------------------------------------------------------Question 4: ------------------------------------------------------------
The model is based on linear regression, where one "target value" is dependent on one or more independent variables/features. Linear regression is highly sensitive to outliers when the relationship between two features is highly correlated. Additionally, linear regression only works when the relationship between the independent variable and target feature is linear.

We also evaluate the root mean square error (RMSE) and mean square error (MSE) to assess how well the model performs when transitioning from training data to testing data. We observe that the values do not change significantly, indicating a generalized model. However, there may still be a low degree of overfitting or underfitting.

We then validate our model using decision trees, which are complex machine learning algorithms that combine trees created from random subsets of data to make accurate predictions. The RMSE and MSE for the decision tree model are similar to those of linear regression. Therefore, it is difficult to clearly determine which model is better (as per the "No Free Lunch" theorem).

Regarding data provenance, it was mentioned that the provided datasets are taken from a research paper, but the resource was not provided to us, which prevents us from providing an unbiased overview and commenting on the reliability of the data. Furthermore, this data cannot address questions related to transparency, accountability, etc. The provided metadata was self-explanatory for the variables and sufficient for designing the model. However, the model cannot be used for predictive decisions by Wikimedia, Go, or even ourselves due to unclear data sources. Working with such data can pose legal, ethical, and technical challenges that cannot be ignored.

To measure validity, we used RMSE and MSE. These errors quantify the difference between the predicted value and the actual value. Low scores/errors generally indicate a good model, but they may not capture the essence of the model comprehensively. For our linear regression model, we obtained Multiple R-squared: 0.4258 and Adjusted R-squared: 0.4258, which are not considered good criteria. Moreover, the accuracy of the static data provided cannot be extended to a more dynamic multi-modal data scenario.

To ensure the reliability of our measurement scores, we used another model that produced similar results. I had hoped that the complex nature of decision trees and bootstrapping to shuffle the subsampled data would yield better results, but that was not the case. There could be contention regarding the feature selection approach; however, I tried three approaches, and the least time-consuming one was correlation metrics-based feature selection. I also employed Recursive Feature Elimination (RFE), Baruto and random forest-based techniques, both of which were time-consuming. I want to emphasize that the measurement scores are the same because we performed data cleaning, normalization, outlier removal, and feature selection, which were common to both test data and training data, helping to reduce errors.

Lastly, as mentioned before, the unknown source of the data and the process of formulating answers or choosing problems based on the data present their own challenges. These challenges often lead engineers and analysts to contemplate ethical questions. For example, the "gender" feature, which was complemented by the "is gender-neutral" feature, assumes only two genders, which may be seen as crude in today's world since the gender-neutral feature has no data. Additionally, technical questions such as predicting code review acceptance based on knowledge of patch size can be answered mathematically but may not be practical or ethical.


Thankyou  



