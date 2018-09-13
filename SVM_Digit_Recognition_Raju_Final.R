############################ SVM handwritten digit recognition #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The business problem here is of handwritten digit recognition which comes under field of pattern recognition. 
#Here we have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices in form of a data set. 
#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

## OBJECTIVE:

#The goal here is to  develop a model using Support Vector Machine which should correctly 
#classify the handwritten digits based on the pixel values given as features.

#####################################################################################

# 2. Data Understanding: 
# We use the MNIST data which is a large database of handwritten digits where  
# we have pixel values of each digit along with its label. 
# Number of Observations: 59,999
# Number of variables: 785
# First column is the output variable which labels digit as 0 to 9 
# and rest of the columns/attributes are independent variables.

#3. Data Preparation: 


#Loading Neccessary libraries

#install.packages("caret")
#install.packages("kernlab")
#install.packages("dplyr")
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("gridExtra")
#install.packages("caTools")

library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
library(caTools)


#Loading Data

#setwd("C:/Training/DataScience-Upgrad/Course4-PredictiveAnalytics 2/Module3_Assignment SupportVectorMachine/SVM Dataset")
mnist_data <- read.csv("mnist_train.csv")

#Stratified sampling 
#We will use 5,000 training observations using stratified sampling

colnames(mnist_data)[1] <- "digit"

#set the seed to 100
set.seed(1)

#For Stratified sampling, first we will group by digit column i.e. output column

by_digit <- mnist_data %>% group_by(digit)

#We will use sample_frac function to select the fraction of rows from each group randomly
# This sampled data will be our training data set.

mnist_sample_data <- sample_frac(by_digit, 0.08)

#Understanding Dimensions

dim(mnist_sample_data)
#4799 observation
#785 variables

#Structure of the dataset

str(mnist_sample_data)
# Here all are integer variavles with output vriables  "digit""values are in the range of 0-9.

#printing first few rows

head(mnist_sample_data)
#Here we can see that most variables have values a 0 as these are pixel values of digits.

#Exploring the data

summary(mnist_sample_data)
#Here we can see that most data summary have values a 0 as these are pixel values of digits.

#checking missing value

sum(is.na(mnist_sample_data))
#Since sum is 0 so no missing values

#We can also check this column wise by applying the is.na function for each column
#which gaves the same result.
sapply(mnist_sample_data, function(x) sum(is.na(x)))

#Check for duplicate rows

sum(duplicated(mnist_sample_data)) 
#0 - No duplicate rows

#This is just to check what are the columns which has all zeros but we cant remove as 0
#value is meaningful in determing the digit value
cols_with_all_zeros <- which(colSums(mnist_sample_data[,2:ncol(mnist_sample_data)]) == 0)

#EDA of mnist_sample_data

summary(mnist_sample_data[, 2:785])

#table command will give count of each digit in the sample data set
mnist_digit_table <- table(mnist_sample_data$digit)

#Display table values
mnist_digit_table
#0   1   2   3   4   5   6   7   8   9 
#474 539 477 490 467 434 473 501 468 476

#Change mnist_digit_table to data frame
mnist_digit_table = as.data.frame(mnist_digit_table)

#Summary of digits

summary(mnist_digit_table)
# Min.   :434.0
# Max.   :539.0
# Median :475.0
# 1st Qu.:469.2
# 3rd Qu.:486.8

#Plot to visualise the frequency of digits
plot1 <- ggplot(mnist_sample_data, aes(x=mnist_sample_data$digit)) + geom_bar() + labs(title = "MNIST Digits Bar Plot", x = "Digits", y = "Frequency of Digits(0-9)")

plot2 <- ggplot(mnist_sample_data, aes(x = digit, y = (..count..)/sum(..count..))) + geom_bar() +labs(title = "MNIST Digits Bar Plot", x = "Digits", y = "Frequency of Digits(0-9) in (%)") + scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) + geom_text(stat = "count", aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

grid.arrange(plot1, plot2, nrow = 2)

# Feature standardisation

# Normalising continuous features 
max_pixel_value = max(mnist_sample_data[ ,2:ncol(mnist_sample_data)]) 
mnist_sample_data[ ,2:ncol(mnist_sample_data)] = mnist_sample_data[ ,2:ncol(mnist_sample_data)]/max_pixel_value

#Making our target class to factor

mnist_sample_data$digit<-factor(mnist_sample_data$digit)

#####################################################################################

# Splitting the data between train and test

set.seed(1)
indices = sample.split(mnist_sample_data$digit , SplitRatio = 0.7)
train = mnist_sample_data[indices, ]
test = mnist_sample_data[!indices, ]

# Model Building
#####################################################################################
#--------------------------------------------------------------------
# Linear model - SVM  at Cost(C) = 1
#####################################################################

# Model with C =1
model_1<- ksvm(digit ~ ., data = train,scale = FALSE,C=1)

# Predicting the model results 
evaluate_1<- predict(model_1, test)

# Confusion Matrix - finding accuracy, sensitivity and specificity
conf_mat_c1_linear <- confusionMatrix(evaluate_1, test$digit)
conf_mat_c1_linear

#Calculate average Sensitivity for all class
mean(conf_mat_c1_linear$byClass[,1])
#Calculate average Specificity for all class
mean(conf_mat_c1_linear$byClass[,2])

# Accuracy    : 0.9472
# Sensitivity : 0.946308         
# Specificity : 0.9941276

#--------------------------------------------------------------------
# 4.2 Linear model - SVM  at Cost(C) = 100
#####################################################################

# Model with C =100.
model_100<- ksvm(digit ~ ., data = train,scale = FALSE,C=100)

# Predicting the model results
evaluate_100<- predict(model_100, test)

# Confusion Matrix - finding accuracy, sensitivity and specificity
conf_mat_c100_linear <- confusionMatrix(evaluate_100, test$digit)
conf_mat_c100_linear
#Calculate average Sensitivity for all class
mean(conf_mat_c100_linear$byClass[,1])
#Calculate average Specificity for all class
mean(conf_mat_c100_linear$byClass[,2])

# Accuracy    : 0.9493
# Sensitivity : 0.9487684         
# Specificity : 0.9943576 


######################################################################

#Using Linear Kernel
Model_linear <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)

#confusion matrix - Linear Kernel
conf_mat_linear <- confusionMatrix(Eval_linear,test$digit)
conf_mat_linear

#Calculate average Sensitivity for all class
mean(conf_mat_linear$byClass[,1])

#Calculate average Specificity for all class
mean(conf_mat_linear$byClass[,2])

# Accuracy        - 0.8992356
#Mean Sensitivity - 0.8977126
#Mean Specificity - 0.9888084

#              Class:0  Class:1  Class:2  Class:3  Class:4  Class:5  Class:6  Class:7 Class:8  Class:9
#Sensitivity   0.97887   0.9877  0.90909  0.80952  0.92143  0.84615  0.97887  0.90667	0.78571  0.85315
#Specificity   0.99152   0.9906  0.98457  0.98762  0.98691  0.98701  0.99614  0.99069	0.98999  0.98302

######################################################################

#Using RBF Kernel
Model_RBF <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)

#confusion matrix - RBF Kernel
conf_mat_rbf <- confusionMatrix(Eval_RBF,test$digit)
conf_mat_rbf

#Calculate average Sensitivity for all class
mean(conf_mat_rbf$byClass[,1])

#Calculate average Specificity for all class
mean(conf_mat_rbf$byClass[,2])

#Accuracy : 0.9472
#Mean Sensitivity - 0.946308
#Mean Specificity - 0.9941276

#             Class:0   Class:1 Class:2  Class:3  Class:4  Class:5  Class:6  Class:7 Class:8  Class:9
#Sensitivity  0.99296   0.9877  0.93706  0.90476  0.95714  0.91538  0.99296   0.9667   0.87143  0.93706
#Specificity  0.99614   0.9930  0.99846  0.98994  0.99384  0.99389  0.99769   0.9922  0.99384  0.99228

#We can clearly see that SVM Model using RBF Kernel gives better accuracy and Sensitivity compared to what
#we got using linear Kernal method.


#####################################################################
# Hyperparameter tuning and Cross Validation  - Linear - SVM 
######################################################################

# We will use the train function from caret package to perform crossvalidation

trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"

set.seed(100)

# making a grid of C values. 
grid <- expand.grid(C=c(0.01, 0.1, 1, 5,10))

# Performing 5-fold cross validation
fit.svmLinear <- train(digit~., data=train, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl,scale = FALSE)

# Printing cross validation result
print(fit.svmLinear)
# The final value used for the model was C = 0.1
# Accuracy - 0.9089447

# Plotting "fit.svm" results
plot(fit.svmLinear)
#Best accuracy is at c = 0.1 and  Accuracy = 0.9089447

# Valdiating the model after cross validation on test data

evaluate_linear_test<- predict(fit.svmLinear, test)
conf_mat_linear_test <- confusionMatrix(evaluate_linear_test, test$digit)
conf_mat_linear_test

#Calculate average Sensitivity for all class
mean(conf_mat_linear_test$byClass[,1])

#Calculate average Specificity for all class
mean(conf_mat_linear_test$byClass[,2])

#Accuracy : 0.9138
#Mean Sensitivity - 0.9122906
#Mean Specificity - 0.9904221

############   Hyperparameter tuning and Cross Validation - svmRadial #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.01, 0.1,0.5,1), .C=c(0.1, 0.25, 0.5, 1) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svmRadial <- train(digit~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl,scale = FALSE)

print(fit.svmRadial)
#The final values used for the model were sigma = 0.01 and C = 1.
#sigma = 0.01 and C = 1   Accuracy = 9276865
plot(fit.svmRadial)

# Validating the model results on test data
evaluate_non_linear<- predict(fit.svmRadial, test)
conf_mat_nonlinear_test <- confusionMatrix(evaluate_non_linear, test$digit)
conf_mat_nonlinear_test

#Calculate average Sensitivity for all class
mean(conf_mat_nonlinear_test$byClass[,1])

#Calculate average Specificity for all class
mean(conf_mat_nonlinear_test$byClass[,2])

# Accuracy    - 0.9472
# Sensitivity - 0.9122906
# Specificity - 0.9904221

########################################################################
##Optimise C value for sigma = 0.01

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.01), .C=c(1,2,3,4) )

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svmRadialOpt <- train(digit~., data=train, method="svmRadial", metric=metric, 
                       tuneGrid=grid, trControl=trainControl,scale = FALSE)

print(fit.svmRadialOpt)
#The final values used for the model were sigma = 0.01 and C = 3.
#sigma = 0.01 and C = 3   Accuracy = 9393013

plot(fit.svmRadialOpt)

# Validating the model results on test data
evaluate_non_linear_opt<- predict(fit.svmRadialOpt, test)
conf_mat_nonlinear_opt_test <- confusionMatrix(evaluate_non_linear_opt, test$digit)
conf_mat_nonlinear_opt_test

#Calculate average Sensitivity for all class
mean(conf_mat_nonlinear_opt_test$byClass[,1])

#Calculate average Specificity for all class
mean(conf_mat_nonlinear_opt_test$byClass[,2])

# Accuracy    - 0.9541
# Sensitivity - 0.9535253
# Specificity - 0.9948974

############################Summary and Conclusion #####################################

#Linear model - SVM  at Cost(C) = 1

#Model Name : model_1
# Accuracy    : 0.9472
# Sensitivity : 0.946308         
# Specificity : 0.9941276

#Model using Linear Kernel (kernel = "vanilladot")

#Model Name : Model_linear
# Accuracy        - 0.8992356
# Mean Sensitivity - 0.8977126
# Mean Specificity - 0.9888084

#Model using RBF Kernel (kernel = "rbfdot")

#Model Name : Model_RBF
#Accuracy : 0.9472
#Mean Sensitivity - 0.946308
#Mean Specificity - 0.9941276

#Model Linear - SVM using Cross Validation (CV = 5, method="svmLinear", C = 0.1)

#Model Name : fit.svmLinear (Final value at c = 0.1)
#Accuracy : 0.9138
#Mean Sensitivity - 0.9122906
#Mean Specificity - 0.9904221

#Model RBF - SVM using Cross Validation (CV = 5, method="svmRadial", sigma = 0.01 and C = 1)

#Model Name : fit.svmRadial (Final value at sigma = 0.01 and C = 1)
# Accuracy    - 0.9472
# Sensitivity - 0.9122906
# Specificity - 0.9904221

#Model RBF - SVM using Cross Validation and Optimised C value for sigma = 0.01 
#(CV = 5, method="svmRadial", sigma = 0.01 and C = 3)

#Model Name : fit.svmRadialOpt (Final value at sigma = 0.01 and C = 3)
# Accuracy    - 0.9541
# Sensitivity - 0.9535253
# Specificity - 0.9948974

#Final Model Selection
#Based upon all the models, model fit.svmRadialOpt is the final model.
#The model fit.svmRadialOpt got from  method="svmRadial" , CV = 5 ,  sigma = 0.01 and C = 3 gives the
#best accuracy as 0.9541 and Sensitivity as 0.9535253

###########Test the Model on Unseen Test Data which has not been touched in model buildingand evaluation
mnist_unseen_test_data <- read.csv("mnist_test.csv",stringsAsFactors = F, header = F)
head(mnist_unseen_test_data[,"digit"], n=20)
#We will use subset of this unseen test data using  Stratified sampling for testing the model.
#Stratified sampling 

colnames(mnist_unseen_test_data) <- colnames(test)

#set the seed to 100
set.seed(1)

#For Stratified sampling, first we will group by digit column i.e. output column

by_digit_test <- mnist_unseen_test_data %>% group_by(digit)

#We will use sample_frac function to select the fraction of rows from each group randomly
# This sampled data will be our training data set.

mnist_unseen_test_sample_data <- sample_frac(by_digit_test, 0.3)

#Understanding Dimensions
dim(mnist_unseen_test_sample_data)
#3000 observation
#785 variables

# Normalising continuous features 
max_pixel_value = max(mnist_unseen_test_sample_data[ ,2:ncol(mnist_unseen_test_sample_data)]) 
mnist_unseen_test_sample_data[ ,2:ncol(mnist_unseen_test_sample_data)] = mnist_unseen_test_sample_data[ ,2:ncol(mnist_unseen_test_sample_data)]/max_pixel_value


#Making our target class to factor

mnist_unseen_test_sample_data$digit<-factor(mnist_unseen_test_sample_data$digit)

# Validating the model results on unseen sample test data
set.seed(7)
model_1
evaluate_unseen_test_data <- predict(model_1, mnist_unseen_test_sample_data)
evaluate_unseen_test_data <- predict(fit.svmRadialOpt, mnist_unseen_test_sample_data)
conf_mat_unseen_test_data <- confusionMatrix(evaluate_unseen_test_data, mnist_unseen_test_sample_data$digit)
conf_mat_unseen_test_data

#Calculate average Sensitivity for all class
mean(conf_mat_unseen_test_data$byClass[,1])

#Calculate average Specificity for all class
mean(conf_mat_unseen_test_data$byClass[,2])

# Accuracy    - 0.9543
# Sensitivity - 0.9538694
# Specificity - 0.9949307

#The model fit.svmRadialOpt got from  method="svmRadial" , CV = 5 ,  sigma = 0.01 and C = 3 gives the
#accuracy as 0.9543 and Sensitivity as 0.9538694 for unseen test data which is slighly even better than validation data.