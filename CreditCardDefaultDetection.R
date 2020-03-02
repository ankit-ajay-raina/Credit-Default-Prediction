# Author: Ankit Raina
# Date of Creation: January 23, 2019
# Title: Credit Card Fraud Detection

## Loading packages
require(tidyverse)
require(dplyr)
require(ggplot2)
require(scales)
require(caret)
require(MASS)
require(mlbench)
require(FSelector)
require(e1071)
require(randomForest)
require(pROC)
require(xgboost)
require(LiblineaR)
require(rpartScore)
require(naivebayes)
require(RRF)
require(DMwR)
require(InformationValue)
require(corrplot)

#Specify options
options(scipen=999)         # Avoid exponential notations
options(max.print=999999) 


## Helper Functions

# Function to create histogram
create_histogram <- function(data, col, col_name, num_bins){
  ggplot(data, aes(x=col)) +
    geom_histogram(bins = num_bins, fill = "purple", col = "blue", alpha = 0.3) +
    labs(title=cat("Distribution of", col_name), x=col_name, y="Count") +
    geom_density(alpha=.2, fill="#FF6666") +
    geom_vline(aes(xintercept=mean(col)),
               color="blue", linetype="dashed", size=1) +
    geom_vline(aes(xintercept=median(col)),
               color="red", linetype="dashed", size=1)  +
    scale_x_continuous(breaks = seq(min(0), max(90), by = 5), na.value = TRUE)
}


# Function to create bar plot
create_bar_plot <- function(data, col, col_name){
  ggplot(data, aes(x=col, fill = col)) + 
    geom_bar(aes(y = (..count..)/sum(..count..))) +
    labs(title= cat("Distribution by", col_name), 
         x=col_name,
         fill=col_name,
         y="Percentage")
}


# Function to create stacked bar plot
create_stacked_bar_plot <- function(data, col1, col1_name, col2, col2_name){
  ggplot(data) + 
    geom_bar(aes(y =(..count..)/sum(..count..), x=col1, fill=col2)) +
    labs(title= cat("Distribution by", col1_name, "and", col2_name), 
         x=col1_name,
         fill=col2_name,
         y="Percentage")
}


# Function to create feature plot
create_feature_plot <- function(col_x, col_y, plot_type){
  featurePlot(x = col_x,                       
              y = col_y,    
              plot = plot_type,                         
              scales=list(x=list(relation="free"), y=list(relation="free")), auto.key=T
  )
}

# Function to calculate classification accuracy
calc_acc = function(actual, predicted) {
  mean(actual == predicted)
}

# Function to get the best model
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

# Function to plot ROC
plot_roc <- function(probs, target){
  roc_ROCR <- performance(prediction(probs, target), measure = "tpr", x.measure = "fpr")
  plot(roc_ROCR, main = "ROC curve", colorize = T)
  abline(a = 0, b = 1)
}

# Function to calculate KS statistic
KS <- function(pred, depvar){
  require("ROCR")
  p   <- prediction(as.numeric(pred),depvar)
  perf <- performance(p, "tpr", "fpr")
  ks <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))
  return(ks)
}


## Reading the dataset
credit_card_data <- read.csv("UCI_Credit_Card.csv", header = T)


## Data Cleaning

# Structure of the dataset
str(credit_card_data)

# Glance at the data
head(credit_card_data)

# Summary of the data
summary(credit_card_data)

# Determining observation with all variables missing (all null values)
credit_card_data[!complete.cases(credit_card_data), ]

# Determining which variables have missing values
sapply(credit_card_data, function(x) sum(is.na(x)))

# We can see that we have no missing values

# Dropping the ID Column
credit_card_data$ID <- NULL

# Converting categorical variables to factors

# Converting variable SEX having values (1,2) to (Male, Female)
credit_card_data$SEX = as.factor(credit_card_data$SEX)
levels(credit_card_data$SEX) <- c("Male","Female")

# Converting variable EDUCATION having values (0,1,2,3,4,5,6) 
# to (Unknown, Graduate school, University, High school, Others, Unknown, Unknown)
credit_card_data$EDUCATION = as.factor(credit_card_data$EDUCATION)
levels(credit_card_data$EDUCATION) <- c("Unknown", "Graduate School", "University", "High school", "Others", "Unknown", "Unknown")

# Converting variable MARRIAGE having values (0,1,2,3) to (Unknown, Married, Single, Others)
credit_card_data$MARRIAGE <- as.factor(credit_card_data$MARRIAGE)
levels(credit_card_data$MARRIAGE) <- c("Unknown" , "Married" , "Single" ,"Others")

# Converting variable default.payment.next.month having values (0,1) to (No, Yes)
credit_card_data$default.payment.next.month <- as.factor(credit_card_data$default.payment.next.month)
levels(credit_card_data$default.payment.next.month) <- c("No" , "Yes")

# Converting repayment status variables to factors
credit_card_data$PAY_0 <- as.factor(credit_card_data$PAY_0)
credit_card_data$PAY_2 <- as.factor(credit_card_data$PAY_2)
credit_card_data$PAY_3 <- as.factor(credit_card_data$PAY_3)
credit_card_data$PAY_4 <- as.factor(credit_card_data$PAY_4)
credit_card_data$PAY_5 <- as.factor(credit_card_data$PAY_5)
credit_card_data$PAY_6 <- as.factor(credit_card_data$PAY_6)


## Data Exploration and Feauture Selection
factor_var_data <- credit_card_data %>% 
  Filter(f = is.factor)

numeric_var_data <- credit_card_data %>% 
  Filter(f = is.numeric)

names(numeric_var_data)

# Creating histograms for numeric variables LIMIT_BAL and AGE

# LIMIT_BAL
create_histogram(credit_card_data, credit_card_data$LIMIT_BAL, "LIMIT_BAL", 20)

# The distribution is slightly skewed to the right, indicating that higher limit was given to 
# less people, which makes sense

# AGE
create_histogram(credit_card_data, credit_card_data$AGE, "AGE", 60)

# Age is pretty much normally distributed, without much people below 20, which makes sense
# as credit card is not given to people below 18
# The mean and median age of credit card customers is about 35 years
# No. of customers peaks in the range of 27 - 31 years

# Creating bar plots for factor variables SEX, MARRIAGE, EDUCATION and DEFAULT PAYMENT

# SEX
create_bar_plot(credit_card_data, credit_card_data$SEX, "SEX")

# 60 % of the customers are females compared to 40% males

# EDUCATION
create_bar_plot(credit_card_data, credit_card_data$EDUCATION, "EDUCATION")

# We can see that about 47% customers have attended university, 
# and about 35% have attended graduate school

# MARRIAGE
create_bar_plot(credit_card_data, credit_card_data$MARRIAGE, "MARRIAGE")

# There are more customers who are Single as opposed those who are Married

# DEFAULT PAYMENT NEXT MONTH
create_bar_plot(credit_card_data, credit_card_data$default.payment.next.month, "DEFAULT PAYMENT NEXT MONTH")

# About 21% of the customers defaulted


# Comparing DEFAULT PAYMENT NEXT MONTH with respect to SEX, MARRIAGE and EDUCATION

# SEX Vs DEFAULT PAYMENT NEXT MONTH 
create_stacked_bar_plot(credit_card_data, credit_card_data$SEX, "SEX", credit_card_data$default.payment.next.month, "DEFAULT PAYMENT NEXT MONTH")

# We see that 25% of male customers defaulted, compared to 20% female customers 

# EDUCATION Vs DEFAULT PAYMENT NEXT MONTH
create_stacked_bar_plot(credit_card_data, credit_card_data$EDUCATION, "EDUCATION", credit_card_data$default.payment.next.month, "DEFAULT PAYMENT NEXT MONTH")

# We see that about 20 % of those who attended Graduate School Defaulted
# About 22% of University customers defaulted and about 25 % of customers who just attended high school defaulted

# MARRIAGE Vs DEFAULT PAYMENT NEXT MONTH
create_stacked_bar_plot(credit_card_data, credit_card_data$MARRIAGE, "MARRIAGE", credit_card_data$default.payment.next.month, "DEFAULT PAYMENT NEXT MONTH")

# About 24 % of married customers defaulted as opposed to 20 % single customers


# Creating Feature Density Plots for 
# LIMIT_BAL, AGE, BAL_AMTs and PAY_AMTs to see which features have discriminating power

# Plot ScatterPlot of features to determine if there is any pattern overlayed Density Plots
# Interpretation: Density plots for default payment Yes and No almost overlap for Limit_bal, clear non-overlapping cases have more discriminating power

plot_type <- "density"

create_feature_plot(numeric_var_data, credit_card_data$default.payment.next.month, plot_type)

# LIMIT_BAL
# We can see that the density plots of defaulters and non-defaulters is fairly different 
# with respect to balance limit 

# AGE
# We can see that the density plots of defaulters and non-defaulters is slightly different 
# with respect to balance limit 

# BILL AMTs
# Through visual inspection we can infer that balance amounts do not have differentiating power

# PAY AMTs
# Through visual inspection we can infer that payment amounts do not have differentiating power


## Outlier Detection

# Creating box plots to do visual inspection about outliers
plot_type <- "box"

create_feature_plot(numeric_var_data, credit_card_data$default.payment.next.month, plot_type)


## Feature Selection

# Determining association between the target variable and the other categorical variables
# using Chi-squared Test for Independence

lapply(factor_var_data, function(x) chi.squared(x~., factor_var_data))

# We can see that demographic details like SEX, EDUCATION and MARRIAGE are not very good variables
# for differentiating between defaulters and non-defaulters
# Previous payment statuses are well correlated with the target variable, with the most recent one
# being the most correlated

credit_card_data$SEX <- NULL
credit_card_data$EDUCATION <- NULL
credit_card_data$MARRIAGE <- NULL

# Also we can see that PAY_0 is strongly correlated with PAY_2 and PAY_3
# PAY_2 is strongly correlated with PAY_3
# PAY_3 is strongly correlated with PAY_4 and PAY_5
# PAY_4 is strongly correlated with PAY_5 and PAY_6
# PAY_5 is strongly correlated with PAY_3 and PAY_4 and PAY_6
# PAY_6 is strongly correlated with PAY_4 and PAY_5

# Thus, in effect all payment statuses from PAY_0 to PAY_6 are strongly correlated with each other
# To avoid problems of multicollinearity, we will just retain the most recent status PAY_0

credit_card_data$PAY_2 <- NULL
credit_card_data$PAY_3 <- NULL
credit_card_data$PAY_4 <- NULL
credit_card_data$PAY_5 <- NULL
credit_card_data$PAY_6 <- NULL

# Determining correlation between numerical variables
corr_matrix <- cor(numeric_var_data)
round(corr_matrix, 2)

# We can see that all Billed Amounts variables i.e. BILL_AMT1 through BILL_AMT6 
# are highly correlated to each other
# Therefore, to avoid multi-collinearity problem, we will only retain BILL_AMT1
# and get rid of the other BilledAmount variables

credit_card_data$BILL_AMT2 <- NULL
credit_card_data$BILL_AMT3 <- NULL
credit_card_data$BILL_AMT4 <- NULL
credit_card_data$BILL_AMT5 <- NULL
credit_card_data$BILL_AMT6 <- NULL

# Automatic Feature Selection using Mean Decrease in Gini
resampled_balanced_credit_data <- SMOTE(default.payment.next.month ~ ., credit_card_data, perc.over = 100, perc.under = 200)
fs_model <- randomForest(default.payment.next.month~., data=resampled_balanced_credit_data, importance=TRUE)

imp <- as.data.frame(randomForest::importance(fs_model))
imp <- data.frame(MeanDecreaseGini = imp$MeanDecreaseGini,
                  names   = rownames(imp))

imp[order(imp$MeanDecreaseGini,decreasing = T),]

randomForest::varImpPlot(fs_model)

# The variable whose removal from the model leads most decrease in the GINI value is the most
# differentiating and thus the most important variable

# Now we will select only the variables which lead to a substantial decrease in GINI value relative
# to the other variables

# We choose 
# PAY_0 - Repayment status in the last payment cycle
# BILL_AMT1 - Amount of bill statement in the last payment cycle (NT dollar)
# AGE - Age of the debtor
# PAY_AMT1 - Amount paid in the last payment cycle
# PAY_AMT2 - Amount paid in the 2nd last payment cycle
# LIMIT_BAL - Amount of given credit in NT dollars
# PAY_AMT3 - Amount paid in the 3rd last payment cycle
# PAY_AMT6 - Amount paid in the 6th last payment cycle
# PAY_AMT5 - Amount paid in the 5th last payment cycle
# PAY_AMT4 - Amount paid in the 4th last payment cycle

# The variables selected by the algorithm also make business sense, as payment behavior
# in recent past along with the credit limit and age of the debtor should give a good
# idea about likelihood of default

features_to_select <- c("PAY_0", "BILL_AMT1", "AGE", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "LIMIT_BAL","default.payment.next.month")
#features_to_select <- names(credit_card_data)
  
## Data Partitioning: Creating training, validation and test data sets
set.seed(430)
trn_idx <- createDataPartition(credit_card_data$default.payment.next.month, p = 0.8, list = FALSE)
X_train <- credit_card_data[trn_idx, which(names(credit_card_data) %in% features_to_select)]
X_valid_test <- credit_card_data[-trn_idx, which(names(credit_card_data) %in% features_to_select)]

valid_idx <- createDataPartition(X_valid_test$default.payment.next.month, p = 0.5, list = FALSE)
X_valid <- X_valid_test[valid_idx, ]
X_test <- X_valid_test[-valid_idx, ]

# Since the data set is unbalanced, with Yes: No being 1:4
# This can be an issue as the 'No' class will dominate the outcome of classification
# Therefore we will create a new data set using SMOTE such that we get
# a balanced data set by oversampling 'Yes' observations and undersampling 'No' observations

prop.table(table(X_train$default.payment.next.month))

X_train <- SMOTE(default.payment.next.month ~ ., X_train, perc.over = 100, perc.under = 200)

prop.table(table(X_train$default.payment.next.month))

## Modeling

# Defining a 10-folds cross-validation scheme
folds <- 10
cvIndex <- createFolds(factor(X_train$default.payment.next.month), folds, returnTrain = T)
control <- trainControl(index = cvIndex, method = "cv", number = folds, search = "random")


# Logistic Regression

# Training the Model
logit_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "glm"
)

summary(logit_model)

# Model Evaluation

# Kolmogorov-Smirnov (KS) Chart and Statistic
#ks_stat(as.numeric(X_train$default.payment.next.month)-1, as.numeric(predict(logit_model, newdata=X_train))-1, returnKSTable = T)
#ks_plot(as.numeric(X_train$default.payment.next.month)-1, as.numeric(predict(logit_model, newdata=X_train))-1)


KS(predict(logit_model, newdata=X_train), X_train$default.payment.next.month)

# We see that we get the best separation between the 2 classes when threshold is 0.49

threshold <- 0.39

# Training
logit_train_probs <- predict(logit_model, newdata=X_train, type = 'prob')
logit_train_class <- as.factor(ifelse(logit_train_probs[, "Yes"] > threshold, "Yes", "No"))
caret::confusionMatrix(data = logit_train_class, reference = X_train$default.payment.next.month, positive = "Yes")

r <- roc(X_train$default.payment.next.month, logit_train_probs[, "Yes"])
plot(r)
auc(r)

# Validation
logit_valid_probs <- predict(logit_model, newdata=X_valid, type = 'prob')
logit_valid_class <- as.factor(ifelse(logit_valid_probs[, "Yes"] > threshold, "Yes", "No"))
caret::confusionMatrix(data = logit_valid_class, reference = X_valid$default.payment.next.month, positive = "Yes")

r <- roc(X_valid$default.payment.next.month, logit_valid_probs[, "Yes"])
plot(r)
auc(r)

# Test
logit_test_probs <- predict(logit_model, newdata=X_test, type = 'prob')
logit_test_class <- as.factor(ifelse(logit_test_probs[, "Yes"] > threshold, "Yes", "No"))
caret::confusionMatrix(data = logit_test_class, reference = X_test$default.payment.next.month, positive = "Yes")

r <- roc(X_test$default.payment.next.month, logit_test_probs[, "Yes"])
plot(r)
auc(r)


# XGBoost Classifier

# Training the Model
xgb_model <- train(
                  form = default.payment.next.month ~ .,
                  data = X_train,
                  trControl = control,
                  method = "xgbTree"
                )

# Plotting the variations due to differnt parameter values
plot(xgb_model)

# Model with the best parameter values
xgb_model$bestTune

# Final Model with the best parameters
final_grid <- expand.grid(
                nrounds = 124,
                eta = 0.55,
                max_depth = 6, 
                gamma = 3.96,
                colsample_bytree = 0.45,
                min_child_weight = 14,
                subsample = 0.58
              )

final_xgb_model <- train(
                      form = default.payment.next.month ~ .,
                      data = X_train,
                      trControl = control,
                      method = "xgbTree",
                      tuneGrid = final_grid
                    )

# Determining threshold based on KS statistic
KS(predict(final_xgb_model, newdata=X_train), X_train$default.payment.next.month)

# We see that we get the best separation between the 2 classes when threshold is 0.49
thresh_hold <- 0.3

# Model Evaluation

# Training
xgb_train_probs <- predict(final_xgb_model, newdata=X_train, type = 'prob')
xgb_train_class <- as.factor(ifelse(xgb_train_probs[, "Yes"] > thresh_hold, "Yes", "No"))
caret::confusionMatrix(data = xgb_train_class, reference = X_train$default.payment.next.month, positive = "Yes")

r <- roc(X_train$default.payment.next.month, xgb_train_probs[, "Yes"])
plot(r)
auc(r)

# Validation

ks_stat(as.numeric(X_valid$default.payment.next.month)-1, as.numeric(predict(logit_model, newdata=X_train))-1, returnKSTable = T)
ks_plot(as.numeric(X_valid$default.payment.next.month)-1, as.numeric(predict(logit_model, newdata=X_train))-1)

# We see that we get the best separation between the 2 classes when threshold is 0.49
thresh_hold <- 0.31

xgb_valid_probs <- predict(final_xgb_model, newdata=X_valid, type = 'prob')
xgb_valid_class <- as.factor(ifelse(xgb_valid_probs[, "Yes"] > thresh_hold, "Yes", "No"))
caret::confusionMatrix(data = xgb_valid_class, reference = X_valid$default.payment.next.month, positive = "Yes")

r <- roc(X_valid$default.payment.next.month, xgb_valid_probs[, "Yes"])
plot(r)
auc(r)

# Test
xgb_test_probs <- predict(final_xgb_model, newdata=X_test, type = 'prob')
xgb_test_class <- as.factor(ifelse(xgb_test_probs[, "Yes"] > thresh_hold, "Yes", "No"))
caret::confusionMatrix(data = xgb_test_class, reference = X_test$default.payment.next.month, positive = "Yes")

r <- roc(X_test$default.payment.next.month, xgb_test_probs[, "Yes"])
plot(r)
auc(r)


# Tree Classifier (CART)

# Training the Model
tree_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "rpartScore"
)

# Plotting the variations due to differnt parameter values
plot(tree_model)

# Model with the best parameter values
tree_model$bestTune

# Final Model with the best parameters
final_grid <- expand.grid(
  cp = 0,
  split = 'abs',
  prune = 'mr' 
)

final_tree_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "rpartScore",
  tuneGrid = final_grid
)


# Model Evaluation

# Training
tree_train_class <- predict(final_tree_model, newdata=X_train, type="raw")
confusionMatrix(data = tree_train_class, reference = X_train$default.payment.next.month, positive = "Yes")


# Validation
tree_valid_class <- predict(final_tree_model, newdata=X_valid, type="raw")
confusionMatrix(data = tree_valid_class, reference = X_valid$default.payment.next.month, positive = "Yes")

# Test
tree_test_class <- predict(final_tree_model, newdata=X_test, type="raw")
confusionMatrix(data = tree_test_class, reference = X_test$default.payment.next.month, positive = "Yes")


# Naive Bayes Classifier

# Training the Model
nb_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "naive_bayes"
)

# Plotting the variations due to differnt parameter values
plot(nb_model)

# Model with the best parameter values
nb_model$bestTune

# Final Model with the best parameters
final_grid <- expand.grid(
  laplace = 0,
  usekernel = FALSE,
  adjust = 1 
)

final_nb_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "naive_bayes",
  tuneGrid = final_grid
)

# Determining threshold based on KS statistic
KS(predict(final_nb_model, newdata=X_train), X_train$default.payment.next.month)

# We see that we get the best separation between the 2 classes when threshold is 0.14

thresh_hold <- 0.14

# Model Evaluation

# Training
nb_train_probs <- predict(final_nb_model, newdata=X_train, type = 'prob')
nb_train_class <- as.factor(ifelse(nb_train_probs[, "Yes"] > thresh_hold, "Yes", "No"))
confusionMatrix(data = nb_train_class, reference = X_train$default.payment.next.month, positive = "Yes")

r <- roc(X_train$default.payment.next.month, nb_train_probs[, "Yes"])
plot(r)
auc(r)

# Validation
nb_valid_probs <- predict(final_nb_model, newdata=X_valid, type = 'prob')
nb_valid_class <- as.factor(ifelse(nb_valid_probs[, "Yes"] > thresh_hold, "Yes", "No"))
confusionMatrix(data = nb_valid_class, reference = X_valid$default.payment.next.month, positive = "Yes")

r <- roc(X_valid$default.payment.next.month, nb_valid_probs[, "Yes"])
plot(r)
auc(r)

# Test
nb_test_probs <- predict(final_nb_model, newdata=X_test, type = 'prob')
nb_test_class <- as.factor(ifelse(nb_test_probs[, "Yes"] > thresh_hold, "Yes", "No"))
confusionMatrix(data = nb_test_class, reference = X_test$default.payment.next.month, positive = "Yes")

r <- roc(X_test$default.payment.next.month, nb_test_probs[, "Yes"])
plot(r)
auc(r)


# Random Forest Classifier

# Training the Model
rf_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "RRF"
)

# Plotting the variations due to differnt parameter values
plot(rf_model)

# Model with the best parameter values
rf_model$bestTune

# Final Model with the best parameters
final_grid <- expand.grid(
  mtry = 3,
  coefReg = 0.4,
  coefImp = 0.67
)

final_rf_model <- train(
  form = default.payment.next.month ~ .,
  data = X_train,
  trControl = control,
  method = "RRF",
  tuneGrid = final_grid
)

# Setting threshold for positive class (Defaulter)
thresh_hold <- 0.25

# Model Evaluation

# Training
rf_train_probs <- predict(final_rf_model, newdata=X_train, type = 'prob')
rf_train_class <- as.factor(ifelse(rf_train_probs[, "Yes"] > thresh_hold, "Yes", "No"))
confusionMatrix(data = rf_train_class, reference = X_train$default.payment.next.month, positive = "Yes")

r <- roc(X_train$default.payment.next.month, rf_train_probs[, "Yes"])
plot(r)
auc(r)

# Validation
rf_valid_probs <- predict(final_rf_model, newdata=X_valid, type = 'prob')
rf_valid_class <- as.factor(ifelse(rf_valid_probs[, "Yes"] > thresh_hold, "Yes", "No"))
confusionMatrix(data = rf_valid_class, reference = X_valid$default.payment.next.month, positive = "Yes")

r <- roc(X_valid$default.payment.next.month, rf_valid_probs[, "Yes"])
plot(r)
auc(r)

# Test
nb_test_probs <- predict(final_nb_model, newdata=X_test, type = 'prob')
nb_test_class <- as.factor(ifelse(nb_test_probs[, "Yes"] > thresh_hold, "Yes", "No"))
confusionMatrix(data = nb_test_class, reference = X_test$default.payment.next.month, positive = "Yes")

r <- roc(X_test$default.payment.next.month, nb_test_probs[, "Yes"])
plot(r)
auc(r)


