#importing libraries
library(tidyverse)
library(mice)
library(plotrix)
library(stringr)
library(tm)
library(stopwords)
library(superml)
library(party)
library(forcats)
library(dplyr)
library(data.table) 
library(glmnet)
library(e1071)
library(caret)
library(adabag)
library(randomForest)
library(xgboost)

#loading the data
getwd()
setwd('C:\\Users\\Abrar Syed\\Documents\\R programming\\HealthCare_classification\\HealthCareDataSet')
data <- read.csv('HealthCareData_2024.csv')

#viewing the data
#view(data)
glimpse(data)

#finding about data and cleaning it
###dealing with NA data
dat.cleaned <- na.omit(data)
#view(dat.cleaned)

###dealing with outliers
boxplot(dat.cleaned$ResponseTime)
dat.cleaned$ResponseTime[dat.cleaned$ResponseTime==99999.00]<-median(dat.cleaned$ResponseTime,na.rm=TRUE)

###Merging regular and unknown networkTypes in one class called Others
dat.cleaned$NetworkInteractionType <- fct_collapse(dat.cleaned$NetworkInteractionType, other = c("Unknown", "Regular"))

#visualizing data
barplot(table(dat.cleaned$NetworkInteractionType),xlab='Categories',ylab='frequencey',main='Network Interactions',
        col=c('red','green','blue','black','purple' ))

barplot(table(dat.cleaned$ AlertCategory),xlab='Categories',ylab='frequencey',main='Alert Messages',
        col=c('red','green','blue'))

pie(table(dat.cleaned$NetworkEventType),labels=c('Policy_Violation','NormalOperation','ThreatDetected'),main='Event Distribution',
    col=c('red','green','blue'))


# LABEL ENCODING
dat.cleaned$AlertCategory <-  as.integer(factor(dat.cleaned$AlertCategory))
dat.cleaned$NetworkEventType <-  as.integer(factor(dat.cleaned$NetworkEventType))
dat.cleaned$NetworkInteractionType <-  as.integer(factor(dat.cleaned$NetworkInteractionType))
dat.cleaned$SessionIntegrityCheck <-  as.integer(factor(dat.cleaned$SessionIntegrityCheck))
dat.cleaned$ResourceUtilizationFlag <-  as.integer(factor(dat.cleaned$ResourceUtilizationFlag))

#TRAIN/TEST Split

### Separate samples of normal and malicious events 
dat.class0 <- dat.cleaned %>% filter(Classification == "Normal") # normal 
dat.class1 <- dat.cleaned %>% filter(Classification == "Malicious") # malicious

### Randomly selecting 9600 non-malicious and 400 malicious samples using my student ID
set.seed(10579306) 
rows.train0 <- sample(1:nrow(dat.class0), size = 9600, replace = FALSE) 
rows.train1 <- sample(1:nrow(dat.class1), size = 400, replace = FALSE) 

### 10000 ‘unbalanced’ training samples 
train.class0 <- dat.class0[rows.train0,] # Non-malicious samples 
train.class1 <- dat.class1[rows.train1,] # Malicious samples 
mydata.ub.train <- rbind(train.class0, train.class1) 

### 19200 ‘balanced’ training samples, i.e. 9600 normal and malicious samples each. 
set.seed(10579306)
train.class1_2 <- train.class1[sample(1:nrow(train.class1), size = 9600,  
                                      replace = TRUE),] 
mydata.b.train <- rbind(train.class0, train.class1_2) 

### testing samples 
test.class0 <- dat.class0[-rows.train0,] 
test.class1 <- dat.class1[-rows.train1,] 
mydata.test <- rbind(test.class0, test.class1) 

#Selecting_Model
set.seed(10579306) 
models.list1 <- c("Logistic Ridge Regression", 
                  "Logistic LASSO Regression", 
                  "Logistic Elastic-Net Regression") 
models.list2 <- c("Classification Tree", 
                  "Bagging Tree", 
                  "Random Forest") 
myModels <- c(sample(models.list1, size = 1), 
              sample(models.list2, size = 1)) 
myModels %>% data.frame 


#USING ML CLASSIFICATION MODELS

## splitting training and test dataets to x and y

### shuffling the data
set.seed(10579306)
mydata.ub.train= mydata.ub.train[sample(1:nrow(mydata.ub.train)), ] 
mydata.b.train= mydata.b.train[sample(1:nrow(mydata.b.train)), ] 
mydata.test=mydata.test[sample(1:nrow(mydata.test)),]

### unbalanced train set
y_ubtrain <- as.factor(mydata.ub.train[,14])
x_ubtrain <- as.matrix(mydata.ub.train[,1:13])

### balanced train set
y_btrain <- as.factor(mydata.b.train[,14])
x_btrain <- as.matrix(mydata.b.train[,1:13])

### test set
y_test <- as.factor(mydata.test[,14])
x_test <- as.matrix(mydata.test[,1:13])

### Label Encoding
y_ubtrain <-  factor(y_ubtrain,levels=c('Normal',"Malicious"),labels=c(1,0))
y_test <-  factor(y_test,levels=c('Normal',"Malicious"),labels=c(1,0))
y_btrain <-  factor(y_btrain,levels=c('Normal',"Malicious"),labels=c(1,0))

### cross-validation
set.seed(10579306)

cont_cv <- trainControl(method = "repeatedcv",number=8, repeats = 2)
Grid_ridge = expand.grid(alpha = 0, lambda = seq(0.001, 0.1,
                                                 by = 0.0002))
### tunning ridge
ridge_unbal_model = train(Classification~.,
                          data=mydata.ub.train,
                          method = "glmnet",
                          trControl = cont_cv,
                          tuneGrid = Grid_ridge,
                          family = "binomial"
)

### Get the Best Model
best_model <- ridge_unbal_model$bestTune
best_model
###best parameter value
best_lambda <- best_model$lambda
best_lambda

## Logistic Ridge Regression for unbalnced dataset
### fitting the model
set.seed(10579306)
best_model <- glmnet(x = x_ubtrain,
                    y = as.matrix(y_ubtrain),
                    lambda = best_lambda, 
                    alpha = 0)

### Checking the prediction
y_ub_pred <- predict(best_model,x_test)
y_ub_pred<-ifelse(y_ub_pred>0.5,1,0)


y_test <- as.matrix(y_test)

### Confusion matrix for rigid regression classifier
### for unbalanced dataset
cm = confusionMatrix(factor(y_ub_pred),factor(y_test))
cm

cm$byClass
cm$overall

### cross-validation
set.seed(10579306)

cont_cv <- trainControl(method = "repeatedcv",number=8, repeats = 2)
Grid_ridge = expand.grid(alpha = 0, lambda = seq(0.001, 0.1,
                                                 by = 0.0002))
### tunning ridge
ridge_unbal_model = train(Classification~.,
                          data=mydata.b.train,
                          method = "glmnet",
                          trControl = cont_cv,
                          tuneGrid = Grid_ridge,
                          family = "binomial"
)

### Get the Best Model
best_model <- ridge_unbal_model$bestTune
best_model
###best parameter value
best_lambda <- best_model$lambda
best_lambda

## Logistic Ridge Regression for unbalnced dataset
### fitting the model
set.seed(10579306)
best_model <- glmnet(x = x_btrain,
                     y = as.matrix(y_btrain),
                     lambda = best_lambda, 
                     alpha = 0)

### Checking the prediction
y_b_pred <- predict(best_model,x_test)
y_b_pred<-ifelse(y_b_pred>0.5,1,0)


### Confusion matrix for rigid regression classifier
### for balanced dataset

cm1 = confusionMatrix(factor(y_b_pred),factor(y_test))
cm1

cm1$byClass
cm1$overall
