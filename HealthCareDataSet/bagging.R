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
glimpse(data)

#finding about data and cleaning it
###dealing with NA data
dat.cleaned <- na.omit(data)

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

## splitting training and test datasets to x and y
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


### Label Encoding
y_ubtrain <-  factor(y_ubtrain,levels=c('Normal',"Malicious"),labels=c(1,0))
y_test <-  factor(y_test,levels=c('Normal',"Malicious"),labels=c(1,0))
y_btrain <-  factor(y_btrain,levels=c('Normal',"Malicious"),labels=c(1,0))



# Hyper parameter tuning

## Cross validation
set.seed(10579306)

ctrl <- trainControl(
  method = "repeatedcv",
  number = 7,
  repeats = 2
)

## param grid
### Specify values to tune
nbagg_values <- c(50, 70, 150)
cp_values <- c(0.01, 0.05, 0.1)
minsplit_values <- c(2, 5, 10)

### intializing lists
nbaggs<-list()
cps<-list()
minsplits<-list()

### Train models for each combination of nbagg, cp, and minsplit
models <- list()
for (nbagg in nbagg_values) {
  nbaggs<-append(nbaggs,list(nbagg))
  for (cp in cp_values) {
    cps<-append(cps,list(cp))
    for (minsplit in minsplit_values) {
      minsplits<-append(minsplits,list(minsplit))
      #### Train the model
      set.seed(10579306)
      model <- train(
        Classification ~ .,
        data = mydata.ub.train,
        method = "treebag",
        trControl = ctrl,
        nbagg = nbagg,
        cp = cp,
        minsplit = minsplit,
      )
      ### Store the model
      models <- append(models, list(model))
    }
  }
}

model

# comparing the results
results <- resamples(models)
summary(results)
dotplot(results)

# After analyzing the summary we can see the best values for nbagg,cp and minsplit 
## are 50, 0.01 and 2 respectively

##Label Encoding
mydata.ub.train$Classification <- factor(mydata.ub.train$Classification,levels=c('Normal',"Malicious"),labels=c(1,0))
mydata.test$Classification <- factor(mydata.test$Classification,levels=c('Normal',"Malicious"),labels=c(1,0))


# fitting the model for unbalanced dataset 
ub_bagging_model <- bagging(
  Classification~.,
  data=mydata.ub.train,
  nbagg = 50,    
  coob = TRUE, 
  control = rpart.control(minsplit = 2, cp = 0.01,  
                          min_depth=2) 
)

dim(mydata.test)

x_test <- mydata.test[,-c(14)]
dim(x_test)

## checking prediction
y1_ub_pred <- predict(ub_bagging_model,x_test)
y1_ub_pred$class


## evaluation using confusion matrix
cm2 = confusionMatrix(table(y1_ub_pred$class,y_test))
cm2

cm2$byClass
cm2$overall

#bagging for balanced dataset
# Train models for each combination of nbagg, cp, and minsplit
models1 <- list()
for (nbagg in nbagg_values) {
 
  for (cp in cp_values) {
   
    for (minsplit in minsplit_values) {
      
      # Train the model
      set.seed(10579306)
      model <- train(
        Classification ~ .,
        data = mydata.b.train,
        method = "treebag",
        trControl = ctrl,
        nbagg = nbagg,
        cp = cp,
        minsplit = minsplit,
      )
      # Store the model
      models1 <- append(models, list(model,nbagg,cp,minsplit))
    }
  }
}

models1

# comparing the results
results <- resamples(models)
summary(results)
dotplot(results)

# After analyzing the summary we can see the best values for nbagg,cp and minsplit 
## are 50, 0.01 and 2 respectively

##Label Encoding
mydata.b.train$Classification <- factor(mydata.b.train$Classification,levels=c('Normal',"Malicious"),labels=c(1,0))


## fitting the model for balanced dataset
b_bagging_model <- bagging(
  Classification~.,
  data=mydata.b.train,
  nbagg = 150,    
  coob = TRUE, 
  control = rpart.control(minsplit = 10, cp = 0.1,  
                          min_depth=2) 
)


class(x_test)

## making predictions
y1_b_pred <- predict(b_bagging_model,x_test)
y1_b_pred$class

class(y1_b_pred)
print(y1_b_pred$class)

class(y_test)

## evaluation using confusion matrix
cm3 = confusionMatrix(factor(y1_b_pred$class),factor(y_test))
cm3

cm3$byClass
cm3$overall

colSums(is.na(dat.cleaned))
