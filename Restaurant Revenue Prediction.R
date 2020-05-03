#------------------------------------------------------------------------------
##
## MMA867 Individual Assignment 1 - David Poon
## Purpose: Predicting Restaurant Revenue, by using regression, lasso and ridge
## Kaggle Data: https://www.kaggle.com/c/restaurant-revenue-prediction
##
#------------------------------------------------------------------------------

## Load these libraries
library(fastDummies)
library(glmnet)
library(tidyverse)
library(dplyr)
library(mice)
library(markdown)
library(corrplot)
library(ggplot2)
library(lubridate)
library(rms)
library(sqldf)
#------------------------------------------------------------------------------
## DATA EXPLORATION AND CLEANING
#------------------------------------------------------------------------------

# Reading in train and test data files locatced in directory C:\Temp\
train = read.csv("C:\\Temp\\restaurant-revenue-prediction\\train.csv", sep = ',')
test = read.csv("C:\\Temp\\restaurant-revenue-prediction\\test.csv", sep = ',')

#------------------------------------------------------------------------------
# Explore training data
#------------------------------------------------------------------------------
head(train)
str(train)       
summary(train)

#Train has 1 more feild for revenue

#Checking Missing Data in training dataset
md.pattern(train) # Train data has no missing

#------------------------------------------------------------------------------
##Explore testing data
#------------------------------------------------------------------------------
head(test)
str(test)
summary(test)

#Checking Missing Data in test dataset
md.pattern(test) # Testing data has no missing

#------------------------------------------------------------------------------
##Combine Train & Test dataset
#------------------------------------------------------------------------------
test$revenue<-0 # created new field in test, loading 'NA' in test sample for future use in cross validation
total <- rbind(train,test)

#------------------------------------------------------------------------------
##Visualize dataset
#------------------------------------------------------------------------------

#Determin min and max for setting histogram bins
min(train$revenue)
max(train$revenue)

#GGPLOT Revenue Histogram of Train Dataset
#options(scipen=10000)
ggplot(train, aes(x = revenue, fill = ..count..)) +
  geom_histogram(binwidth = 1000000) +
  ggtitle("Figure 1. Histogram of Restaurant Revenue") +
  ylab("Count of Restaurants") +
  xlab("Revenue") + 
  theme(plot.title = element_text(hjust = 0.5))

# There are outlier restaurants earning greater than 10 Million

#Take log of revenue
#options(scipen=10000)
ggplot(train, aes(x = log(revenue), fill = ..count..)) +
  geom_histogram(binwidth = 1) +
  ggtitle("Figure 2. Histogram of Restaurant Revenue") +
  ylab("Count of Restaurants") +
  xlab("Revenue") + 
  theme(plot.title = element_text(hjust = 0.5))

#Plot and observe density and distributions
revd <-density(train$revenue)
plot(revd,,col="blue")
boxplot(train$revenue) # 3 outliers

#Scatter Plot
dev.off() # fixing call graphics error
pscatter<-ggplot(train, aes(y=train$City,x=train$revenue)) +geom_point() + ylab("City") +  xlab("Revenue")
pscatter # City vs. Revenue

# 3 outliers confirmed
# In the data overview, it is mentioned that within the obscured data, there are location data, hence city desciption could be removed and will be explored


#------------------------------------------------------------------------------
## DATA WRANGLE & TRANSFORMATION & Feature Engineering on Total Dataset
#------------------------------------------------------------------------------
train1<-train #version control on train
train1$Open.Date<-mdy(train$Open.Date) #Convert date factor into date datatype


test1<-test #version control on test file
test1$Open.Date<-mdy(test$Open.Date) #Convert date factor into date datatype

total1<-total #version control total file (test & train)
total1$Open.Date<-mdy(total$Open.Date) #Convert date factor into date datatype

#New Feature - Total years since first store opening in 1996-05-08

train1$YearsSinceFirstStore <- as.numeric(train1$Open.Date-ymd("1996-05-08"), units="days")/365
test1$YearsSinceFirstStore<- as.numeric(test1$Open.Date-ymd("1996-05-08"), units="days")/365
total1$YearsSinceFirstStore<- as.numeric(total1$Open.Date-ymd("1996-05-08"), units="days")/365


# Create dummy variables for categorical fields
train1d <- fastDummies::dummy_cols(train1)
test1d <- fastDummies::dummy_cols(test1)
total1d <- fastDummies::dummy_cols(total1)


# Check correlation of Px variables against Revenue TRAIN DATA RAW
M <- cor(train[,c(6:43)])
corrplot(M, method = "ellipse",order = "hclust")

# Subsetting Data for use in LM
train_px<-(train[,c(1,6:43)]) # Using only Px values and Revenue

#Remove noted outliers above ID 16,75,99
train_px2 <- sqldf('select *
                         from train_px   
                            where Id not in (16,75,99)')


# Check correlation of Px variables against Revenue DUMMY DATA ENRICHMENT & FIELD ENGINEERING
M <- cor(train1d[,c(6:83)])
corrplot(M, method = "ellipse",order = "hclust")

# Subsetting Data for use in LM
train1d_px<-(train1d[,c(1,6:83)]) # Using only Px values and Revenue

#Remove noted outliers above ID 16,75,99
train1d_px2 <- sqldf('select *
                         from train_px   
                            where Id not in (16,75,99)')
#------------------------------------------------------------------------------
## LM BASE REGRESSION MODEL
#------------------------------------------------------------------------------
fit<-lm(revenue~ . -revenue, data=train1d_px2) 
summary(fit)
par(mfrow=c(1,4)) # this command sets the plot window to show 1 row of 4 plots
plot(fit) #diagnostic plots for the "fit" model -- will discuss them in class

plot(density(resid(fit)))


#------------------------------------------------------------------------------
#LOG-LOG MODEL Log Revenue
#------------------------------------------------------------------------------
fit.log<-lm(log(revenue)~ . -revenue, data=train1d_px2) #added "log()" for revenue
summary(fit.log)
par(mfrow=c(1,4)) # this command sets the plot window to show 1 row of 4 plots
plot(fit.log) #diagnostic plots for the "fit" model -- will discuss them in class

plot(density(resid(fit.log)))

#------------------------------------------------------------------------------
#PREDICT
#------------------------------------------------------------------------------

# LM Model
predicted.revenue<-predict(fit, test) #use the "fit" model to predict prices for the prediction data
write.csv(predicted.revenue, file = "C:\\Temp\\restaurant-revenue-prediction\\Predicted Revenue.csv") # export the predicted prices into a CSV file

plot(predicted.revenue ~ test$revenue) 
abline(0,0)

#LOOKS LIKE THE DATA CAN"T BE MODEL USING REGRESSION :(

# LOG LOG Model
predicted.revenue.log<-exp(predict(fit.log, test)) #use the "fit" model to predict prices for the prediction data
write.csv(predicted.revenue.log, file = "C:\\Temp\\restaurant-revenue-prediction\\Log_Predicted Revenue.csv") # export the predicted prices into a CSV file

#Which is better
########

percent.errors <- abs((diamond.data.testing$Price-predicted.prices.testing)/diamond.data.testing$Price)*100 #calculate absolute percentage errors
mean(percent.errors) #display Mean Absolute Percentage Error (MAPE)

rev_percent.errors<- abs((test$revenue-predicted.revenue)/test$revenue)*100 #calculate absolute percentage errors
mean(rev_percent.errors) #display Mean Absolute Percentage Error (MAPE)

rev_percent.errors

#------------------------------------------------------------------------------
## Resolving to use other model - RANDOM FOREST on Transformed Dataset
#------------------------------------------------------------------------------
?randomForest
fit<-randomForest(train1d_px2$revenue~.,data=train1d_px2)
summary(fit)

y_pred<-predict(fit,test)
test2 = read.csv("C:\\Temp\\restaurant-revenue-prediction\\test.csv", sep = ',')
Id<-test2$Id
Prediction<-y_pred
submission<-data.frame(Id,Prediction)
write.csv(submission,"submission2.csv",row.names=F)