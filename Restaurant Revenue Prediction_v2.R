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
##Combine Train & Test dataset for more robust testing
#------------------------------------------------------------------------------
test$revenue<-0 # created new field in test, loading 'NA' in test sample for future use in cross validation
total <- rbind(train,test)

#------------------------------------------------------------------------------
# Explore Total data
#------------------------------------------------------------------------------
head(total)
str(total)       
summary(total)

#Checking Missing Data in training dataset
md.pattern(total) # Train data has no missing


#------------------------------------------------------------------------------
##Visualize dataset
#------------------------------------------------------------------------------

#We know that the toal dataset contains only revenue for our Train set.
#So we are only going to visualize the train set which has real revenue values

#GGPLOT Revenue Histogram of Train Dataset

#Determine min and max for setting histogram bins
min(train$revenue)
max(train$revenue)

#Setting bin width to 2,000,000
ggplot(train, aes(x = revenue, fill = ..count..)) +
  geom_histogram(binwidth = 1000000) +
  ggtitle("Figure 1. Histogram of Restaurant Revenue") +
  ylab("Count of Restaurants") +
  xlab("Revenue") + 
  theme(plot.title = element_text(hjust = 0.5))

# We have 3 extreme outliers
# There are outlier restaurants earning greater than 10,000,000
# Suggest to exclude restaurant sales greater than 12,000,000 to remove skew, will check with and w/out lasso/ridge

#Take log of revenue
options(scipen=10000) #remove scientific notation
ggplot(train, aes(x = log(revenue), fill = ..count..)) +
  geom_histogram(binwidth = 1) +
  ggtitle("Figure 2. Histogram of Restaurant Revenue") +
  ylab("Count of Restaurants") +
  xlab("Revenue") + 
  theme(plot.title = element_text(hjust = 0.5))

#Perform boxplot to confirm outliers
#Plot and observe density and distributions
revd <-density(train$revenue)
plot(revd,col="blue")
boxplot(train$revenue) # 3 outliers

#Scatter Plot
dev.off() # fixing call graphics error
options(scipen=10000) #remove scientific notation
pscatter<-ggplot(train, aes(y=train$City,x=train$revenue)) +geom_point() + ylab("City") +  xlab("Revenue")
pscatter # City vs. Revenue

# With visualization 3 outliers are confirmed confirmed
# In the data overview, it is mentioned that within the obscured data, 
# there are location data, hence city desciption could be removed and will be explored as 
# it could be possible we do not need to generate dummy data for city factor data


#------------------------------------------------------------------------------
## DATA WRANGLE & TRANSFORMATION & Feature Engineering on Total Dataset
#------------------------------------------------------------------------------

#Date Transformations peformed on train, test, total dataset
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

total1f<-(total1[,c(1,6:43)]) # Using only Px values and Revenue
#total1 is will be used for Lasso & ridge

# Create dummy variables for categorical fields for train, test, total datasets
train1d <- fastDummies::dummy_cols(train1)
test1d <- fastDummies::dummy_cols(test1)
total1d <- fastDummies::dummy_cols(total1)

#dummies create over 100 variables, of which only a handful are useful as predictors

# Check correlation of Px variables against Revenue Train dataset as it contatins revenue
M <- cor(train[,c(6:43)])
corrplot(M, method = "ellipse",order = "hclust")

# Subsetting Data for use in LM
train_px<-(train[,c(1,6:43)]) # Using only Px values and Revenue

# Check correlation of Px variables against Revenue DUMMY DATA ENRICHMENT & FIELD ENGINEERING
M <- cor(train1d[,c(6:83)])
corrplot(M, method = "ellipse",order = "hclust")

# Subsetting Data for use in LM
train1d_px<-(train1d[,c(1,6:83)]) # Using only Px values and Revenue

#############################################################################
# CONTROL OUTLIERS: KEEP OR REMOVE OUTLIER by removing hash #, 
# removing will filter out, keeping will leave in
#Remove noted outliers above with sales greater than 12,000,000 ID 16,75,99

#train1d_px <- sqldf('select *
#                         from train_px   
#                            where revenue < 12000000')

#############################################################################

#------------------------------------------------------------------------------
#LOG-LOG MODEL Log Revenue
#------------------------------------------------------------------------------
fit.log<-lm(log(revenue)~ . -revenue, data=train1d_px) #added "log()" for revenue
summary(fit.log)
par(mfrow=c(1,4)) # this command sets the plot window to show 1 row of 4 plots
plot(fit.log) #diagnostic plots for the "fit" model -- will discuss them in class

plot(density(resid(fit.log)))

####
##Create a matrix data
###


###############################################
### Regularizations (LASSO and ridge)
###############################################

xtrain<-train1 #No Dummies version control of dataset used for
#xtrain<-train1d #Dummies version control of dataset used for

####
# create the y variable and matrix (capital X) of x variables (will make the code below easier to read + will ensure that all interactoins exist)
# Build the matrix
####
y<-log(train1$revenue)
X<-model.matrix(Id~ ., train1)[,-1]
X<-cbind(total1f$Id,X) #Id is inserted as V1....?
X

#--------------------------------------------------------------------------------------------
# split X into testing, trainig/holdout and prediction as before
X.training<-subset(X,X[,1]<=136)
X.testing<-subset(X, (X[,1]>=137 & X[,1]<=137))
X.prediction<-subset(X,X[,1]>=138)

####
## Split Training data with dummies into 70/30 for further training & testing 
####

#a <- sample.int(n=nrow(train1d_px2), size=floor(.7*nrow(train1d_px2)), replace=F)
#a.train1d_px2.train <- train1d_px2[a,] #used again for splitting in b
#a.train1d_px2.prediction <- train1d_px2[-a,] #holdout a

#Further split the train into train and test
#b <- sample.int(n=nrow(a.train1d_px2.train), size=floor(.7*nrow(a.train1d_px2.train)), replace=F)
#b.train1d_px2.train.train <- a.train1d_px2.train[b,]
#b.train1d_px2.train.test  <- a.train1d_px2.train[-b,]

#--------------------------------------------------------------------------------------------
## LASSO (alpha=1)
#--------------------------------------------------------------------------------------------
lasso.fit<-glmnet(x = X.training, y = y, alpha = 1)
plot(lasso.fit, xvar = "lambda")

## Use Cross Validation to determine optimal penalty value - lambda

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.training, y = y, alpha = 1) #create cross-validation data
plot(crossval)
penalty.lasso <- crossval$lambda.min #determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph
plot(crossval,xlim=c(-8.5,-6),ylim=c(0.006,0.008)) # lets zoom-in
lasso.opt.fit <-glmnet(x = X.training, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lasso.opt.fit) #resultant model coefficients

# predicting the performance on the testing set
lasso.testing <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X.testing))
mean(abs(lasso.testing-test1$revenue)/test1$revenue*100) #calculate and display MAPE

#------------------------------------------------------------------------------
## RIDGE
#------------------------------------------------------------------------------

#ridge (alpha=0)
ridge.fit<-glmnet(x = X.training, y = y, alpha = 0)
plot(ridge.fit, xvar = "lambda")

## Use Cross Validation to determine optimal penalty value - lambda

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = X.training, y = y, alpha = 0)
plot(crossval)
penalty.ridge <- crossval$lambda.min 
log(penalty.ridge) 
ridge.opt.fit <-glmnet(x = X.training, y = y, alpha = 0, lambda = penalty.ridge) #estimate the model with that
coef(ridge.opt.fit)


ridge.testing <- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =X.testing))
mean(abs(ridge.testing-test1$revenue)/test1$revenue*100) 

#Errors

#------------------------------------------------------------------------------
## Resolving to use other model - RANDOM FOREST on Transformed Dataset
#------------------------------------------------------------------------------
# Suggested by other competitors

?randomForest
fit<-randomForest(train1d_px$revenue~.,data=train1d_px2)
summary(fit)

y_pred<-predict(fit,test)
test2 = read.csv("C:\\Temp\\restaurant-revenue-prediction\\test.csv", sep = ',')
Id<-test2$Id
Prediction<-y_pred
out<-data.frame(Id,Prediction)
write.csv(out,"out2.csv",row.names=F)



