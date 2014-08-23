## R program for Data Sciences/Practical Machine Learning Class Project

## load required packages (assumes they have been installed)
require(caret)
require(rpart)
set.seed(951)

## load training data
## if training data has not been downloaded into working directory, use the following:
## file <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
## pml.training <- read.csv(file, stringsAsFactors=FALSE)
pml.training <- read.csv("pml-training.csv", stringsAsFactors=FALSE)

## load test data
## if test data has not been downloaded into working directory, use the following:
## file <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
## pml.test <- read.csv(file, stringsAsFactors=FALSE)
pml.test <- read.csv("pml-testing.csv", stringsAsFactors=FALSE) ## if in working directory

## identify columns in training set without any NAs
keep <- colSums(is.na(pml.training))==0

## remove NA columns
train <- pml.training[,keep]

## identify columns with mostly blank cells
## (there are about 400 rows with data in otherwise blank columns)
keep <- colSums(train=="")<19000

## remove mostly blank columns
train <- train[,keep]

## convert selected text columns to factors
train$user_name <- as.factor(train$user_name)
train$new_window <- as.factor(train$new_window)
train$classe <- as.factor(train$classe)

## remove text version of time to numeric, scale timestamp_part_1
train$cvtd_timestamp <- NULL
z <- min(train$raw_timestamp_part_1)
train$raw_timestamp_part_1 <- train$raw_timestamp_part_1 - z

## remove row number (perfect predictor, since the outcomes are ordered!)
train <- train[,-1]

## reduce test set to same columns as training set
test <- pml.test[,names(train)[1:ncol(train)-1]]  ## no classe in test set

## make variable changes to test to match train
test$user_name <- as.factor(test$user_name)
test$new_window <- as.factor(test$new_window)
test$raw_timestamp_part_1 <- test$raw_timestamp_part_1 - z

## split original training set into training (80%) and cross-valiadation sets (20%)
inTrain <- createDataPartition(y=train$classe,p=.8,list=FALSE)
train1 <- train[inTrain,]
train2 <- train[-inTrain,]

## do analysis of variable variance
nsv <- nearZeroVar(train1, saveMetrics=TRUE)

## show variables with zero or near-zero variance
row.names(nsv)[nsv$zerovar==TRUE]
row.names(nsv)[nsv$nzv==TRUE]

## show variables with high correlation
trainCorr <- cor(train1[,6:57])  ## numeric variables only
highCorr <- findCorrelation(trainCorr, 0.9) + 5
names(train1[highCorr])

## remove no-longer-needed variables
rm(pml.training, pml.test, keep, z, inTrain, nsv, trainCorr, highCorr)

## fit a model to the training subset
model <- rpart(train1$classe ~ ., data=train1[,-(ncol(train1)-1)], method="class")

## display model results
plotcp(model)

## predict using model for training, cross-validation and test sets
p1 <- as.data.frame(predict(model,train1[,1:ncol(train1)-1]))
p2 <- as.data.frame(predict(model,train2[,1:ncol(train2)-1]))
ptest <- as.data.frame(predict(model,test[,1:ncol(test)]))

## function to convert prediction matrix to prediction value (using max)
createClasse <- function (pred){
  pp <- data.frame(classe=NA)
  for (i in 1:nrow(pred)) {
    pp[i,1] <- names(pred)[which(pred[i,]==max(pred[i,]))]
  }
  return(as.matrix(pp))
}

## compare predicted to actual values for training set
actual1_classe <- as.matrix(as.character(train1$classe))
pred1_classe<-createClasse(p1)
confusionMatrix(train1$classe,as.factor(pred1_classe))

## compare predicted to actual values for cross-validation set
actual2_classe <- as.matrix(as.character(train2$classe))
pred2_classe<-createClasse(p2)
confusionMatrix(train2$classe,as.factor(pred2_classe))


## predict for test set
predTest_classe<-createClasse(ptest)

## display results
paste(predTest_classe)

## write files for prediction submission

## function to create one-charater files from test-set predictions
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE,eol="")
  }
}
## write files
pml_write_files(as.character(predTest_classe))
