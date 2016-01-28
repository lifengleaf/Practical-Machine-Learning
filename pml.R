
# r code for practical machine learning course project

# read data
pmlTrainingRaw<- read.csv("pml-training.csv", header = TRUE,
                          na.strings = c("NA", ""))
pmlTestingRaw<- read.csv("pml-testing.csv", header = TRUE,
                         na.strings = c("NA", ""))

# remove all the variables with NAs
pmlTraining<- pmlTrainingRaw[, colSums(is.na(pmlTrainingRaw)) == 0]
pmlTesting<- pmlTestingRaw[, colSums(is.na(pmlTestingRaw)) == 0]

# remove variables related to time series and IDs
pmlTraining<- pmlTraining[,-(1:7)]

# drop variables with near zero variance
library(caret)
nzvData<- nearZeroVar(pmlTraining, saveMetrics = TRUE)

# find the columns with absolute correlation >= 0.9, excluding the factor outcome
highCor<- findCorrelation(cor(pmlTraining[,-53]), cutoff = 0.9)
pmlTraining<- pmlTraining[,-highCor]

# subset test data by keeping only the variables in the training data
pmlTesting<- pmlTesting[,(names(pmlTesting) %in% names(pmlTraining[,-53]))]
dim(pmlTraining)
dim(pmlTesting)

# divide the given training set into 4 small sets
set.seed(110)
inTrain1<- createDataPartition(y = pmlTraining$classe,
                               p = 0.25, list = FALSE)
training1<- pmlTraining[inTrain1, ]
trainRemd<- pmlTraining[-inTrain1,]

set.seed(110)
inTrain2<- createDataPartition(y = trainRemd$classe,
                               p = 0.33, list = FALSE)
training2<- trainRemd[inTrain2, ]
trainRemd<- trainRemd[-inTrain2,]

set.seed(110)
inTrain3<- createDataPartition(y = trainRemd$classe,
                               p = 0.5, list = FALSE)
training3<- trainRemd[inTrain3, ]
training4<- trainRemd[-inTrain3,]

# divide each small data set into training and testing sets
set.seed(110)
myInTrain1<- createDataPartition(y = training1$classe,
                                 p=0.6, list = FALSE)
myTraining1<- training1[myInTrain1,]
myTesting1<-  training1[-myInTrain1,]

set.seed(110)
myInTrain2<- createDataPartition(y = training2$classe,
                                 p=0.6, list = FALSE)
myTraining2<- training2[myInTrain2,]
myTesting2<-  training2[-myInTrain2,]

set.seed(110)
myInTrain3<- createDataPartition(y = training3$classe,
                                 p=0.6, list = FALSE)
myTraining3<- training3[myInTrain3,]
myTesting3<-  training3[-myInTrain3,]

set.seed(110)
myInTrain4<- createDataPartition(y = training4$classe,
                                 p=0.6, list = FALSE)
myTraining4<- training4[myInTrain4,]
myTesting4<-  training4[-myInTrain4,]


# Classification Tree
library(rpart)
model1<- train(myTraining1$classe ~., data = myTraining1,
               method = "rpart")

# out of sample error
prediction1<- predict(model1, newdata = myTesting1)
confusionMatrix(prediction1, myTesting1$classe)


# Support Vector Machine 
library(kernlab)
# predictors are obtained from a pca preprocessing with 5 components
pca_svm<- preProcess(myTraining1, method = "pca",
                     pcaComp = 5)
myTrainingPca<- predict(pca_svm, myTraining1)

# 5-fold cross validation
model2<- train(myTraining1$classe ~., data = myTrainingPca,
               method = "svmRadial", 
               trControl = trainControl(method = "cv", number = 5))

myTestingPca<- predict(pca_svm, myTesting1)
prediction2<- predict(model2, myTestingPca)
confusionMatrix(prediction2, myTesting1$classe)


# K Nearest Neighbor
model3<- train(myTraining1$classe ~., data = myTraining1,
               trControl = trainControl(method = "adaptive_cv"),
               method = "knn")

prediction3<- predict(model3, newdata = myTesting1)
confusionMatrix(prediction3, myTesting1$classe)


# Random Forest
library(randomForest)
# with only cross validation
model4<- train(myTraining1$classe ~., data = myTraining1,
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 4),
               method = "rf")

prediction4<- predict(model4, newdata = myTesting1)
confusionMatrix(prediction4, myTesting1$classe)

# same training method on the second small training set
model5<- train(myTraining2$classe ~., data = myTraining2,
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 4),
               method = "rf")

prediction5<- predict(model5, newdata = myTesting2)
confusionMatrix(prediction5, myTesting2$classe)

# predict against the given testing set
predict(model5, pmlTesting)

# same training method on the third small training set
model6<- train(myTraining3$classe ~., data = myTraining3,
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 4),
               method = "rf")

# predict on the third small testing set
prediction6<- predict(model6, newdata = myTesting3)
confusionMatrix(prediction6, myTesting3$classe)

# same training method on the fourth small training set
model7<- train(myTraining4$classe ~., data = myTraining4,
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 4),
               method = "rf")
# predict on the fourth small testing set
prediction7<- predict(model7, newdata = myTesting4)
confusionMatrix(prediction7, myTesting4$classe)

# predict against the given testing set
predict(model7, pmlTesting)

