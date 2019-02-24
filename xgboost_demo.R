#################################################################################
# This demo is limited to explaining how xgboost works in R. 
# The problem was provided from a previous Kaggle competition. 
# Scripts based on tutorials from Tianqi Chen, Tong He and Michaël Benesty
# http://www.kaggle.com/c/otto-group-product-classification-challenge
#################################################################################

# Introduction
# XGBoost is an implementation of the famous gradient boosting algorithm. The following demo includes a pipelined workflow of solving a product classification problems:
# 1. Data Preparation
# 2. 

options(java.parameters = "-Xmx10g")
# set work directory
setwd("/home/wxc011/RProjectsGIT/XGBoost")

# First, let’s load the packages and the dataset.
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
# magrittr and data.table are here to make the code cleaner and more rapid.

train <- fread('input/train.csv', header = T, stringsAsFactors = F)
test <- fread('input/test.csv', header=T, stringsAsFactors = F)

# Train dataset dimensions
dim(train)
# [1] 61878    95

# Training content
train[1:6, c(1,2:4,ncol(train)), with = F]

# Test dataset dimensions
dim(test)
# [1] 144368     94

# Test content
test[1:6, 1:5, with = F]

# Each column represents a feature measured by an integer. Each row is a product.
# Obviously the first column (ID) doesn’t contain any useful information. To let the algorithm focus on real stuff, we will delete the column.

# Delete ID column in training dataset
train[, id := NULL]

# Delete ID column in testing dataset
test[, id := NULL]

############ data structure ############
# feat_1 feat_2 feat_3 feat_4  target
# 1:      1      0      0      0 Class_1
# 2:      0      0      0      0 Class_1
# 3:      0      0      0      0 Class_1
# 4:      1      0      0      1 Class_1
# 5:      0      0      0      0 Class_1
# 6:      2      1      0      0 Class_1

# This is a  multi-class classification problem, i.e., each product will only be given one and only one of those labels. It is differnet from a multi-label classification problem, where each case can be assigned to multiple classes at the same time.

# Save the name of the last column
nameLastCol <- names(train)[ncol(train)]

# XGBoost only supports numberic input. So we will convert classes to integers. Also, the first class should start at 0.

# For that purpose, we will:
  # 1. extract target
  # 2. remove the word "Class_"
  # 3. convert to integers
  # 4. shift class index so that first class starts at 0

# Convert to classes to numbers
y <- train[, nameLastCol, with = F][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}
# Display the first 5 levels
y[1:5]

# We remove label column from training dataset, otherwise XGBoost would use it to guess the labels.
train[, nameLastCol:=NULL, with = F]

# data.table is an awesome implementation of data.frame, unfortunately it is not a format supported natively by XGBoost. We need to convert both datasets (training and test) in numeric Matrix format.

trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
# trainMatrix is training set with features only.

testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix


# Model training

# Before the learning we will use 3 fold cross validation through 5 iterations to evaluate the our error rate.
# 
# Look at the function documentation for more information.

numberOfClasses <- max(y) + 1

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 5
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)
# [1]	train-mlogloss:1.541044+0.002208	test-mlogloss:1.554883+0.004874 
# [2]	train-mlogloss:1.283153+0.002363	test-mlogloss:1.305527+0.005347 
# [3]	train-mlogloss:1.114745+0.002365	test-mlogloss:1.143681+0.006185 
# [4]	train-mlogloss:0.993974+0.002716	test-mlogloss:1.027952+0.006472 
# [5]	train-mlogloss:0.902192+0.003245	test-mlogloss:0.941489+0.005930 

# Finally, we are ready to train the real model!!!
nround = 50
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)

# Model understanding
# Feature importance
# So far, we have built a model made of nround trees.
# 
# To build a tree, the dataset is divided recursively several times. At the end of the process, you get groups of observations (here, these observations are properties regarding OTTO products).
# 
# Each division operation is called a split.
# 
# Each group at each division level is called a branch and the deepest level is called a leaf.
# 
# In the final model, these leafs are supposed to be as pure as possible for each tree, meaning in our case that each leaf should be made of one class of OTTO product only (of course it is not true, but that’s what we try to achieve in a minimum of splits).
# 
# Not all splits are equally important. Basically the first split of a tree will have more impact on the purity that, for instance, the deepest split. Intuitively, we understand that the first split makes most of the work, and the following splits focus on smaller parts of the dataset which have been missclassified by the first tree.
# 
# In the same way, in Boosting we try to optimize the missclassification at each round (it is called the loss). So the first tree will do the big work and the following trees will focus on the remaining, on the parts not correctly learned by the previous trees.
# 
# The improvement brought by each split can be measured, it is the gain.
# 
# Each split is done on one feature only at one value.

# Let’s see what the model looks like.

model <- xgb.dump(bst, with.stats = T)
model[1:10]


# Clearly, it is not easy to understand what it means.

# Basically each line represents a branch, there is the tree ID, the feature ID, the point where it splits, and information regarding the next branches (left, right, when the row for this feature is N/A).

# Hopefully, XGBoost offers a better representation: feature importance.

# Feature importance is about averaging the gain of each feature for all split and all trees.

# Then we can use the function xgb.plot.importance.

# Get the feature real names
names <- dimnames(trainMatrix)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")

(gg <- xgb.ggplot.importance(importance_matrix, top_n = 50, rel_to_first = TRUE))
