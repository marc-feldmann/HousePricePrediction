
# ---  Predict the sales prices of houses (SalePrice)    --- #

## DEPENDENCIES
library(readxl)
library(visdat)
library(corrplot)
library(imputeTS)
library(olsrr)
library(car)
library(glmnet)
library(ISLR)
library(tree)
library(neuralnet)

## 1) EXPLORATORY DATA ANALYSIS (EDA)
# To support further analysis, we will first generate some basic descriptive statistics.
data = read_excel("Ames housing dataset.xls")

# check overall data structure
# str(data)
# vis_dat(data)
# vis_miss(data)
# table(sapply(data, class))
# summary(data)

# check target variable distribution
hist(data$SalePrice, breaks="FD")

# Some learnings from EDA:
# > only 37 of the features are numerical
#   > of these, 4 are actually dates (years)
#   > of these, 1 is an observation identifier
# > the proportion of missing values is very low (0.3%)
# > the target variable (SalePrice) is not normally distributed
# > most to all numerical features have different scales


## 2) DATA PREPROCESSING
# remove whitespaces from column names for easier referencing
names(data)<-make.names(names(data),unique = TRUE)

# drop categorical features and observation identifier (since equals row number)
mask = sapply(data, is.numeric)
data_red = data[mask]
data_red = data_red[c(-1)]

# impute missing vales with column means
data_red = na_mean(data_red)
anyNA(data_red)

# remove multicollinearity > 0.7 (the feature less correlated with target var)
# and Totals as can be expected to be linear combinations of other features
corrplot(cor(data_red))
cor(data_red)
drops = c("1st.Flr.SF","TotRms.AbvGrd", "Garage.Area", "Garage.Yr.Blt", "Total.Bsmt.SF")
data_red = data_red[ , !(names(data_red) %in% drops)]

# log transform target variable
data_red$SalePrice = log(data_red$SalePrice)

# normalize scales
data_red_scaled = scale(data_red)
data_red_scaled = as.data.frame(data_red_scaled)


## 3) SPLIT DATA INTO TRAINING AND TEST DATA
# "Use 70% of the data for training, 30% of the data for testing.
# Use a seed value of 123."
set.seed(123)
n = dim(data_red_scaled)[1]
test = sample(n, round(n*0.3))
data_red_scaled_train = data_red_scaled[-test,]
data_red_scaled_test = data_red_scaled[test,]

## 4) MODEL TRAINING AND EVALUATION
# For prediction, we implement a variety of models and analyze their respective performance (MSE).

# -- 4a) linear regression
model_linreg = lm(SalePrice ~ ., data_red_scaled_train)
summary(model_linreg)
vif(model_linreg) # > there are aliased coefficients in the model

# identify and remove aliased coefficients
alias(model_linreg)

# train model without these aliased coefficients
model_linreg = lm(SalePrice ~ . -X1st.Flr.SF -X2nd.Flr.SF -Low.Qual.Fin.SF,
                  data_red_scaled_train)
summary(model_linreg)
vif(model_linreg)

# compute MSE on test data
preds = predict(model_linreg, data_red_scaled_test)
model_linreg_MSE = mean((data_red_scaled_test$SalePrice - preds)^2)

# -- 4b) LASSO
# transform data for use in glmnet functions
data_red_scaled_train_x = model.matrix(SalePrice ~ . -X1st.Flr.SF -X2nd.Flr.SF -Low.Qual.Fin.SF,
                 data_red_scaled_train)[,-1]
data_red_scaled_train_x = data_red_scaled_train_x[,-1] # remove first column
data_red_scaled_test_x = model.matrix(SalePrice ~ . -X1st.Flr.SF -X2nd.Flr.SF -Low.Qual.Fin.SF,
                 data_red_scaled_test)
data_red_scaled_test_x = data_red_scaled_test_x[,-1] # remove first column


# identify optimal penalty coefficient lambda_opt via 10-fold CV 
cv_model_lasso = cv.glmnet(data_red_scaled_train_x, data_red_scaled_train$SalePrice, alpha=1)
plot(cv_model_lasso)
lambda_opt = cv_model_lasso$lambda.min

# train model with lambda_opt
model_lasso = glmnet(data_red_scaled_train_x, data_red_scaled_train$SalePrice, alpha=1, lambda=lambda_opt)

# compute MSE on test data
preds = predict(model_lasso, s=lambda_opt, data_red_scaled_test_x)
model_lasso_MSE = mean((data_red_scaled_test$SalePrice - preds)^2)


# -- 4c) regression tree
# train model
model_tree = tree(SalePrice ~ . -X1st.Flr.SF -X2nd.Flr.SF -Low.Qual.Fin.SF,
                  data_red_scaled_train)
summary(model_tree)
plot(model_tree)
text(model_tree)

# check whether pruning (reducing notes to <10) can improve prediction quality
cv_model_tree = cv.tree(model_tree)
plot(cv_model_tree$size, cv_model_tree$dev, type="b")
# > no

# use tree for prediction
preds = predict(model_tree, data_red_scaled_test)
plot(preds, data_red_scaled_test$SalePrice)
abline(0,1)

# compute MSE on test data
model_tree_MSE = mean((data_red_scaled_test$SalePrice - preds)^2)

  
# -- 4d) neural network
model_ANN = neuralnet(SalePrice ~ . -X1st.Flr.SF -X2nd.Flr.SF -Low.Qual.Fin.SF,
                      data=data_red_scaled_train, hidden=c(6, 3),
                      linear.output=TRUE, lifesign='full', rep=3, stepmax=30000,
                      threshold=0.3)
plot(model_ANN, rep="best")

# use ANN for prediction
preds = predict(model_ANN, data_red_scaled_test)

# compute MSE on test data
model_ANN_MSE = mean((data_red_scaled_test$SalePrice - preds)^2)


## 5) MODEL RECOMMENDATION
# "Make a final recommendation about your preferred model."

# Compare Model MSEs
MSEs = c(model_linreg_MSE, model_lasso_MSE, model_tree_MSE, model_ANN_MSE)
barplot(MSEs, main="MSE comparison", names.arg=c("Linear Regression", "LASSO",
                                                 "Regression Tree",
                                                 "Neural Net"))

# Reflection on model comparison:
# At first glance, model comparison based on MSE suggests this case to nicely
# demonstrate that simpler models such as linear regression models
# can outperform more complex models while being simpler to implement,
# thus questioning the hype around models such as ANNs.
# However, at second glance, this conclusion is highly uncertain
# for two main reasons.
# 1) MSE difference to the next best model is marginal.
# Since some randomness is involved in model computation (e.g. initialization
# of weights in the neural net model), a very large number of training runs
# might lead to convergences on different MSEs, with LinReg possibly falling
# behind other models.
# 2) Optimization potentials for especially the tree and ANN models has by far
# not been exhausted, strongly suggesting that these models' MSEs may
# be significantly improved. It is here particularly important to note
# that categorical variables had not been utilized here to reduce data pre-
# processing effort as well as model complexity that e.g. one-hot encoding
# would have introduced.

# Conclusion: I recommend using the linear regression model as a preliminary
# model for predicting property prices but at the same time systematically
# explore other models' optimization potentials. This model also has the benefit
# of being comparatively easy to explain to decision-makers. This applies less
# for the other models, especially the ANN model. 
# In optimization, a priority could be to explore the predictive power of
# the not-yet-included categorical variables, potentially in combination with
# dimensionality reduction techniques such as PCA as encoding will significantly
# increase dataset dimensionality.
# After optimization, comparisons of optimized models against the preliminary
# Lin Reg model should then not be based solely on MSE, but jointly consider 
# other common regression model evaluation metrics such as RMSE and MAE.



