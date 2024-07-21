# Set CRAN mirror non-interactively
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install libraries if not already installed
if (!require(httpgd)) install.packages("httpgd")
if (!require(rpart.plot)) install.packages('rpart.plot')
if (!require(rpart)) install.packages('rpart')
if (!require(smotefamily)) install.packages('smotefamily')
if (!require(ROSE)) install.packages('ROSE')
if (!require(caret)) install.packages('caret')
if (!require(dplyr)) install.packages("dplyr")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(caTools)) install.packages('caTools')


# Load the httpgd package
library(httpgd)

# Close any existing graphics devices
while (dev.cur() > 1) dev.off()

# Start the httpgd graphics device
hgd()

# Importing the dataset
credit_card <- read.csv('/Users/paulaluput/Desktop/portfolio/portfolio.github.io/ML_proj/creditcard.csv')

# Glance at the structure of the dataset
str(credit_card)

# Convert class to a factor variable
# 0 is legitimate, 1 is fraud case
credit_card$Class <- factor(credit_card$Class, levels = c(0, 1))

# Get the summary of the data
summary(credit_card)

# Count the missing values
sum(is.na(credit_card))

#-----------------------------------------------------

# get the distribution of fraud and legit transactions in the dataset
table(credit_card$Class)

# get the percentage of fraud and legit transactions in the dataset
prop.table(table(credit_card$Class))

# pie chart of credit card transactions
labels <- c("legit", "fraud")
# rounding to 2 digits
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)), 2))
labels <- paste0(labels, "%")

pie(table(credit_card$Class), labels, col = c("pink", "red"), 
    main = "Pie Chart of Credit Card Transactions")

#-----------------------------------------------------

# no model predictions
predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0, 1))


library(caret)
confusionMatrix(data = predictions, reference = credit_card$Class)
# so far we haven't flagged any fraud transactions AS frauds

#-----------------------------------------------------
# before building model, let's take a smaller part of our dataset to make it computationally faster

library(dplyr)

set.seed(1)
# extracting 10% of our data
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)

library(ggplot2)

ggplot(data = credit_card, aes(x = V1, y = V2, col = Class)) +
    geom_point() +
    theme_bw() +
    scale_color_manual(values = c('pink', 'red'))

# Open the httpgd viewer in your browser
httpgd::hgd_browse()

#-----------------------------------------------------

# creating training and test sets for fraud detection model

library(caTools)

set.seed(123)

# sample.split splits data from vector Y into two sets in predefined ratio while
#   preserving relative ratios of different labels in Y. Used to split data used during classification
#   into train and test subsets.

# 80% of data will be in train set (true), 20% will be in test set (false)
data_sample = sample.split(credit_card$Class, SplitRatio=0.80)

train_data = subset(credit_card, data_sample==TRUE)

test_data = subset(credit_card, data_sample==FALSE)

# display dimensions
dim(train_data)
dim(test_data)

#-----------------------------------------------------

# random over-sampling (ROS)

# 22750 are legitimate, and only 35 are frauds
table(train_data$Class)

n_legit <- 22750
new_frac_legit <- 0.50
# how many legitimate cases we want (half)
new_n_total <- n_legit / new_frac_legit # = 22750 / 0.50

library(ROSE)
# creates possibly balanced samples by random over-sampling, under-sampling, or both minority examples
# Class ~ . -> we are using Class variables, all other variables are independent
oversampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "over",
                                    N = new_n_total,
                                    seed = 2024)

oversampled_credit <- oversampling_result$data

# now 22750 legitimate AND fraud cases
table(oversampled_credit$Class)

# position = position_jitter(width = 0.2) -> prevent overlapping and easier to interpert
ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
    geom_point(position = position_jitter(width = 0.2)) +
    theme_bw() +
    scale_color_manual(values = c('pink', 'red'))

#-----------------------------------------------------

# random under-sampling (RUS)

table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
# how many legitimate cases we want (half)
new_n_total <- n_fraud / new_frac_fraud # = 22750 / 0.50

# ROSE is installed alrdy
library(ROSE)

undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2024)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

# sadly losing a lot of cases
ggplot(data = undersampled_credit, aes(x = V1, y = V2, col = Class)) +
    geom_point() +
    theme_bw() +
    scale_color_manual(values = c('pink', 'red'))

#-----------------------------------------------------

# ROS and RUS

n_new <- nrow(train_data) # = 22785
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~ .,
                                data = train_data,
                                method = "both",
                                N = n_new,
                                p = fraction_fraud_new,
                                seed = 2024)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)


ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
    geom_point(position = position_jitter(width = 0.2)) +
    theme_bw() +
    scale_color_manual(values = c('pink', 'red'))


#-----------------------------------------------------

# using SMOTE to balance the dataset

library(smotefamily)

table(train_data$Class)

# Set the number of fraud and legitimate cases, and the desired percentage of legitimate cases
n0 <- 22750
n1 <- 35
# 60% will be legitimate cases
r0 <- 0.6

# Calculate the value for the dup_size parameter of SMOTE
ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1

# Generate synthetic positive instances
# SMOTE(data_frame, target_class, num_of_nearest_neighbors = 5, dup_size = 0)
# dup_size is num of times we run our SMOTE
smote_output = SMOTE(X = train_data[ , -c(1, 31)],
                     target = train_data$Class,
                     K = 5,
                     dup_size = ntimes)

# 60% of class cases will be 0 (legitimate)
# 40% of class cases will be 1 (fraud)
credit_smote <- smote_output$data

# change col name 'class' to 'Class'
colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

# Let's compare data sets now

# original data set
ggplot(train_data, aes(x = V1, y = V2, color = Class)) +
       geom_point() +
       scale_color_manual(values = c('pink', 'red'))

# new data set
ggplot(credit_smote, aes(x = V1, y = V2, color = Class)) +
       geom_point() +
       scale_color_manual(values = c('pink', 'red'))

# Open the httpgd viewer in your browser again to ensure the last plot is displayed
# httpgd::hgd_browse()


#-----------------------------------------------------

# Building a decision tree WITH smote

library(rpart)
library(rpart.plot)

# building a decision tree (ML model)
# Class is independent vars, . is everything but Class which is our dependent vars
CART_model <- rpart(Class ~ . , credit_smote)

# plot decision tree
# extra, plot, and tweak just change visuals
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)
# basically, if col v14 is >= -2.6 its legit, less is fraud

# Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

# Build confusion matrix to check how many are correct
confusionMatrix(predicted_val, test_data$Class)

# almost 99% accuracy!
# can predict 7 out of 9 cases as frauds

# ------------------------------
# now predict on WHOLE data set
predicted_val <- predict(CART_model, credit_card[,-1], type = 'class')
confusionMatrix(predicted_val, credit_card$Class)

# predicted 40 out of the 44 cases as fraud


#-----------------------------------------------------

# COMPARE w building a decision tree WITHOUT smote

# [,-1] excludes the first column (time column)
CART_model <- rpart(Class ~ . , train_data[,-1])

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud classes
predicted_val <- predict(CART_model, test_data[,-1], type = 'class')

# Build confusion matrix to check how many are correct
confusionMatrix(predicted_val, test_data$Class)

# 99% accuracy
# can predict only 6 out of 9 cases as fraud

# ------------------------------
# now predict on WHOLE data set
predicted_val <- predict(CART_model, credit_card[,-1], type = 'class')
confusionMatrix(predicted_val, credit_card$Class)

# predicted only 35 out of the 44 cases as fraud