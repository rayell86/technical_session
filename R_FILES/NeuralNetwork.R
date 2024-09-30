
#title: "Fraud Detection with Sequential Neural Network Model"
#author: "Ray Elliott"

# necessary libraries
library(dplyr)
library(tidyr)
library(caret)
library(tidyverse)
library(tensorflow)
library(keras)
library(reticulate)
install_tensorflow()


# to estblish reproducibility
set.seed(1)

# generate mock claim data
n <- 100000
accident_data <- data.frame(
  claim_id = 1:n,
  time_of_accident = sample(0:23, n, replace = TRUE),  # Time in hours (0-23)
  accident_type = factor(sample(c("bi", "umbi", "um"), n, replace = TRUE)),
  police_report = sample(c(0, 1), n, replace = TRUE),  # 1 = Report filed, 0 = No report
  num_individuals = sample(1:5, n, replace = TRUE),  # Number of individuals involved
  involved_relationship = factor(sample(c("sibiling", "friend", "child","parent","neighbor"), n, replace = TRUE)),
  type_of_coverage = factor(sample(c("BI", "UMBI", "UM", "MED PAY"), n, replace = TRUE)),
  report_source = factor(sample(c("online", "phone"), n, replace = TRUE))
)
# I checked to see if the table was created properly
head(accident_data)

# generate mock policy data
policy_data <- data.frame(
  claim_id = 1:n,
  policy_limit = factor(sample(c("15/30", "50/100", "25/50", "100/300"), n, replace = TRUE)),  # Policy limits in $
  policy_tenure = sample(1:30, n, replace = TRUE),  # Tenure in years
  num_drivers = sample(1:4, n, replace = TRUE),  # Number of drivers on the policy
  age = sample(18:80, n, replace = TRUE),  # Age of policyholder
  sex = factor(sample(c("male", "female"), n, replace = TRUE)),  # Gender of policyholder
  car_type = factor(sample(c("sedan", "SUV", "truck", "van"), n, replace = TRUE)),  # Car type
  make_model = factor(sample(c("model_a", "model_b", "model_c"), n, replace = TRUE)),  # Car model
  driver_status_change_date = sample(seq(as.Date('2010/01/01'), as.Date('2023/01/01'), by="day"), n, replace = TRUE),  # Driver status change date
  fraudulent = sample(c(0, 1), n, replace = TRUE)  # Fraud indicator (0 = Non-fraud, 1 = Fraud)
)


# Check for missing values - it is important to make sure there are no na values
sum(is.na(accident_data))
sum(is.na(policy_data))

# in order to wrangle the dara properly and make sure that valuable data is not going to get deleted, I used a median for replacing the missing numerical data, if any!
# if the missing data is categorical, then I would use mode
accident_data$time_of_accident[is.na(accident_data$time_of_accident)] <- median(accident_data$time_of_accident, na.rm = TRUE)
policy_data$policy_tenure[is.na(policy_data$policy_tenure)] <- median(policy_data$policy_tenure, na.rm = TRUE)

# -- dates cannot be modified using one-hot encoding so I set a reference date to change its class to numeric --

reference_date <- as.Date("1970-01-01")
policy_data$driver_status_change_date_numeric <- as.numeric(policy_data$driver_status_change_date - reference_date)

# in order to faclitate scaling, I extracted year, month, and day for one-hot encoding
policy_data$driver_status_change_year <- as.numeric(format(policy_data$driver_status_change_date, "%Y"))
policy_data$driver_status_change_month <- as.factor(format(policy_data$driver_status_change_date, "%m"))
policy_data$driver_status_change_day <- as.factor(format(policy_data$driver_status_change_date, "%d"))

#since our data manipulation is in palce for date, I removed the original date column
policy_data$driver_status_change_date <- NULL

# Here I applied one-hot encoding on categorical features
accident_data_encoded <- model.matrix(~ accident_type + involved_relationship + type_of_coverage + report_source - 1, data = accident_data)
policy_data_encoded <- model.matrix(~ policy_limit + sex + car_type + make_model + driver_status_change_month + driver_status_change_day - 1, data = policy_data)

# I converted the data back to a data frame and removed original categorical columns
accident_data <- cbind(accident_data[, !(names(accident_data) %in% c("accident_type", "involved_relationship", "type_of_coverage", "report_source"))], accident_data_encoded)
policy_data <- cbind(policy_data[, !(names(policy_data) %in% c("policy_limit", "sex", "car_type", "make_model", "driver_status_change_month", "driver_status_change_day"))], policy_data_encoded)


scale_min_max <- function(x) { (x - min(x)) / (max(x) - min(x)) }

# After successful scaling, I applied the function to accident data (numeric columns only)
accident_data$time_of_accident <- scale_min_max(accident_data$time_of_accident)
accident_data$num_individuals <- scale_min_max(accident_data$num_individuals)
policy_data$policy_tenure <- scale_min_max(policy_data$policy_tenure)
policy_data$num_drivers <- scale_min_max(policy_data$num_drivers)
policy_data$age <- scale_min_max(policy_data$age)
policy_data$driver_status_change_date_numeric <- scale_min_max(policy_data$driver_status_change_date_numeric)

# not necessary but I also standardized Z Score
scale_standard <- function(x) { (x - mean(x)) / sd(x) }

# now that my data set is successfully wranglled, I merged themtogether
complete_data <- merge(accident_data, policy_data, by = "claim_id")

# I prepared the data for training by partitioning it to create training and test sets - training (80%) and testing (20%) 
trainIndex <- createDataPartition(complete_data$fraudulent, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_data <- complete_data[trainIndex, ]
test_data <- complete_data[-trainIndex, ]

# just to make sure the size of the training and testing sets checks out
dim(train_data)
dim(test_data)

# I then created y_train and y_test as the target variable - aka dependent variables
y_train <- as.numeric(as.integer(train_data$fraudulent))
y_test <- as.numeric(as.integer(test_data$fraudulent))

# I then separated features and target for better scaling and fine tuning - here I removed claim_id and the fradulent column to make sure they won't be accounted for in the model
x_train <- as.matrix(train_data[, !(names(train_data) %in% c('fraudulent', 'claim_id'))])
x_test <- as.matrix(test_data[, !(names(test_data) %in% c('fraudulent', 'claim_id'))])


y_train <- as.numeric(y_train)
y_test <- as.numeric(y_test)


# This is to define input shape based on the number of columns aka model attributes
input_shape <- ncol(x_train)

# I created a Sequential Neural Network model using Keras.
model <- keras_model_sequential() %>%

  # I added the first layer with 500 neurons (units) and used the ReLU activation function.
  # I also specified the input shape to match the input data.
  layer_dense(units = 500, activation = 'relu', input_shape = input_shape) %>%

  # I inserted a Dropout layer with a rate of 0.2 to reduce overfitting by randomly 
  # dropping 20% of the neurons during each training iteration.
  layer_dropout(rate = 0.2) %>%

  # I added a second dense layer with 400 neurons, again using the ReLU activation function.
  layer_dense(units = 400, activation = 'relu') %>%

  # I applied another Dropout layer with a 10% dropout rate to further prevent overfitting.
  layer_dropout(rate = 0.1) %>%

  # I added a third dense layer with 300 neurons, also with the ReLU activation function.
  layer_dense(units = 300, activation = 'relu') %>%

  # I used a third Dropout layer, this time with a 10% dropout rate.
  layer_dropout(rate = 0.1) %>%

  # Lastly, I added an output layer with 1 neuron and the sigmoid activation function.
  # This was used because I was working on a binary classification problem, and sigmoid 
  # would output a probability between 0 and 1.
  layer_dense(units = 1, activation = 'sigmoid')

# I then compiled the model
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)


x_train <- apply(x_train, 2, as.numeric)
x_test <- apply(x_test, 2, as.numeric)

# I then train the model
history <- model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 20,         
  batch_size = 32,     
  validation_split = 0.2  
)

# Then evaluate the model on the test data
model_evaluation <- model %>% evaluate(x_test, y_test)

cat("Test Loss:", model_evaluation[1], "\n")
cat("Test Accuracy:", model_evaluation[2], "\n")

# I also generated predictions for the test set (probabilities)
predictions <- model %>% predict(x_test)

# I streamlined predictions to binary (0 or 1) based on threshold (0.5)
predicted_labels <- ifelse(predictions > 0.5, 1, 0)
view(predicted_labels)

confusionMatrix(as.factor(predicted_labels), as.factor(y_test))


cm <- confusionMatrix(as.factor(predicted_labels), as.factor(y_test))

# Precision (TP / (TP + FP))
precision <- cm$byClass['Pos Pred Value']

# Recall or Sensitivity (TP / (TP + FN))
recall <- cm$byClass['Sensitivity']

# F1-Score (2 * Precision * Recall) / (Precision + Recall)
f1_score <- 2 * (precision * recall) / (precision + recall)

Confusion Matrix and Statistics

#          Reference
#Prediction    0    1
 #        0    0    0
 #        1  977 1023
                                          
 #              Accuracy : 0.5115          
 #                95% CI : (0.4893, 0.5336)
 #  No Information Rate : 0.5115          
 #  P-Value [Acc > NIR] : 0.509           
                                          
 #                Kappa : 0               
                                          
 #Mcnemar's Test P-Value : <2e-16          
                                          
 #          Sensitivity : 0.0000          
 #         Specificity : 1.0000          
 #     Pos Pred Value :    NaN          
 #        Neg Pred Value : 0.5115          
 #            Prevalence : 0.4885          
 #        Detection Rate : 0.0000          
 #  Detection Prevalence : 0.0000          
 #     Balanced Accuracy : 0.5000          
                                          
 #      'Positive' Class : 0      

model %>% save_model_hdf5("fraud_detection_model.h5")
final_model <- load_model_hdf5("fraud_detection_model.h5")
new_predictions <- final_model %>% predict(new_data)

final_predicted_labels <- ifelse(new_predictions > 0.5, 1, 0)

if (predicted_labels == 1) {
  # flag indicator for manual review
} else {
  
}

# I used a madeup threshold of 0.3 to predict fraud
predicted_labels <- ifelse(predictions > 0.3, 1, 0)
