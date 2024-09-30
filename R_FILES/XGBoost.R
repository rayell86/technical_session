#title: Fraud Detection with XGBoost
#author: Ray Elliott

# I loaded the necessary libraries for data manipulation, visualization, and machine learning.
library(dplyr)
library(tidyr)
library(caret)
library(tidyverse)
library(fastDummies)
library(xgboost)
library(ggplot2)

# I generated a dataset for accident data, simulating 100,000 records
n <- 100000
accident_data <- data.frame(
  claim_id = 1:n,  # I assigned a unique claim ID to each record
  time_of_accident = sample(0:23, n, replace = TRUE),  # I randomly selected the accident time (hours)
  accident_type = factor(sample(c("bi", "umbi", "um"), n, replace = TRUE)),  # I assigned accident types
  police_report = sample(c(0, 1), n, replace = TRUE),  # I simulated whether a police report was filed
  num_individuals = sample(1:5, n, replace = TRUE),  # I included the number of individuals involved
  involved_relationship = factor(sample(c("sibling", "friend", "child","parent","neighbor"), n, replace = TRUE)),  # I assigned relationships for the involved individuals
  type_of_coverage = factor(sample(c("BI", "UMBI", "UM", "MED PAY"), n, replace = TRUE)),  # I simulated the type of coverage used
  report_source = factor(sample(c("online", "phone"), n, replace = TRUE))  # I recorded the report source
)

# I viewed the first few rows of accident data to confirm the structure
head(accident_data)

# I generated a dataset for policy data, again simulating 100,000 records
policy_data <- data.frame(
  claim_id = 1:n,  # I ensured claim_id matched the accident data
  policy_limit = factor(sample(c("15/30", "50/100", "25/50", "100/300"), n, replace = TRUE)),  # I simulated policy limits
  policy_tenure = sample(1:30, n, replace = TRUE),  # I included the number of years the policy has been active
  num_drivers = sample(1:4, n, replace = TRUE),  # I added the number of drivers on the policy
  age = sample(18:80, n, replace = TRUE),  # I included the policyholder's age
  sex = factor(sample(c("male", "female"), n, replace = TRUE)),  # I assigned the policyholder's gender
  car_type = factor(sample(c("sedan", "SUV", "truck", "van"), n, replace = TRUE)),  # I assigned a car type
  make_model = factor(sample(c("model_a", "model_b", "model_c"), n, replace = TRUE)),  # I selected car models
  driver_status_change_date = sample(seq(as.Date('2010/01/01'), as.Date('2023/01/01'), by="day"), n, replace = TRUE),  # I generated random dates for driver status changes.
  fraudulent = sample(c(0, 1), n, replace = TRUE)  # I simulated whether a claim was flagged as fraudulent
)

# I viewed the first few rows of policy data to confirm the structure
head(policy_data)

# I merged the accident and policy data based on the claim_id
complete_data <- merge(accident_data, policy_data, by = "claim_id")

# I handled the date column by converting the driver_status_change_date to a numeric format
# I calculated the number of days since a reference date (2000-01-01)
reference_date <- as.Date("2000-01-01")
complete_data$driver_status_change_date_numeric <- as.numeric(difftime(complete_data$driver_status_change_date, reference_date, units = "days"))

# I removed the original date and claim_id columns as they were no longer needed
complete_data <- complete_data %>%
  select(-c(driver_status_change_date, claim_id))

# I one-hot encoded the categorical variables for both accident and policy data 
accident_encoded <- model.matrix(~ accident_type + involved_relationship + type_of_coverage + report_source - 1, data = complete_data)
policy_encoded <- model.matrix(~ policy_limit + sex + car_type + make_model - 1, data = complete_data)

# I combined the encoded columns with the rest of the data, excluding the original categorical columns.
complete_data_encoded <- cbind(complete_data[, !(names(complete_data) %in% c("accident_type", "involved_relationship", 
                                                                             "type_of_coverage", "report_source", 
                                                                             "policy_limit", "sex", "car_type", "make_model"))],
                               accident_encoded, policy_encoded)

# I viewed the first few rows of the encoded data to confirm the changes
head(complete_data_encoded)

# I scaled the numeric columns to a range between 0 and 1 for model optimization
scale_min_max <- function(x) { (x - min(x)) / (max(x) - min(x)) }

# I applied the scaling function to relevant numeric columns
complete_data_encoded <- complete_data_encoded %>%
  mutate(across(c(time_of_accident, num_individuals, policy_tenure, age, driver_status_change_date_numeric), scale_min_max))

# I viewed the first few rows of the scaled data to confirm the changes
head(complete_data_encoded)

# I separated the features (X) and the target variable (y) for the model
x <- as.matrix(complete_data_encoded[, -which(names(complete_data_encoded) == "fraudulent")])
y <- complete_data_encoded$fraudulent 

# Step 6: I trained an XGBoost model to assess feature importance.
xgb_model <- xgboost(data = x, label = y, max_depth = 6, nrounds = 100, objective = "binary:logistic", verbose = 0)

# I obtained the feature importance from the trained XGBoost model.
importance_matrix <- xgb.importance(model = xgb_model, feature_names = colnames(x))

# I viewed the most important features to understand their impact on the prediction of fraudulent claims.
head(importance_matrix)

# I plotted the feature importance to visually interpret the most influential variables.
xgb.plot.importance(importance_matrix)

#                             Feature       Gain      Cover  Frequency
#1: driver_status_change_date_numeric 0.29197356 0.59924980 0.28653295
#2:                               age 0.11991262 0.11774279 0.12187202
#3:                     policy_tenure 0.11834133 0.08058566 0.11766953
#4:                  time_of_accident 0.10345008 0.05569503 0.09952245
#5:                   num_individuals 0.04169573 0.01616905 0.04164279
#6:                       num_drivers 0.03840404 0.01392054 0.03992359
