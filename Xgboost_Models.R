setwd("~/Desktop/Columbia/Spring22/5200/Kaggle/Models")

# Clear the memory
rm(list = ls())

# Read Data
data <- read.csv(file = "../Data/0417_data.csv")
scoringData <- read.csv(file = "../Data/0417_scoringData.csv")

# Load Libraries
library(dplyr)
library(tidyr)
library(stringr)
library(rpart)
library(caret)
library(ipred)
library(gbm)
library(vtreat)
library(xgboost)
library(lightgbm)
library(qdap)
library(ngram)
library(ggmap)
library(mltools)
library(data.table)

# Using one-hot to create dummy variables
data_xgboost_test = data %>%   
  select("id","property_type")
data_xgboost_test = as.data.table(data_xgboost_test)
data_xgboost_oh = one_hot(data_xgboost_test)    

data_xgboost = merge(data, data_xgboost_oh, by = "id", all.x = TRUE)   ## merge the dummy columns with the original data

score_xgboost_test = scoringData %>%   
  select("id","property_type")
score_xgboost_test = as.data.table(score_xgboost_test)
score_xgboost_oh = one_hot(score_xgboost_test)

score_xgboost = merge(scoringData, score_xgboost_oh, by = "id", all.x = TRUE)

#transfer
as.factor(data$host_identity_verified)

#property_type word count
data$property_type = as.character(data$property_type)
data = data %>%
  rowwise() %>%
  mutate(property_type_count = wordcount(property_type))
scoringData$property_type = as.character(scoringData$property_type)
scoringData = scoringData %>%
  rowwise() %>%
  mutate(property_type_count = wordcount(property_type))

table(data$property_type_count)

#cozy
text_cozy <- grep(pattern = "cozy", x = tolower(data$description))
length(text_cozy)/nrow(data) #percentage of rows in "description" that contains cozy
data$cozy = 0
data$cozy[text_cozy] = 1
#data %>% group_by(cozy) %>% summarize(mean(price)) 
text_cozy_s <- grep(pattern = "cozy", x = tolower(scoringData$description))
length(text_cozy_s)/nrow(scoringData) #percentage of rows in "description" that contains cozy
scoringData$cozy = 0
scoringData$cozy[text_cozy_s] = 1

#Clean Amenities
#amenities
#First, remove irrelevant signs
# remove the dot sign
data$amenities = gsub("\\.", "", data$amenities)   
# remove the space
data$amenities = data$amenities %>% stringr::str_replace_all("\\s", "") 
# remove quotation sign
data$amenities = noquote(data$amenities)   

# Second split the column and create dummy variables
data = cbind(data,mtabulate(strsplit(as.character(data$amenities), ',')))
head(data$amenities, 3)

#First, remove irrelevant signs
# remove the dot sign
scoringData$amenities = gsub("\\.", "", scoringData$amenities)   
# remove the space
scoringData$amenities = scoringData$amenities %>% stringr::str_replace_all("\\s", "") 
# remove quotation sign
scoringData$amenities = noquote(scoringData$amenities)   

# Second split the column and create dummy variables
scoringData = cbind(scoringData,mtabulate(strsplit(as.character(scoringData$amenities), ',')))
head(scoringData$amenities, 3)

#host_verifications
data$host_verifications = gsub("\\[", "", data$host_verifications) ## remove [
data$host_verifications = gsub("\\]", "", data$host_verifications) ## remove ]
data$host_verifications = gsub("\\'", "", data$host_verifications) ## remove '
data$host_verifications = data$host_verifications %>% stringr::str_replace_all("\\s", "") ## remove the space
data$host_verifications = noquote(data$host_verifications) ##  remove quotation sign

## Create dummy variables
data = cbind(data, mtabulate(strsplit(as.character(data$host_verifications), split = ','))) 


# Construct Model
#Select Variables from the datasets
data_vars = data %>% select(cozy, property_type_count, government_id, Doorman, Hotwater, Hottub, email, phone, Wifi, Airconditioning, Kitchen, Elevator, Washer, Dryer, perfect, beds, bathrooms, bedrooms_0, bedrooms_1, bedrooms_2, bedrooms_3plus, Bronx_dum, Brooklyn_dum, Manhattan_dum, Queens_dum, StatenIsland_dum, property_type_dum, maximum_nights, minimum_nights, availability_90, cancellation_policy, host_listings_count, accommodates, room_type, luxur, ideal, close, review_scores_rating, reviews_per_month, host_is_superhost, is_location_exact,host_identity_verified)

scoringData_vars = scoringData %>% select(cozy, property_type_count, government_id, Doorman, Hotwater, Hottub, email, phone, Wifi, Airconditioning, Kitchen, Elevator, Washer, Dryer, perfect, beds, bathrooms, bedrooms_0, bedrooms_1, bedrooms_2, bedrooms_3plus, Bronx_dum, Brooklyn_dum, Manhattan_dum, Queens_dum, StatenIsland_dum, property_type_dum, maximum_nights, minimum_nights, availability_90, cancellation_policy, host_listings_count, accommodates, room_type, luxur, ideal, close, review_scores_rating, reviews_per_month, host_is_superhost, is_location_exact,host_identity_verified)

#Dummy code the variables
trt = designTreatmentsZ(dframe = data,
                        varlist = names(data_vars))
newvars = trt$scoreFrame[trt$scoreFrame$code%in% c('clean','lev'),'varName']

train_input = prepare(treatmentplan = trt, 
                      dframe = data,
                      varRestriction = newvars)
test_input = prepare(treatmentplan = trt, 
                     dframe = scoringData,
                     varRestriction = newvars)
#xgboost
set.seed(1031)
tune_nrounds = xgb.cv(data=as.matrix(train_input), 
                      label = data$price,
                      nrounds=250,
                      nfold = 5,
                      verbose = 0)

#optimal n rounds
which.min(tune_nrounds$evaluation_log$test_rmse_mean)


xgboost = xgboost(data=as.matrix(train_input), 
                  label = data$price,
                  nrounds=which.min(tune_nrounds$evaluation_log$test_rmse_mean),
                  verbose = 0)
model_0417_03 = xgboost

#Calculate RMSE for train data
pred_0417_03 = predict(model_0417_03, newdata = as.matrix(data_vars))
rmse_0417_03 = sqrt(mean((pred_0417_03-data$price)^2)); rmse_0417_03


# Generate Predictions and check new levels
pred = predict(model_0417_03,newdata=as.matrix(scoringData_vars))

# Submission Files
submissionFile = data.frame(id = scoringData$id, price = pred) 

#check NA
checkna = (sum(sapply(submissionFile, function(x) sum(is.na(x)))) == 0)

#Loop for submission
if(checkna){
  write.csv(submissionFile, '../Predictions/0417_03_submission.csv',row.names = F)
} else {submissionFile$price[is.na(submissionFile$price)] = median(submissionFile$price, na.rm = TRUE)
}
