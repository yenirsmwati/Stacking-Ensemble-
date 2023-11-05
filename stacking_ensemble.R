#Load  libraries
library(caret)
library(e1071)
library(glmnet)
library(caretEnsemble)


#load dataset
data <-read.csv("D:\\Covid Dataset.csv")
head(data)


# Columns to factorize
columns_to_factorize <- c("Breathing.Problem", "Fever", "Dry.Cough", "Sore.throat", "Running.Nose", 
                          "Asthma", "Chronic.Lung.Disease", "Headache", "Heart.Disease", 
                          "Diabetes", "Hyper.Tension", "Fatigue", "Gastrointestinal", 
                          "Abroad.travel", "Contact.with.COVID.Patient", "Attended.Large.Gathering", 
                          "Visited.Public.Exposed.Places", "Family.working.in.Public.Exposed.Places", "COVID.19")

# Apply factor to selected columns
data[columns_to_factorize] <- lapply(data[columns_to_factorize], factor)


# Check the structure of the dataframe
str(data)

# Split the dataset into training and testing sets
set.seed(12345)
trainIndex = (createDataPartition(y=data$COVID.19, p=0.75, list=FALSE))
covid_train=data[trainIndex, ]
covid_test=data[-trainIndex, ]


# Train the stacking ensemble model
stack_control = trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions='all')
stacked_models = caretList(COVID.19~., data = covid_train, trControl = stack_control, methodList =c('svmRadial','glm'))
output = resamples(stacked_models)
output
summary(output)

# Stacking Ensemble using glm methods
stack = caretStack(stacked_models, method="glm", trControl=stack_control)
stack
summary(stack)

# Testing using stacking ensemble model
pred = predict(stack, covid_test[,1:18])
cm = confusionMatrix(covid_test$COVID.19, pred)
print(cm)

system("git --version")
