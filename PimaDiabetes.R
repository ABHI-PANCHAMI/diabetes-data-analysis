# Install required packages if not already installed
if (!require("e1071")) install.packages("e1071")
if (!require("rpart")) install.packages("rpart")
if (!require("caret")) install.packages("caret")
if (!require("caTools")) install.packages("caTools")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("reshape2")) install.packages("reshape2")
if (!require("ggalt")) install.packages("ggalt")

# Load packages
library(e1071)
library(rpart)
library(caret)
library(caTools)
library(ggplot2)
library(reshape2)
library(ggalt)

# Load and view the dataset
data <- read.csv(file.choose())
View(data)
sum(is.na(data))
str(data)
head(data)

# Split data into training and test sets
library(caret)

set.seed(128)
trainIndex <- createDataPartition(data$Outcome, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]
head(train_data)

# Helper function for plotting confusion matrix
plot_confusion_matrix <- function(predictions, true_values, model_name) {
  cm <- confusionMatrix(factor(predictions), factor(true_values))
  cm_table <- as.data.frame(cm$table)
  colnames(cm_table) <- c("Predicted", "Actual", "Count")
  
  ggplot(data = cm_table, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile() +
    geom_text(aes(label = Count), color = "white", size = 6) +
    labs(title = paste("Confusion Matrix for", model_name),
         x = "Actual Outcome", y = "Predicted Outcome") +
    scale_fill_gradient(low = "blue", high = "red") +
    theme_minimal()
}

# Metrics calculation function
calculate_metrics <- function(true, predicted) {
  cm <- confusionMatrix(factor(predicted), factor(true))
  accuracy <- cm$overall['Accuracy']
  precision <- cm$byClass['Precision']
  recall <- cm$byClass['Recall']
  f1 <- cm$byClass['F1']
  error_rate <- 1 - accuracy
  return(c(accuracy, precision, recall, f1, error_rate))
}

# ---- SVM Model ----
svm_model <- svm(Outcome ~ ., data = train_data, type = "C-classification", kernel = "linear")
svm_pred <- predict(svm_model, test_data)
print(plot_confusion_matrix(svm_pred, test_data$Outcome, "SVM"))
svm_metrics <- calculate_metrics(test_data$Outcome, svm_pred)
print("SVM Metrics:")
print(svm_metrics)

# ---- Naive Bayes Model ----
nb_model <- naiveBayes(Outcome ~ ., data = train_data)
nb_pred <- predict(nb_model, test_data)
print(plot_confusion_matrix(nb_pred, test_data$Outcome, "Naive Bayes"))
nb_metrics <- calculate_metrics(test_data$Outcome, nb_pred)
print("Naive Bayes Metrics:")
print(nb_metrics)

# ---- Decision Tree Model ----
dt_model <- rpart(Outcome ~ ., data = train_data, method = "class")
dt_pred <- predict(dt_model, test_data, type = "class")
print(plot_confusion_matrix(dt_pred, test_data$Outcome, "Decision Tree"))
dt_metrics <- calculate_metrics(test_data$Outcome, dt_pred)
print("Decision Tree Metrics:")
print(dt_metrics)

# ---- Logistic Regression Model ----
lr_model <- glm(Outcome ~ ., data = train_data, family = binomial)
lr_pred_prob <- predict(lr_model, test_data, type = "response")
lr_pred <- ifelse(lr_pred_prob > 0.5, 1, 0)
print(plot_confusion_matrix(lr_pred, test_data$Outcome, "Logistic Regression"))
lr_metrics <- calculate_metrics(test_data$Outcome, lr_pred)
print("Logistic Regression Metrics:")
print(lr_metrics)

# ---- Comparison of Model Metrics ----
metrics <- data.frame(
  Model = c("SVM", "Naive Bayes", "Decision Tree", "Logistic Regression"),
  Accuracy = c(svm_metrics[1], nb_metrics[1], dt_metrics[1], lr_metrics[1]),
  Precision = c(svm_metrics[2], nb_metrics[2], dt_metrics[2], lr_metrics[2]),
  Recall = c(svm_metrics[3], nb_metrics[3], dt_metrics[3], lr_metrics[3]),
  F1_Score = c(svm_metrics[4], nb_metrics[4], dt_metrics[4], lr_metrics[4]),
  Error_Rate = c(svm_metrics[5], nb_metrics[5], dt_metrics[5], lr_metrics[5])
)
print("Comparison of Model Metrics:")
print(metrics)

# ---- Plot the Metrics for Visualization ----
metrics_long <- melt(metrics, id.vars = "Model")

# Bar plot
ggplot(metrics_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Comparison", y = "Score", x = "Model") +
  theme_minimal()

# Line plot
ggplot(metrics_long, aes(x = variable, y = value, group = Model, color = Model)) +
  geom_line(size = 1) + geom_point(size = 3) +
  labs(title = "Model Performance Across Metrics", x = "Metric", y = "Score") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Dumbbell plot
ggplot(metrics_long, aes(x = value, y = variable, color = Model)) +
  geom_dumbbell(aes(xend = value, y = variable), size = 3, colour_x = "lightblue", colour_xend = "darkblue") +
  labs(title = "Comparison of Model Metrics (Dumbbell Plot)", x = "Score", y = "Metric") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")
