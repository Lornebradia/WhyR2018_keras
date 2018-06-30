library(keras)
library(tidyverse)

# EXAMPLE
# Load train and test sets
load("data/boston.RData")

# Check shape of the data
boston_train_X %>% dim()
boston_train_Y %>% dim()

# Initialize sequential model: keras_model_sequential()
boston_model <- keras_model_sequential()
summary(boston_model)

# Add hidden layer_dense() with 16 units and tanh function as activation
boston_model %>%
  layer_dense(units = 16, activation = "tanh", input_shape = c(13))
summary(boston_model)

# Explain why do we have 224 params ?
13 * 16 + 16

# Add output layer_dense() with 2 units and linear function as activation
boston_model %>%
  layer_dense(units = 1, activation = "linear")
summary(boston_model)

# Configure model for training. Use SGD as optimizer, binary_crossentropy as loss function and add accuracy as additional metric.
boston_model %>% compile(
  optimizer = "sgd",
  loss = "mse",
  metrics = c("mae")
)

# Fit the model
history <- boston_model %>%
  fit(x = boston_train_X,
      y = boston_train_Y,
      validation_split = 0.2,
      epochs = 100,
      batch_size = 30)

# Evaluate on test set
boston_model %>%
  evaluate(boston_test_X, boston_test_Y)

# Get predictions
boston_predictions <- boston_model %>% predict(boston_test_X)

# Save the model
save_model_hdf5(boston_model, "boston_model.hdf5")

# Ex.1 - Build a MLP for 10-class classification problem.
load("data/fashion_mnist.RData")

# 1. Change labels vectors to one-hot-encoding matrix using to_categorical() function
fashion_mnist_train_Y <- 
fashion_mnist_test_Y <- 

# 2. Scale pixel values to [0, 1] interval
fashion_mnist_train_X <- 
fashion_mnist_test_X <- 

# 3. Model architecture:
# Dense layer with 512 units and "relu" activation
# Dropout layer with 20% drop rate
# Dense layer with 512 units and "relu" activation
# Dropout layer with 20% drop rate
# Output dense layer (how many units and what activation should You use?)
fashion_model <- 

# 4. Set SGD as optimizer and use categorical crossentropy as loss function. Use accuracy as additional metric.


# 5. Fit the model. Use 20% of the data for validation, 20 epochs and 128 samples for batch size.
history <- 

# 6. Evaluate model on test set


# 7. Calculate predictions for the test set
fashion_predictions <-
