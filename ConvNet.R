library(keras)
library(tidyverse)
library(gridExtra)
source("utils.R")

# Load train and test sets
load("data/ships.RData")

# Check shape of the data
ships_train$data %>% dim()
ships_train$labels %>% dim()

# Sample image
plot_sample_image(ships_train$data, ships_train$labels, show_layers = TRUE, row_nr = 7)

# Ex. 2 - Build a simple ConvNet for binary classification
# 1. Change labels vectors to one-hot-encoding matrix
ships_train$labels <- 
  ships_test$labels <- 
  
# 2. Initialize sequential model and add 2d convolutional layer
# with 32 filters, 3x3 kernel, 1x1 stride, "relu" activation
ships_model <- 
summary(ships_model)

# 3. Explain output shape and nr of params

# 4. Add more layers
# max pooling with 2x2 pool (kernel), 2x2 strides
# convolution with 64 filters, 3x3 kernel, 1x1 strides, "relu" activation
# max pooling with 2x2 pool (kernel), 2x2 strides
# flattening layer
# Dense layer as output

summary(ships_model)

# 5. Compile the model using binary crossentropy as loss function and
# SGD as optimizer with learning rate equal to 0.0001 and learning rate decay equal to 1e-6


# 6. Fit the model. Use 20% of the data for validation, 20 epochs and 32 samples for batch size.
ships_fit <- 
  
# 7. Evaluate model on test set
  
  
# 8. Save model in hdf5 format
  
  
# Ex 3. Build second model using batch normalization. Use early stopping and checkpoints. Save logs to Tensorboard
# 1. Model architecture:
# convolution with 64 filters, 3x3 kernel, 1x1 strides, "linear" activation, "same" padding
# batch normalization
# "relu" activation
# max pooling with 2x2 pool (kernel), 2x2 strides
# dropout layer with 25% drop rate
# convolution with 64 filters, 3x3 kernel, 1x1 strides, "linear" activation, "same" padding
# batch normalization
# "relu" activation
# max pooling with 2x2 pool (kernel), 2x2 strides
# dropout layer with 25% drop rate
# flattening layer
# dense layer with 512 units and "relu" activation
# dropout layer with 25% drop rate
# Dense layer as output
ships_model2 <- 

# 2. Compile the model using binary crossentropy as loss function and
# Adamax as optimizer with learning rate equal to 0.0001 and learning rate decay equal to 1e-6


# 3. Fit the model. Use 20% of the data for validation, 20 epochs and 32 samples for batch size.
# Use early stopping with respect to validation loss, set patience to 5.
# Save best model (create checkpoint) with respect to validation loss every 2 epochs.
# Save logs for Tensorboard (to the "logs" folder)
dir.create("logs")
ships_fit2 <- 

# 7. Evaluate model on test set

