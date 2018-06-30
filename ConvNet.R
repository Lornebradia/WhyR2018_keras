library(keras)
library(tidyverse)
source("utils.R")

# Load train and test sets
load("data/ships.RData")

# Check shape of the data
ships_train$data %>% dim()
ships_train$labels %>% dim()

# Sample image
plot_sample_image(ships_train$data, show_layers = TRUE, row_nr = 4)

# Change labels vectors to one-hot-encoding matrix
ships_train$labels <- ships_train$labels %>% to_categorical(., 2)
ships_test$labels <- ships_test$labels %>% to_categorical(., 2)

model1 <- keras_model_sequential()
summary(model1)

model1 %>%
  layer_conv_2d(
    input_shape = c(80, 80, 3),
    filter = 32, kernel_size = c(3, 3), strides = c(1, 1),
    activation = "relu")
summary(model1)

model1 %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3, 3), strides = c(1, 1),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(2, activation = "softmax")
summary(model1)

model1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_sgd(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

ships_fit1 <- model1 %>% fit(ships_train$data, ships_train$labels,
                             epochs = 20, batch_size = 32,
                             validation_split = 0.2)

model1 %>%
  predict(ships_test$data) %>%
  head()

model1 %>% evaluate(ships_test$data, ships_test$labels)

save_model_hdf5(model1, "model1.hdf5")

# More complicated model

model2 <- keras_model_sequential() %>%
  layer_conv_2d(
    filter = 32, kernel_size = c(3, 3), padding = "same", 
    input_shape = c(80, 80, 3), activation = "relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3, 3), padding = "same",
                activation = "relu") %>%
  layer_conv_2d(filter = 64, kernel_size = c(3, 3),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(2, activation = "softmax")

model2 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adamax(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

ships_fit2 <- model2 %>% fit(ships_test$data, ships_test$labels,
                             epochs = 20, batch_size = 32,
                             validation_split = 0.2)

model2 %>% evaluate(ships_test$data, ships_test$labels)
