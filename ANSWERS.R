# Ex. 1.
fashion_mnist_train_Y <- fashion_mnist_train_Y %>% to_categorical(., 10)
fashion_mnist_test_Y <- fashion_mnist_test_Y %>% to_categorical(., 10)

fashion_mnist_train_X <- fashion_mnist_train_X / 255
fashion_mnist_test_X <- fashion_mnist_test_X / 255

fashion_model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 10, activation = "softmax")

fashion_model %>% compile(
  optimizer = "sgd",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- fashion_model %>%
  fit(x = fashion_mnist_train_X,
      y = fashion_mnist_train_Y,
      validation_split = 0.2,
      epochs = 20,
      batch_size = 128)

fashion_model %>% evaluate(fashion_mnist_test_X, fashion_mnist_test_Y)

fashion_predictions <- fashion_model %>% predict(fashion_mnist_test_X)

# Ex. 2.
ships_train$labels <- ships_train$labels %>% to_categorical(., 2)
ships_test$labels <- ships_test$labels %>% to_categorical(., 2)

ships_model <- keras_model_sequential() %>%
  layer_conv_2d(
    input_shape = c(80, 80, 3),
    filter = 32, kernel_size = c(3, 3), strides = c(1, 1),
    activation = "relu")

ships_model %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3, 3), strides = c(1, 1),
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(2, activation = "softmax")

ships_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_sgd(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

ships_fit <- ships_model %>% fit(ships_train$data, ships_train$labels,
                                 epochs = 20, batch_size = 32,
                                 validation_split = 0.2)

ships_model %>% evaluate(ships_test$data, ships_test$labels)

save_model_hdf5(ships_model, "ships_model.hdf5")

# Ex. 3.
ships_model2 <- keras_model_sequential() %>%
  layer_conv_2d(
    filter = 32, kernel_size = c(3, 3), padding = "same", 
    input_shape = c(80, 80, 3), activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3, 3), padding = "same",
                activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(2, activation = "softmax")

ships_model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adamax(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

ships_fit2 <- ships_model2 %>% fit(ships_test$data, ships_test$labels,
                                   epochs = 20, batch_size = 32,
                                   validation_split = 0.2,
                                   callbacks = c(callback_early_stopping(monitor = "val_loss", patience = 5),
                                                 callback_model_checkpoint(monitor = "val_loss", period = 2,
                                                                           save_best_only = TRUE,
                                                                           filepath = "ships_best.hdf5"),
                                                 callback_tensorboard(log_dir = "logs")))
tensorboard("logs")

ships_model2 %>% evaluate(ships_test$data, ships_test$labels)
