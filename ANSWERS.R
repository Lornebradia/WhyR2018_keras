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
