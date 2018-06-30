library(tidyverse)
library(keras)
library(jsonlite)
library(abind)
library(pracma)

# Boston housing (https://www.kaggle.com/mlg-ulb/creditcardfraud)
boston <- dataset_boston_housing(path = "boston_housing.npz", test_split = 0.2, seed = 113L)
train_X_mean <- boston$train$x %>% apply(., 2, mean)
train_X_sd <- boston$train$x %>% apply(., 2, sd)
boston_train_X <- boston$train$x %>% sweep(., 2, train_X_mean, "-") %>% sweep(., 2, train_X_sd, "/")
boston_train_Y <- boston$train$y
boston_test_X <- boston$test$x %>% sweep(., 2, train_X_mean, "-") %>% sweep(., 2, train_X_sd, "/")
boston_test_Y <- boston$test$y
save(file = "data/boston.RData",
     list = c("boston_train_X", "boston_train_Y",
              "boston_test_X", "boston_test_Y"))

# Fashion MNIST (https://www.kaggle.com/zalando-research/fashionmnist)
fashion_mnist_train <- read_csv("data/fashion-mnist_train.csv")
fashion_mnist_test <- read_csv("data/fashion-mnist_test.csv")
fashion_mnist_train_X <- fashion_mnist_train %>% select(-label) %>% as.matrix()
fashion_mnist_train_Y <- fashion_mnist_train %>% pull(label)
fashion_mnist_test_X <- fashion_mnist_test %>% select(-label) %>% as.matrix()
fashion_mnist_test_Y <- fashion_mnist_test %>% pull(label)
save(file = "data/fashion_mnist.RData",
     list = c("fashion_mnist_train_X", "fashion_mnist_train_Y",
              "fashion_mnist_test_X", "fashion_mnist_test_Y"))

# Ships (https://www.kaggle.com/rhammell/ships-in-satellite-imagery)
ships_json <- fromJSON("data/shipsnet.json")[1:2]
ships_data <- ships_json$data %>% apply(., 1, function(x) {
  r <- matrix(x[1:6400], 80, 80, byrow = TRUE) / 255
  g <- matrix(x[6401:12800], 80, 80, byrow = TRUE) / 255
  b <- matrix(x[12801:19200], 80, 80, byrow = TRUE) / 255
  list(array(c(r, g, b), dim = c(80, 80, 3)))
}) %>% do.call(c, .) %>% abind(., along = 4) %>% aperm(c(4, 1, 2, 3))
ships_labels <- ships_json$labels
set.seed(1234)
indexes <- sample(1:2800, 0.7 * 2800)
ships_train <- list(data = ships_data[indexes,,,], labels = ships_labels[indexes])
ships_test <- list(data = ships_data[-indexes,,,], labels = ships_labels[-indexes])
save(file = "data/ships.RData",
     list = c("ships_train", "ships_test"))
