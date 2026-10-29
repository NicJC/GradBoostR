test_that("neural-network prediction returns a numeric vector", {
  set.seed(864)

  n <- 120L

  X <- cbind(
    X1 = rnorm(n),
    X2 = rnorm(n),
    X3 = rnorm(n)
  )

  y <- (
    2.0 * X[, "X1"] -
      0.5 * X[, "X2"] +
      rnorm(n, sd = 0.7)
  )

  X_scaled <- scale(X)

  set.seed(273)

  model <- gradboostr_fit(
    X_scaled,
    y,
    method = "nn",
    hidden = 10,
    epochs = 50,
    learning_rate = 0.01
  )

  prediction <- gradboostr_predict(
    model,
    X_scaled,
    type = "response"
  )

  expect_s3_class(
    model,
    "gradboostr_model"
  )

  expect_identical(
    model$n_features,
    ncol(X_scaled)
  )

  expect_type(
    prediction,
    "double"
  )

  expect_null(
    dim(prediction)
  )

  expect_length(
    prediction,
    nrow(X_scaled)
  )

  expect_true(
    all(is.finite(prediction))
  )
})

test_that("prediction rejects the wrong number of predictors", {
  set.seed(864)

  X <- matrix(
    rnorm(90),
    ncol = 3
  )

  y <- rnorm(
    nrow(X)
  )

  model <- gradboostr_fit(
    X,
    y,
    method = "nn",
    hidden = 5,
    epochs = 5,
    learning_rate = 0.01
  )

  bad_newdata <- matrix(
    rnorm(20),
    ncol = 2
  )

  expect_error(
    gradboostr_predict(
      model,
      bad_newdata,
      type = "response"
    ),
    paste0(
      "`newdata` has 2 predictors, but the model ",
      "was fitted with 3 predictors."
    ),
    fixed = TRUE
  )
})

test_that("prediction rejects missing predictor values", {
  set.seed(864)

  X <- matrix(
    rnorm(90),
    ncol = 3
  )

  y <- rnorm(
    nrow(X)
  )

  model <- gradboostr_fit(
    X,
    y,
    method = "nn",
    hidden = 5,
    epochs = 5,
    learning_rate = 0.01
  )

  newdata <- X
  newdata[1, 1] <- NA_real_

  expect_error(
    gradboostr_predict(
      model,
      newdata
    ),
    "must not contain missing values",
    fixed = TRUE
  )
})

test_that("random forest prediction returns a numeric vector", {
  set.seed(123)

  X <- matrix(
    rnorm(120),
    ncol = 3
  )

  y <- (
    2 * X[, 1] -
      0.5 * X[, 2] +
      rnorm(nrow(X), sd = 0.2)
  )

  model <- gradboostr_fit(
    X,
    y,
    method = "rf"
  )

  prediction <- gradboostr_predict(
    model,
    X,
    type = "response"
  )

  expect_type(
    prediction,
    "double"
  )

  expect_null(
    dim(prediction)
  )

  expect_length(
    prediction,
    nrow(X)
  )

  expect_true(
    all(is.finite(prediction))
  )
})

test_that("gradient boosting stores n_features", {
  set.seed(123)

  X <- matrix(
    rnorm(90),
    ncol = 3
  )

  y <- rnorm(
    nrow(X)
  )

  model <- gradboostr_fit(
    X,
    y,
    method = "gbm",
    learning_rate = 0.1,
    n_trees = 10
  )

  expect_true(
    "n_features" %in% names(model)
  )

  expect_identical(
    model$n_features,
    as.integer(ncol(X))
  )
})

test_that("gradient boosting rejects an incorrect predictor count", {
  set.seed(123)

  X <- matrix(
    rnorm(90),
    ncol = 3
  )

  y <- rnorm(
    nrow(X)
  )

  model <- gradboostr_fit(
    X,
    y,
    method = "gbm",
    learning_rate = 0.1,
    n_trees = 10
  )

  bad_newdata <- matrix(
    rnorm(20),
    ncol = 2
  )

  expect_error(
    gradboostr_predict(
      model,
      bad_newdata,
      type = "response"
    ),
    paste0(
      "`newdata` has 2 predictors, but the model ",
      "was fitted with 3 predictors."
    ),
    fixed = TRUE
  )
})


