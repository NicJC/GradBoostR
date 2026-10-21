test_that("GBM model trains and predicts", {
  set.seed(1)
  X <- matrix(rnorm(200), 100, 2)
  y <- rnorm(100)

  m <- gradboostr_fit(X, y, method = "gbm", n_trees = 20)

  # Model structure
  expect_true(is.list(m$model))
  expect_equal(length(m$model), 20)

  stump <- m$model[[1]]
  expect_true(all(c("feature", "threshold", "left_val", "right_val") %in% names(stump)))

  # Prediction
  preds <- gradboostr_predict(m, X)
  expect_equal(length(preds), nrow(X))

  # Basic sanity: predictions should be numeric
  expect_true(is.numeric(preds))

  # Boosting should reduce MSE compared to init_value alone
  init_only <- rep(m$init_value, length(y))
  expect_true(mean((y - preds)^2) < mean((y - init_only)^2))
})
test_that("GBM stumps have correct structure", {
  set.seed(1)
  X <- matrix(rnorm(200), 100, 2)
  y <- rnorm(100)

  m <- gradboostr_fit(X, y, method = "gbm", n_trees = 5)

  for (st in m$model) {
    expect_true(is.list(st))
    expect_true(all(c("feature", "threshold", "left_val", "right_val") %in% names(st)))
    expect_true(is.numeric(st$threshold))
    expect_true(is.numeric(st$left_val))
    expect_true(is.numeric(st$right_val))
  }
})
