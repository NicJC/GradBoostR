test_that("NN model trains and predicts", {
  set.seed(1)
  X <- matrix(rnorm(200), 100, 2)
  y <- rnorm(100)

  m <- gradboostr_fit(X, y, method = "nn")
  preds <- gradboostr_predict(m, X)

  expect_equal(length(preds), nrow(X))
  expect_true(is.numeric(preds))
})
