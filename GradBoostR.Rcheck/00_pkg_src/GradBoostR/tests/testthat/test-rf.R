test_that("RF regression works", {
  set.seed(1)
  X <- matrix(rnorm(300), 100, 3)
  y <- rnorm(100)

  m <- gradboostr_fit(X, y, method = "rf")
  preds <- gradboostr_predict(m, X)

  expect_equal(length(preds), nrow(X))
  expect_true(is.numeric(preds))
})
test_that("RF classification works", {
  set.seed(1)
  X <- matrix(rnorm(300), 100, 3)
  y <- factor(sample(c("A", "B"), 100, TRUE))

  m <- gradboostr_fit(X, y, method = "rf_class")

  preds_class <- gradboostr_predict(m, X, type = "response")
  preds_prob  <- gradboostr_predict(m, X, type = "prob")

  expect_true(is.factor(preds_class))
  expect_equal(levels(preds_class), levels(y))

  expect_true(is.matrix(preds_prob))
  expect_equal(nrow(preds_prob), nrow(X))
})
