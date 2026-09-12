test_that("Wrapper errors behave correctly", {
  X <- data.frame(a = 1:10, b = 1:10)
  y <- rnorm(10)

  expect_error(gradboostr_fit("not matrix", y))
  expect_error(gradboostr_predict("not model", X))

  m <- gradboostr_fit(as.matrix(X), y)
  expect_error(gradboostr_predict(m, "not matrix"))
})
