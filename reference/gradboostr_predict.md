# Generate predictions from a GradBoostR model

Generates regression predictions, predicted classes, or class
probabilities from a model fitted with
[`gradboostr_fit()`](https://nicjc.github.io/GradBoostR/reference/gradboostr_fit.md).

## Usage

``` r
gradboostr_predict(object, newdata, type = c("response", "prob"), ...)
```

## Arguments

- object:

  A fitted object created by
  [`gradboostr_fit()`](https://nicjc.github.io/GradBoostR/reference/gradboostr_fit.md).

- newdata:

  A numeric matrix or data frame containing predictor values.

- type:

  Prediction type. Use `"response"` for regression predictions or
  classification labels. Use `"prob"` for classification probabilities.

- ...:

  Additional arguments reserved for future use.

## Value

For regression models, a numeric vector with one prediction per
observation. For classification models, either a factor of predicted
classes when `type = "response"` or a numeric probability matrix when
`type = "prob"`.

## Examples

``` r
set.seed(9578)

X <- cbind(
  x1 = rnorm(30),
  x2 = rnorm(30)
)

y <- 2 * X[, "x1"] -
  X[, "x2"] +
  rnorm(30, sd = 0.2)

model <- gradboostr_fit(
  X,
  y,
  method = "gbm",
  learning_rate = 0.1,
  n_trees = 5
)

predictions <- gradboostr_predict(
  model,
  X,
  type = "response"
)

head(predictions)
#> [1]  0.7547721 -0.6574127 -0.6574127  0.7547721  0.7547721  0.7547721

stopifnot(
  is.numeric(predictions),
  length(predictions) == nrow(X),
  all(is.finite(predictions))
)
```
