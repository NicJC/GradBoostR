# Predict from a Fast Regression Random Forest

Generates numeric response predictions from a regression forest fitted
with
[`rf_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_fit_fast.md).

## Usage

``` r
rf_predict_fast(X, forest_list)
```

## Arguments

- X:

  A numeric matrix containing predictor values. The columns must
  correspond, in number and order, to those used to fit `forest_list`.

- forest_list:

  A fitted regression forest returned by
  [`rf_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_fit_fast.md).

## Value

A numeric vector of length `nrow(X)`. Each element is the predicted
numeric response for the corresponding observation.

## Examples

``` r
set.seed(672)

X <- cbind(
  x1 = rnorm(30),
  x2 = rnorm(30)
)

y <- (
  2 * X[, "x1"] -
    X[, "x2"] +
    rnorm(30, sd = 0.2)
)

forest <- rf_fit_fast(
  X,
  y,
  n_trees = 5L,
  max_depth = 3L,
  mtry = 1L
)

predictions <- rf_predict_fast(
  X,
  forest
)

head(predictions)
#> [1]  3.7548992 -0.3517303  1.1785604  1.1785604 -1.8439502 -2.7548203

stopifnot(
  is.numeric(predictions),
  is.null(dim(predictions)),
  length(predictions) == nrow(X),
  all(is.finite(predictions))
)
```
