# Predict Hard Classes from a Fast Classification Forest

Generates hard-class predictions from a classification forest fitted
with
[`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).
This is a lower-level prediction interface. Most users should use
[`rf_class_predict_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_predict_fast.md).

## Usage

``` r
rf_predict_fast_class(X, forest_list)
```

## Arguments

- X:

  A numeric matrix containing predictor values. The columns must
  correspond, in number and order, to those used to fit `forest_list`.

- forest_list:

  A fitted classification forest returned by
  [`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).

## Value

An integer vector of length `nrow(X)`. Each element is the predicted
class index for the corresponding observation. For a binary model fitted
with classes encoded as `0` and `1`, the returned values are `0L` or
`1L`.

## Examples

``` r
set.seed(614)

X <- cbind(
  x1 = rnorm(30),
  x2 = rnorm(30)
)

y <- as.integer(
  X[, "x1"] + X[, "x2"] > 0
)

forest <- rf_class_fit_fast(
  X,
  y,
  n_trees = 5L,
  max_depth = 3L,
  mtry = 1L,
  use_gini = TRUE
)

predictions <- rf_predict_fast_class(
  X,
  forest
)

head(predictions)
#> [1] 1 0 1 0 0 0

stopifnot(
  is.integer(predictions),
  is.null(dim(predictions)),
  length(predictions) == nrow(X),
  all(predictions %in% c(0L, 1L))
)
```
