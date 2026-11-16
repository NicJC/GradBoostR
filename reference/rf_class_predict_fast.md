# Predict Classes from a Fast Classification Random Forest

Generates binary class predictions from a classification forest fitted
with
[`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).

## Usage

``` r
rf_class_predict_fast(X, forest_list)
```

## Arguments

- X:

  A numeric matrix containing predictor values. The columns must
  correspond, in number and order, to those used to fit `forest_list`.

- forest_list:

  A fitted classification forest returned by
  [`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).

## Value

A numeric vector of length `nrow(X)`. Each element is the predicted
binary class, encoded as `0` or `1`, for the corresponding observation.

## Examples

``` r
set.seed(467)

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

predicted_class <- rf_class_predict_fast(
  X,
  forest
)

head(predicted_class)
#> [1] 1 1 1 0 0 0

stopifnot(
  is.numeric(predicted_class),
  is.null(dim(predicted_class)),
  length(predicted_class) == nrow(X),
  all(predicted_class %in% c(0, 1))
)
```
