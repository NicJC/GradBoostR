# Predict Positive-Class Probabilities

Generates estimated probabilities for class `1` from a binary random
forest fitted with
[`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).

## Usage

``` r
rf_class_predict_prob_fast(X, forest_list)
```

## Arguments

- X:

  A numeric matrix containing predictor values. The columns must
  correspond, in number and order, to the predictors used to fit
  `forest_list`.

- forest_list:

  A fitted binary-classification forest returned by
  [`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).

## Value

A numeric vector of length `nrow(X)`. Each element is the estimated
probability that the corresponding observation belongs to class `1`.
Returned values are between zero and one.

## Examples

``` r
set.seed(872)

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
  n_trees = 10,
  max_depth = 3
)

probabilities <- rf_class_predict_prob_fast(
  X,
  forest
)

head(probabilities)
#> [1] 0.2750000 1.0000000 0.7000000 0.9000000 1.0000000 0.2666667

stopifnot(
  is.numeric(probabilities),
  is.null(dim(probabilities)),
  length(probabilities) == nrow(X),
  all(is.finite(probabilities)),
  all(probabilities >= 0),
  all(probabilities <= 1)
)
```
