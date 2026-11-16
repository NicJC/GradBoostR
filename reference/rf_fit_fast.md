# Fit a Fast Regression Random Forest

Fits a random forest for a numeric response using the native GradBoostR
implementation.

## Usage

``` r
rf_fit_fast(X, y, n_trees = 200, max_depth = 5, mtry = 3)
```

## Arguments

- X:

  A numeric matrix containing predictor values. Rows represent
  observations and columns represent predictors.

- y:

  A numeric response vector containing one value for each row of `X`.

- n_trees:

  A positive integer giving the number of trees to fit.

- max_depth:

  A positive integer giving the maximum depth of each tree.

- mtry:

  A positive integer giving the number of predictors considered at each
  split. It must not exceed `ncol(X)`.

## Value

A list with one element per fitted regression tree. Each element stores
the structure of one tree in the random forest. The length of the list
equals `n_trees`. The returned forest can be passed to
[`rf_predict_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_predict_fast.md)
to generate numeric response predictions.

## Examples

``` r
set.seed(528)

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

length(forest)
#> [1] 5

stopifnot(
  is.list(forest),
  length(forest) == 5L
)
```
