# Fit a Fast Classification Random Forest

Fits a random forest for classification using the native GradBoostR
implementation.

## Usage

``` r
rf_class_fit_fast(
  X,
  y,
  n_trees = 100,
  max_depth = 5,
  mtry = 3,
  use_gini = TRUE
)
```

## Arguments

- X:

  A numeric matrix containing predictor values. Rows represent
  observations and columns represent predictors.

- y:

  An integer or numeric vector containing an encoded class label for
  each row of `X`.

- n_trees:

  A positive integer giving the number of trees to fit.

- max_depth:

  A positive integer giving the maximum depth of each tree.

- mtry:

  A positive integer giving the number of predictors considered at each
  split. It must not exceed `ncol(X)`.

- use_gini:

  A logical value indicating whether Gini impurity is used to evaluate
  candidate splits.

## Value

A list with one element per fitted classification tree. Each element
stores the structure of one tree in the random forest. The length of the
list equals `n_trees`. The returned forest can be passed to
[`rf_class_predict_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_predict_fast.md),
[`rf_class_predict_prob_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_predict_prob_fast.md),
[`rf_predict_fast_class()`](https://nicjc.github.io/GradBoostR/reference/rf_predict_fast_class.md),
or
[`rf_predict_fast_class_prob()`](https://nicjc.github.io/GradBoostR/reference/rf_predict_fast_class_prob.md)
to generate predictions.

## Examples

``` r
set.seed(983)

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

length(forest)
#> [1] 5

stopifnot(
  is.list(forest),
  length(forest) == 5L
)
```
