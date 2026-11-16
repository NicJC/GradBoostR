# Predict Class Probabilities from a Fast Classification Forest

Generates estimated class probabilities from a classification forest
fitted with
[`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).
This is a lower-level probability interface. For binary models where
only the positive-class probability is required, use
[`rf_class_predict_prob_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_predict_prob_fast.md).

## Usage

``` r
rf_predict_fast_class_prob(X, forest_list, n_classes)
```

## Arguments

- X:

  A numeric matrix containing predictor values. The columns must
  correspond, in number and order, to those used to fit `forest_list`.

- forest_list:

  A fitted classification forest returned by
  [`rf_class_fit_fast()`](https://nicjc.github.io/GradBoostR/reference/rf_class_fit_fast.md).

- n_classes:

  A positive integer giving the number of response classes.

## Value

A numeric matrix with `nrow(X)` rows and `n_classes` columns. Each row
contains the estimated class probabilities for one observation, and each
column corresponds to one encoded response class.

## Examples

``` r
set.seed(358)

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

probabilities <- rf_predict_fast_class_prob(
  X,
  forest,
  n_classes = 2L
)

head(probabilities)
#>      [,1] [,2]
#> [1,]    0    1
#> [2,]    0    1
#> [3,]    1    0
#> [4,]    1    0
#> [5,]    1    0
#> [6,]    1    0

stopifnot(
  is.matrix(probabilities),
  identical(
    dim(probabilities),
    c(nrow(X), 2L)
  ),
  is.numeric(probabilities),
  all(is.finite(probabilities)),
  all(probabilities >= 0),
  all(probabilities <= 1)
)
```
