# Fast Random Forest (Regression) Training

Internal function used by GradBoostR to train a regression random
forest.

## Usage

``` r
rf_fit_fast(X, y, n_trees = 200, max_depth = 5, mtry = 3)
```

## Arguments

- X:

  Numeric matrix of features.

- y:

  Numeric vector of responses.

- n_trees:

  Number of trees.

- max_depth:

  Maximum tree depth.

- mtry:

  Number of variables sampled at each split.

## Value

A list representing the trained forest.
