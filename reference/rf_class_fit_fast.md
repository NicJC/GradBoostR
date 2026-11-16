# Fast Random Forest (Classification) Training

Internal function used by GradBoostR to train a classification random
forest. Not intended for direct user use.

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

  Numeric matrix of features.

- y:

  Numeric vector of class labels.

- n_trees:

  Number of trees.

- max_depth:

  Maximum tree depth.

- mtry:

  Number of variables sampled at each split.

- use_gini:

  Logical; use Gini impurity.

## Value

A list representing the trained forest.
