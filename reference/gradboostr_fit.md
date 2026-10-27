# Train a GradBoostR model

Train a GradBoostR model

## Usage

``` r
gradboostr_fit(x, y, method = c("gbm", "rf", "rf_class", "nn"), ...)
```

## Arguments

- x:

  Matrix or data.frame of features.

- y:

  Numeric (regression) or factor/integer (classification).

- method:

  One of: "gbm", "rf", "rf_class", "nn".

- ...:

  Additional hyperparameters passed to the underlying C++ engine.

## Value

A gradboostr_model object.
