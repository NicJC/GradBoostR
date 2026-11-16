# Train a GradBoostR model

Train a GradBoostR model

## Usage

``` r
gradboostr_fit(x, y, method = c("gbm", "rf", "rf_class", "nn"), ...)
```

## Arguments

- x:

  A numeric matrix or data frame of predictor variables.

- y:

  A numeric response for regression or a factor/integer response for
  classification.

- method:

  One of `"gbm"`, `"rf"`, `"rf_class"`, or `"nn"`.

- ...:

  Additional hyperparameters passed to the selected model engine.

## Value

A fitted object of class `"gradboostr_model"`.
