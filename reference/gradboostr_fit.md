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

## Examples

``` r
set.seed(35)

X <- cbind(
  x1 = rnorm(30),
  x2 = rnorm(30)
)

y <- 2 * X[, "x1"] -
  X[, "x2"] +
  rnorm(30, sd = 0.2)

model <- gradboostr_fit(
  X,
  y,
  method = "gbm",
  learning_rate = 0.1,
  n_trees = 5
)

model$method
#> [1] "gbm"
model$n_features
#> [1] 2
```
