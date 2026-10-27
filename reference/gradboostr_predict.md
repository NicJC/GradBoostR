# Predict using a GradBoostR model

Predict using a GradBoostR model

## Usage

``` r
gradboostr_predict(object, newdata, type = c("response", "prob"), ...)
```

## Arguments

- object:

  A gradboostr_model from gradboostr_fit().

- newdata:

  Matrix or data.frame of features.

- type:

  "response" or "prob" (classification only).

- ...:

  Not used.

## Value

Numeric vector (regression), factor (classification), or probability
matrix.
