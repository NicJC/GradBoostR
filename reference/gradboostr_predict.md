# Generate predictions from a GradBoostR model

Generates regression predictions, predicted classes, or class
probabilities from a model fitted with
[`gradboostr_fit()`](https://nicjc.github.io/GradBoostR/reference/gradboostr_fit.md).

## Usage

``` r
gradboostr_predict(object, newdata, type = c("response", "prob"), ...)
```

## Arguments

- object:

  A fitted object created by
  [`gradboostr_fit()`](https://nicjc.github.io/GradBoostR/reference/gradboostr_fit.md).

- newdata:

  A numeric matrix or data frame containing predictor values.

- type:

  Prediction type. Use `"response"` for regression predictions or
  classification labels. Use `"prob"` for classification probabilities.

- ...:

  Additional arguments reserved for future use.

## Value

For regression models, a numeric vector with one prediction per
observation. For classification models, either a factor of predicted
classes when `type = "response"` or a numeric probability matrix when
`type = "prob"`.
