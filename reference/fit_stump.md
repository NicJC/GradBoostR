# Fit a decision stump

Fit a decision stump

## Usage

``` r
fit_stump(X, y)
```

## Arguments

- X:

  Numeric matrix of features

- y:

  Numeric vector of labels

## Value

A list containing stump parameters

## Examples

``` r
set.seed(96842)

X <- cbind(
  x1 = rnorm(30),
  x2 = rnorm(30)
)

y <- ifelse(
  X[, "x1"] > 0,
  2,
  -2
)

stump <- fit_stump(
  X,
  y
)

stump
#> $feature
#> [1] 1
#> 
#> $threshold
#> [1] 0.128611
#> 
#> $left_value
#> [1] -2
#> 
#> $right_value
#> [1] 2
#> 
```
