#' Train a GradBoostR model
#'
#' @param x Matrix or data.frame of features.
#' @param y Numeric (regression) or factor/integer (classification).
#' @param method One of: "gbm", "rf", "rf_class", "nn".
#' @param ... Additional hyperparameters passed to the underlying C++ engine.
#'
#' @return A gradboostr_model object.
#' @export
gradboostr_fit <- function(x, y, method = c("gbm", "rf", "rf_class", "nn"), ...) {
  method <- match.arg(method)

  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) stop("`x` must be a matrix or data.frame.")

  if (method == "rf_class") {
    if (!is.factor(y)) y <- factor(y)
    y_int <- as.integer(y) - 1L

    model <- rf_class_fit_fast(x, y_int, ...)   # <- use rf_class_fit_fast
    class_labels <- levels(y)

    learning_rate <- NA_real_
    init_value    <- NA_real_
    n_trees       <- NA_integer_

  } else if (method == "rf") {
    model <- rf_fit_fast(x, y, ...)             # <- your RF regression C++ function
    class_labels <- NULL

    learning_rate <- NA_real_
    init_value    <- NA_real_
    n_trees       <- NA_integer_
  }

  else if (method == "gbm") {
    args <- list(...)
    learning_rate <- args$learning_rate %||% 0.1
    n_trees       <- args$n_trees       %||% 100
    init_value    <- mean(y)

    model <- grad_boost_fit(
      x,
      y,
      learning_rate = learning_rate,
      n_trees       = n_trees
    )
    class_labels <- NULL
  }
  else if (method == "nn") {

    args <- list(...)
    hidden        <- args$hidden        %||% 10
    epochs        <- args$epochs        %||% 50
    learning_rate <- args$learning_rate %||% 0.01

    model <- nn_train_arma(
      x,
      y,
      hidden_units   = hidden,
      epochs         = epochs,
      learning_rate  = learning_rate
    )

    class_labels <- if (is.factor(y)) levels(y) else NULL
    init_value    <- NA_real_
    n_trees       <- NA_integer_
  }

  structure(
    list(
      method        = method,
      model         = model,
      class_labels  = class_labels,
      learning_rate = learning_rate,
      init_value    = init_value,
      n_trees       = n_trees
    ),
    class = "gradboostr_model"
  )

}
