#' Train a GradBoostR model
#'
#' @param x A numeric matrix or data frame of predictor variables.
#' @param y A numeric response for regression or a factor/integer response
#'   for classification.
#' @param method One of `"gbm"`, `"rf"`, `"rf_class"`, or `"nn"`.
#' @param ... Additional hyperparameters passed to the selected model engine.
#'
#' @return A fitted object of class `"gradboostr_model"`.
#'
#' @export
gradboostr_fit <- function(
    x,
    y,
    method = c(
      "gbm",
      "rf",
      "rf_class",
      "nn"
    ),
    ...
) {
  method <- match.arg(method)

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.matrix(x)) {
    stop(
      "`x` must be a matrix or data frame.",
      call. = FALSE
    )
  }

  if (!is.numeric(x)) {
    stop(
      "`x` must contain numeric predictors.",
      call. = FALSE
    )
  }

  if (nrow(x) == 0L) {
    stop(
      "`x` must contain at least one observation.",
      call. = FALSE
    )
  }

  if (ncol(x) == 0L) {
    stop(
      "`x` must contain at least one predictor.",
      call. = FALSE
    )
  }

  if (anyNA(x)) {
    stop(
      "`x` must not contain missing values.",
      call. = FALSE
    )
  }

  if (any(!is.finite(x))) {
    stop(
      "`x` must contain only finite values.",
      call. = FALSE
    )
  }

  if (length(y) != nrow(x)) {
    stop(
      sprintf(
        paste0(
          "`y` has length %d, but `x` contains ",
          "%d observations."
        ),
        length(y),
        nrow(x)
      ),
      call. = FALSE
    )
  }

  # Store this before fitting so every model records the same metadata.
  n_features <- as.integer(
    ncol(x)
  )

  if (identical(method, "rf_class")) {
    if (!is.factor(y)) {
      y <- factor(y)
    }

    if (nlevels(y) < 2L) {
      stop(
        "`y` must contain at least two classes.",
        call. = FALSE
      )
    }

    y_int <- as.integer(y) - 1L

    model <- rf_class_fit_fast(
      x,
      y_int,
      ...
    )

    class_labels <- levels(y)
    learning_rate <- NA_real_
    init_value <- NA_real_
    n_trees <- NA_integer_
  } else if (identical(method, "rf")) {
    if (!is.numeric(y)) {
      stop(
        "`y` must be numeric for random forest regression.",
        call. = FALSE
      )
    }

    if (anyNA(y) || any(!is.finite(y))) {
      stop(
        "`y` must contain only finite values.",
        call. = FALSE
      )
    }

    model <- rf_fit_fast(
      x,
      y,
      ...
    )

    class_labels <- NULL
    learning_rate <- NA_real_
    init_value <- NA_real_
    n_trees <- NA_integer_
  } else if (identical(method, "gbm")) {
    if (!is.numeric(y)) {
      stop(
        "`y` must be numeric for gradient boosting regression.",
        call. = FALSE
      )
    }

    if (anyNA(y) || any(!is.finite(y))) {
      stop(
        "`y` must contain only finite values.",
        call. = FALSE
      )
    }

    args <- list(...)

    learning_rate <- args$learning_rate %||% 0.1
    n_trees <- args$n_trees %||% 100L
    init_value <- mean(y)

    model <- grad_boost_fit(
      x,
      y,
      learning_rate = learning_rate,
      n_trees = n_trees
    )

    class_labels <- NULL
  } else if (identical(method, "nn")) {
    if (!is.numeric(y)) {
      stop(
        "`y` must be numeric for neural network regression.",
        call. = FALSE
      )
    }

    if (anyNA(y) || any(!is.finite(y))) {
      stop(
        "`y` must contain only finite values.",
        call. = FALSE
      )
    }

    args <- list(...)

    hidden <- args$hidden %||% 10L
    epochs <- args$epochs %||% 50L
    learning_rate <- args$learning_rate %||% 0.01

    model <- nn_train_arma(
      x,
      y,
      hidden_units = hidden,
      epochs = epochs,
      learning_rate = learning_rate
    )

    class_labels <- NULL
    init_value <- NA_real_
    n_trees <- NA_integer_
  }

  structure(
    list(
      method = method,
      model = model,
      class_labels = class_labels,
      learning_rate = learning_rate,
      init_value = init_value,
      n_trees = n_trees,
      n_features = n_features
    ),
    class = "gradboostr_model"
  )
}
