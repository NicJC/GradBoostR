#' Generate predictions from a GradBoostR model
#'
#' Generates regression predictions, predicted classes, or class
#' probabilities from a model fitted with [gradboostr_fit()].
#'
#' @param object A fitted object created by [gradboostr_fit()].
#' @param newdata A numeric matrix or data frame containing predictor values.
#' @param type Prediction type. Use `"response"` for regression predictions
#'   or classification labels. Use `"prob"` for classification probabilities.
#' @param ... Additional arguments reserved for future use.
#'
#' @return For regression models, a numeric vector with one prediction per
#'   observation. For classification models, either a factor of predicted
#'   classes when `type = "response"` or a numeric probability matrix when
#'   `type = "prob"`.
#'
#' @export
gradboostr_predict <- function(
    object,
    newdata,
    type = c("response", "prob"),
    ...
) {
  if (!inherits(object, "gradboostr_model")) {
    stop(
      "`object` must be a fitted `gradboostr_model`.",
      call. = FALSE
    )
  }

  if (missing(newdata) || is.null(newdata)) {
    stop(
      "`newdata` must be supplied.",
      call. = FALSE
    )
  }

  if (is.data.frame(newdata)) {
    newdata <- as.matrix(newdata)
  }

  if (!is.matrix(newdata)) {
    stop(
      "`newdata` must be a numeric matrix or data frame.",
      call. = FALSE
    )
  }


  if (!is.numeric(newdata)) {
    stop(
      "`newdata` must contain numeric predictors.",
      call. = FALSE
    )
  }

  if (!is.null(object$n_features) &&
      ncol(newdata) != object$n_features) {
    stop(
      sprintf(
        paste0(
          "`newdata` has %d predictors, but the model ",
          "was fitted with %d predictors."
        ),
        ncol(newdata),
        object$n_features
      ),
      call. = FALSE
    )
  }

  if (nrow(newdata) == 0L) {
    stop(
      "`newdata` must contain at least one observation.",
      call. = FALSE
    )
  }

  if (ncol(newdata) == 0L) {
    stop(
      "`newdata` must contain at least one predictor.",
      call. = FALSE
    )
  }

  if (anyNA(newdata)) {
    stop(
      "`newdata` must not contain missing values.",
      call. = FALSE
    )
  }

  if (any(!is.finite(newdata))) {
    stop(
      "`newdata` must contain only finite values.",
      call. = FALSE
    )
  }

  type <- match.arg(type)

  method <- object$method

  if (is.null(method) ||
      length(method) != 1L ||
      !is.character(method)) {
    stop(
      "The fitted model does not contain a valid `method` field.",
      call. = FALSE
    )
  }

  if (identical(method, "gbm")) {
    if (!identical(type, "response")) {
      stop(
        paste0(
          "Gradient boosting regression supports only ",
          "`type = \"response\"`."
        ),
        call. = FALSE
      )
    }

    prediction <- gbm_predict(
      newdata,
      object$model,
      object$learning_rate,
      object$init_value
    )

    return(
      validate_regression_prediction(
        prediction,
        nrow(newdata),
        method = "Gradient boosting"
      )
    )
  }

  if (identical(method, "rf")) {
    if (!identical(type, "response")) {
      stop(
        paste0(
          "Random forest regression supports only ",
          "`type = \"response\"`."
        ),
        call. = FALSE
      )
    }

    prediction <- rf_predict_fast(
      newdata,
      object$model
    )

    return(
      validate_regression_prediction(
        prediction,
        nrow(newdata),
        method = "Random forest"
      )
    )
  }

  if (identical(method, "nn")) {
    if (!identical(type, "response")) {
      stop(
        paste0(
          "Neural network regression supports only ",
          "`type = \"response\"`."
        ),
        call. = FALSE
      )
    }

    prediction <- nn_predict_model(
      newdata,
      object$model
    )

    return(
      validate_regression_prediction(
        prediction,
        nrow(newdata),
        method = "Neural network"
      )
    )
  }

  if (identical(method, "rf_class")) {
    if (is.null(object$class_labels) ||
        length(object$class_labels) == 0L) {
      stop(
        "The classification model does not contain class labels.",
        call. = FALSE
      )
    }

    if (identical(type, "response")) {
      raw_prediction <- as.vector(
        rf_predict_fast_class(
          newdata,
          object$model
        )
      )

      if (length(raw_prediction) != nrow(newdata)) {
        stop(
          paste0(
            "The classification engine returned an incorrect ",
            "number of predictions."
          ),
          call. = FALSE
        )
      }

      class_index <- as.integer(
        raw_prediction
      ) + 1L

      if (anyNA(class_index) ||
          any(class_index < 1L) ||
          any(class_index > length(object$class_labels))) {
        stop(
          "The classification engine returned invalid class indices.",
          call. = FALSE
        )
      }

      return(
        factor(
          object$class_labels[class_index],
          levels = object$class_labels
        )
      )
    }

    n_classes <- length(
      object$class_labels
    )

    probabilities <- rf_predict_fast_class_prob(
      newdata,
      object$model,
      n_classes
    )

    probabilities <- as.matrix(
      probabilities
    )

    if (nrow(probabilities) != nrow(newdata)) {
      stop(
        paste0(
          "The classification engine returned an incorrect ",
          "number of probability rows."
        ),
        call. = FALSE
      )
    }

    if (ncol(probabilities) != n_classes) {
      stop(
        paste0(
          "The classification engine returned an incorrect ",
          "number of probability columns."
        ),
        call. = FALSE
      )
    }

    if (any(!is.finite(probabilities))) {
      stop(
        "The classification engine returned non-finite probabilities.",
        call. = FALSE
      )
    }

    colnames(probabilities) <- object$class_labels

    return(probabilities)
  }

  stop(
    sprintf(
      "Unknown GradBoostR model method: `%s`.",
      method
    ),
    call. = FALSE
  )
}
