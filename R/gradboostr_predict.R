#' Predict using a GradBoostR model
#'
#' @param object A gradboostr_model from gradboostr_fit().
#' @param newdata Matrix or data.frame of features.
#' @param type "response" or "prob" (classification only).
#' @param ... Not used.
#'
#' @return Numeric vector (regression), factor (classification), or probability matrix.
#' @export
gradboostr_predict <- function(object, newdata, type = c("response", "prob"), ...) {

  # --- NEW: early guards ---
  if (!inherits(object, "gradboostr_model"))
    stop("`object` must be a gradboostr_model.")

  if (is.data.frame(newdata)) newdata <- as.matrix(newdata)
  if (!is.matrix(newdata))
    stop("`newdata` must be a matrix or data.frame.")
  # --------------------------

  type <- match.arg(type)
  method <- object$method

  if (method == "gbm") {
    return(gbm_predict(newdata,
                       object$model,
                       object$learning_rate,
                       object$init_value))
  }

  if (method == "rf") {
    return(rf_predict_fast(newdata, object$model))
  }

  if (method == "rf_class") {

    if (type == "response") {
      raw <- rf_predict_fast_class(newdata, object$model)
      return(factor(object$class_labels[raw + 1],
                    levels = object$class_labels))
    }

    if (type == "prob") {
      n_classes <- length(object$class_labels)
      probs <- rf_predict_fast_class_prob(newdata, object$model, n_classes)
      colnames(probs) <- object$class_labels
      return(probs)
    }
  }

  if (method == "nn") {
    return(nn_predict_model(newdata, object$model))
  }

  stop("Unknown method")
}
