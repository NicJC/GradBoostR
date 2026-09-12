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
  if (!inherits(object, "gradboostr_model"))
    stop("object must be a gradboostr_model.")

  if (is.data.frame(newdata)) newdata <- as.matrix(newdata)
  if (!is.matrix(newdata)) stop("newdata must be a matrix or data.frame.")

  method <- object$method
  model  <- object$model
  labels <- object$class_labels

  if (method %in% c("rf_class", "nn")) {
    type <- match.arg(type)
  }

  preds <- switch(
    method,
    gbm = gbm_predict(
      newdata,
      model,
      learning_rate = object$learning_rate,
      init_value    = object$init_value
    ),


    rf  = rf_predict_fast(newdata, model),
    rf_class = {
      if (type == "response") {
        idx <- rf_predict_fast_class(newdata, model)
        factor(labels[idx + 1L], levels = labels)
      } else {
        rf_predict_fast_class(newdata, model, prob = TRUE)
      }
    },
    nn = {
      if (type == "response") {
        idx <- nn_predict_model(newdata, model)
        if (!is.null(labels))
          factor(labels[idx + 1L], levels = labels)
        else
          idx
      } else {
        nn_predict_model(newdata, model, prob = TRUE)
      }
    }
  )

  preds
}
