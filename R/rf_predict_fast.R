#' Fast Random Forest Prediction
#'
#' @param X Numeric matrix of predictors
#' @param forest A forest object returned by rf_fit_fast()
#'
#' @return Numeric vector of predictions
#' @export
rf_predict_fast <- function(X, forest) {
  .Call(`_GradBoostR_rf_predict_fast`, X, forest)
}

