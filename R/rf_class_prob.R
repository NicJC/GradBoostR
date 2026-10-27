#' Random Forest Classification Probabilities
#'
#' Predict class probabilities from a fitted random forest.
#'
#' @param X Numeric matrix of predictors.
#' @param forest_list Forest returned by rf_class_fit_fast().
#'
#' @export
rf_class_predict_prob_fast <- function(
    X,
    forest_list
) {
  .Call(
    `_GradBoostR_rf_class_predict_prob_fast`,
    X,
    forest_list
  )
}

