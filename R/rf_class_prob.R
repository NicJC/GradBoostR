#' Predict Positive-Class Probabilities
#'
#' Generates estimated probabilities for class `1` from a binary random
#' forest fitted with [rf_class_fit_fast()].
#'
#' @param X A numeric matrix containing predictor values. The columns must
#'   correspond, in number and order, to the predictors used to fit
#'   `forest_list`.
#' @param forest_list A fitted binary-classification forest returned by
#'   [rf_class_fit_fast()].
#'
#' @return A numeric vector of length `nrow(X)`. Each element is the estimated
#'   probability that the corresponding observation belongs to class `1`.
#'   Returned values are between zero and one.
#'
#' @examples
#' set.seed(872)
#'
#' X <- cbind(
#'   x1 = rnorm(30),
#'   x2 = rnorm(30)
#' )
#'
#' y <- as.integer(
#'   X[, "x1"] + X[, "x2"] > 0
#' )
#'
#' forest <- rf_class_fit_fast(
#'   X,
#'   y,
#'   n_trees = 10,
#'   max_depth = 3
#' )
#'
#' probabilities <- rf_class_predict_prob_fast(
#'   X,
#'   forest
#' )
#'
#' head(probabilities)
#'
#' stopifnot(
#'   is.numeric(probabilities),
#'   is.null(dim(probabilities)),
#'   length(probabilities) == nrow(X),
#'   all(is.finite(probabilities)),
#'   all(probabilities >= 0),
#'   all(probabilities <= 1)
#' )
#'
#' @export
rf_class_predict_prob_fast <- function(
    X,
    forest_list
) {
  probabilities <- .Call(
    `_GradBoostR_rf_class_predict_prob_fast`,
    X,
    forest_list
  )

  as.vector(probabilities)
}
