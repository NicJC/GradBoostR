#' Fast Random Forest Fit
#'
#' @param X Numeric matrix of predictors
#' @param y Numeric vector of responses
#' @param n_trees Number of trees
#' @param max_depth Maximum depth of each tree
#' @param mtry Number of features sampled at each split
#'
#' @return A list representing the forest
#' @export
rf_fit_fast <- function(X, y, n_trees = 200, max_depth = 5, mtry = 3) {
  .Call(`_GradBoostR_rf_fit_fast`, X, y, n_trees, max_depth, mtry)
}

