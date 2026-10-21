#' Fast Random Forest (Classification) Training
#'
#' Internal function used by GradBoostR to train a classification random forest.
#' Not intended for direct user use.
#'
#' @param X Numeric matrix of features.
#' @param y Numeric vector of class labels.
#' @param n_trees Number of trees.
#' @param max_depth Maximum tree depth.
#' @param mtry Number of variables sampled at each split.
#' @param use_gini Logical; use Gini impurity.
#'
#' @return A list representing the trained forest.
#' @keywords internal
#' @export
rf_class_fit_fast <- function(X, y, n_trees = 100, max_depth = 5, mtry = 3, use_gini = TRUE) {
  .Call(`_GradBoostR_rf_class_fit_fast`, X, y, n_trees, max_depth, mtry, use_gini)
}

#' Fast Random Forest (Classification) Prediction
#'
#' Internal prediction function for classification random forests.
#'
#' @param X Numeric matrix of features.
#' @param forest_list List of trained trees.
#'
#' @return A vector of predicted class labels.
#' @keywords internal
#' @export
rf_class_predict_fast <- function(X, forest_list) {
  .Call(`_GradBoostR_rf_class_predict_fast`, X, forest_list)
}

#' Fast Random Forest (Regression) Training
#'
#' Internal function used by GradBoostR to train a regression random forest.
#'
#' @param X Numeric matrix of features.
#' @param y Numeric vector of responses.
#' @param n_trees Number of trees.
#' @param max_depth Maximum tree depth.
#' @param mtry Number of variables sampled at each split.
#'
#' @return A list representing the trained forest.
#' @keywords internal
#' @export
rf_fit_fast <- function(X, y, n_trees = 200, max_depth = 5, mtry = 3) {
  .Call(`_GradBoostR_rf_fit_fast`, X, y, n_trees, max_depth, mtry)
}

#' Fast Random Forest (Regression) Prediction
#'
#' Internal prediction function for regression random forests.
#'
#' @param X Numeric matrix of features.
#' @param forest_list List of trained trees.
#'
#' @return A numeric vector of predictions.
#' @keywords internal
#' @export
rf_predict_fast <- function(X, forest_list) {
  .Call(`_GradBoostR_rf_predict_fast`, X, forest_list)
}

#' Fast Random Forest Classification (Hard Labels)
#'
#' Internal function returning predicted class labels.
#'
#' @param X Numeric matrix of features.
#' @param forest_list List of trained trees.
#'
#' @return Integer vector of predicted classes.
#' @keywords internal
#' @export
rf_predict_fast_class <- function(X, forest_list) {
  .Call(`_GradBoostR_rf_predict_fast_class`, X, forest_list)
}

#' Fast Random Forest Classification (Probabilities)
#'
#' Internal function returning class probabilities.
#'
#' @param X Numeric matrix of features.
#' @param forest_list List of trained trees.
#' @param n_classes Number of classes.
#'
#' @return Numeric matrix of class probabilities.
#' @keywords internal
#' @export
rf_predict_fast_class_prob <- function(X, forest_list, n_classes) {
  .Call(`_GradBoostR_rf_predict_fast_class_prob`, X, forest_list, n_classes)
}
