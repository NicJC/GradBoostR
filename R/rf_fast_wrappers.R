#' Fit a Fast Classification Random Forest
#'
#' Fits a random forest for classification using the native GradBoostR
#' implementation.
#'
#' @param X A numeric matrix containing predictor values. Rows represent
#'   observations and columns represent predictors.
#' @param y An integer or numeric vector containing an encoded class label
#'   for each row of `X`.
#' @param n_trees A positive integer giving the number of trees to fit.
#' @param max_depth A positive integer giving the maximum depth of each tree.
#' @param mtry A positive integer giving the number of predictors considered
#'   at each split. It must not exceed `ncol(X)`.
#' @param use_gini A logical value indicating whether Gini impurity is used
#'   to evaluate candidate splits.
#'
#' @return A list with one element per fitted classification tree. Each
#'   element stores the structure of one tree in the random forest. The
#'   length of the list equals `n_trees`. The returned forest can be passed
#'   to [rf_class_predict_fast()], [rf_class_predict_prob_fast()],
#'   [rf_predict_fast_class()], or [rf_predict_fast_class_prob()] to generate
#'   predictions.
#'
#' @export
#'
#' @examples
#' set.seed(983)
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
#'   n_trees = 5L,
#'   max_depth = 3L,
#'   mtry = 1L,
#'   use_gini = TRUE
#' )
#'
#' length(forest)
#'
#' stopifnot(
#'   is.list(forest),
#'   length(forest) == 5L
#' )
rf_class_fit_fast <- function(
    X,
    y,
    n_trees = 100,
    max_depth = 5,
    mtry = 3,
    use_gini = TRUE
) {
  .Call(
    `_GradBoostR_rf_class_fit_fast`,
    X,
    y,
    n_trees,
    max_depth,
    mtry,
    use_gini
  )
}


#' Predict Classes from a Fast Classification Random Forest
#'
#' Generates binary class predictions from a classification forest fitted
#' with [rf_class_fit_fast()].
#'
#' @param X A numeric matrix containing predictor values. The columns must
#'   correspond, in number and order, to those used to fit `forest_list`.
#' @param forest_list A fitted classification forest returned by
#'   [rf_class_fit_fast()].
#'
#' @return A numeric vector of length `nrow(X)`. Each element is the predicted
#'   binary class, encoded as `0` or `1`, for the corresponding observation.
#'
#' @export
#'
#' @examples
#' set.seed(467)
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
#'   n_trees = 5L,
#'   max_depth = 3L,
#'   mtry = 1L,
#'   use_gini = TRUE
#' )
#'
#' predicted_class <- rf_class_predict_fast(
#'   X,
#'   forest
#' )
#'
#' head(predicted_class)
#'
#' stopifnot(
#'   is.numeric(predicted_class),
#'   is.null(dim(predicted_class)),
#'   length(predicted_class) == nrow(X),
#'   all(predicted_class %in% c(0, 1))
#' )
rf_class_predict_fast <- function(
    X,
    forest_list
) {
  predictions <- .Call(
    `_GradBoostR_rf_class_predict_fast`,
    X,
    forest_list
  )

  as.vector(predictions)
}


#' Fit a Fast Regression Random Forest
#'
#' Fits a random forest for a numeric response using the native GradBoostR
#' implementation.
#'
#' @param X A numeric matrix containing predictor values. Rows represent
#'   observations and columns represent predictors.
#' @param y A numeric response vector containing one value for each row
#'   of `X`.
#' @param n_trees A positive integer giving the number of trees to fit.
#' @param max_depth A positive integer giving the maximum depth of each tree.
#' @param mtry A positive integer giving the number of predictors considered
#'   at each split. It must not exceed `ncol(X)`.
#'
#' @return A list with one element per fitted regression tree. Each element
#'   stores the structure of one tree in the random forest. The length of the
#'   list equals `n_trees`. The returned forest can be passed to
#'   [rf_predict_fast()] to generate numeric response predictions.
#'
#' @export
#'
#' @examples
#' set.seed(528)
#'
#' X <- cbind(
#'   x1 = rnorm(30),
#'   x2 = rnorm(30)
#' )
#'
#' y <- (
#'   2 * X[, "x1"] -
#'     X[, "x2"] +
#'     rnorm(30, sd = 0.2)
#' )
#'
#' forest <- rf_fit_fast(
#'   X,
#'   y,
#'   n_trees = 5L,
#'   max_depth = 3L,
#'   mtry = 1L
#' )
#'
#' length(forest)
#'
#' stopifnot(
#'   is.list(forest),
#'   length(forest) == 5L
#' )
rf_fit_fast <- function(
    X,
    y,
    n_trees = 200,
    max_depth = 5,
    mtry = 3
) {
  .Call(
    `_GradBoostR_rf_fit_fast`,
    X,
    y,
    n_trees,
    max_depth,
    mtry
  )
}


#' Predict from a Fast Regression Random Forest
#'
#' Generates numeric response predictions from a regression forest fitted
#' with [rf_fit_fast()].
#'
#' @param X A numeric matrix containing predictor values. The columns must
#'   correspond, in number and order, to those used to fit `forest_list`.
#' @param forest_list A fitted regression forest returned by
#'   [rf_fit_fast()].
#'
#' @return A numeric vector of length `nrow(X)`. Each element is the predicted
#'   numeric response for the corresponding observation.
#'
#' @export
#'
#' @examples
#' set.seed(672)
#'
#' X <- cbind(
#'   x1 = rnorm(30),
#'   x2 = rnorm(30)
#' )
#'
#' y <- (
#'   2 * X[, "x1"] -
#'     X[, "x2"] +
#'     rnorm(30, sd = 0.2)
#' )
#'
#' forest <- rf_fit_fast(
#'   X,
#'   y,
#'   n_trees = 5L,
#'   max_depth = 3L,
#'   mtry = 1L
#' )
#'
#' predictions <- rf_predict_fast(
#'   X,
#'   forest
#' )
#'
#' head(predictions)
#'
#' stopifnot(
#'   is.numeric(predictions),
#'   is.null(dim(predictions)),
#'   length(predictions) == nrow(X),
#'   all(is.finite(predictions))
#' )
rf_predict_fast <- function(
    X,
    forest_list
) {
  predictions <- .Call(
    `_GradBoostR_rf_predict_fast`,
    X,
    forest_list
  )

  as.vector(predictions)
}


#' Predict Hard Classes from a Fast Classification Forest
#'
#' Generates hard-class predictions from a classification forest fitted with
#' [rf_class_fit_fast()]. This is a lower-level prediction interface. Most
#' users should use [rf_class_predict_fast()].
#'
#' @param X A numeric matrix containing predictor values. The columns must
#'   correspond, in number and order, to those used to fit `forest_list`.
#' @param forest_list A fitted classification forest returned by
#'   [rf_class_fit_fast()].
#'
#' @return An integer vector of length `nrow(X)`. Each element is the predicted
#'   class index for the corresponding observation. For a binary model fitted
#'   with classes encoded as `0` and `1`, the returned values are `0L` or
#'   `1L`.
#'
#' @export
#'
#' @examples
#' set.seed(614)
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
#'   n_trees = 5L,
#'   max_depth = 3L,
#'   mtry = 1L,
#'   use_gini = TRUE
#' )
#'
#' predictions <- rf_predict_fast_class(
#'   X,
#'   forest
#' )
#'
#' head(predictions)
#'
#' stopifnot(
#'   is.integer(predictions),
#'   is.null(dim(predictions)),
#'   length(predictions) == nrow(X),
#'   all(predictions %in% c(0L, 1L))
#' )
rf_predict_fast_class <- function(
    X,
    forest_list
) {
  .Call(
    `_GradBoostR_rf_predict_fast_class`,
    X,
    forest_list
  )
}


#' Predict Class Probabilities from a Fast Classification Forest
#'
#' Generates estimated class probabilities from a classification forest
#' fitted with [rf_class_fit_fast()]. This is a lower-level probability
#' interface. For binary models where only the positive-class probability is
#' required, use [rf_class_predict_prob_fast()].
#'
#' @param X A numeric matrix containing predictor values. The columns must
#'   correspond, in number and order, to those used to fit `forest_list`.
#' @param forest_list A fitted classification forest returned by
#'   [rf_class_fit_fast()].
#' @param n_classes A positive integer giving the number of response classes.
#'
#' @return A numeric matrix with `nrow(X)` rows and `n_classes` columns.
#'   Each row contains the estimated class probabilities for one observation,
#'   and each column corresponds to one encoded response class.
#'
#' @export
#'
#' @examples
#' set.seed(358)
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
#'   n_trees = 5L,
#'   max_depth = 3L,
#'   mtry = 1L,
#'   use_gini = TRUE
#' )
#'
#' probabilities <- rf_predict_fast_class_prob(
#'   X,
#'   forest,
#'   n_classes = 2L
#' )
#'
#' head(probabilities)
#'
#' stopifnot(
#'   is.matrix(probabilities),
#'   identical(
#'     dim(probabilities),
#'     c(nrow(X), 2L)
#'   ),
#'   is.numeric(probabilities),
#'   all(is.finite(probabilities)),
#'   all(probabilities >= 0),
#'   all(probabilities <= 1)
#' )
rf_predict_fast_class_prob <- function(
    X,
    forest_list,
    n_classes
) {
  .Call(
    `_GradBoostR_rf_predict_fast_class_prob`,
    X,
    forest_list,
    n_classes
  )
}
