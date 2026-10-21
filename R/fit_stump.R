#' Fit a decision stump
#'
#' @param X Numeric matrix of features
#' @param y Numeric vector of labels
#' @return A list containing stump parameters
#' @export
fit_stump <- function(X, y) {
  # Temporary placeholder until you implement the real stump logic
  list(
    feature = 1L,
    threshold = mean(X[, 1]),
    left_value = mean(y[X[, 1] <= mean(X[, 1])]),
    right_value = mean(y[X[, 1] > mean(X[, 1])])
  )
}

