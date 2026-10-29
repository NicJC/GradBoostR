validate_regression_prediction <- function(
    prediction,
    expected_length,
    method
) {
  prediction <- as.vector(
    prediction
  )

  if (!is.numeric(prediction)) {
    stop(
      sprintf(
        "%s prediction did not return numeric values.",
        method
      ),
      call. = FALSE
    )
  }

  if (length(prediction) != expected_length) {
    stop(
      sprintf(
        paste0(
          "%s prediction returned %d values for ",
          "%d observations."
        ),
        method,
        length(prediction),
        expected_length
      ),
      call. = FALSE
    )
  }

  if (any(!is.finite(prediction))) {
    stop(
      sprintf(
        "%s prediction returned non-finite values.",
        method
      ),
      call. = FALSE
    )
  }

  prediction
}
