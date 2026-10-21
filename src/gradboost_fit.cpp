#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// forward declaration
Rcpp::List fit_stump(NumericMatrix X, NumericVector residuals);

// [[Rcpp::export]]
List grad_boost_fit(const arma::mat& X,
                    const arma::vec& y,
                    double learning_rate,
                    int n_trees) {

  int n = X.n_rows;

  // Convert X to NumericMatrix for fit_stump
  NumericMatrix Xr(as<NumericMatrix>(wrap(X)));

  // Initial prediction = mean(y)
  double init_value = mean(y);

  // Current predictions
  NumericVector preds(n, init_value);

  // Residuals
  NumericVector residuals(n);

  // List of stumps
  List stumps(n_trees);

  for (int m = 0; m < n_trees; ++m) {

    // Compute residuals
    for (int i = 0; i < n; i++) {
      residuals[i] = y[i] - preds[i];
    }

    // Fit stump to residuals
    List stump = fit_stump(Xr, residuals);

    int feature      = as<int>(stump["feature"]);
    double threshold = as<double>(stump["threshold"]);
    double left_val  = as<double>(stump["left_val"]);
    double right_val = as<double>(stump["right_val"]);

    // Update predictions
    for (int i = 0; i < n; i++) {
      if (X(i, feature) <= threshold)
        preds[i] += learning_rate * left_val;
      else
        preds[i] += learning_rate * right_val;
    }

    // Store stump
    stumps[m] = List::create(
      _["feature"]   = feature,
      _["threshold"] = threshold,
      _["left_val"]  = left_val,
      _["right_val"] = right_val
    );
  }

  return stumps;
}
