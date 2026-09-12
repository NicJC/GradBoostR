#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector nn_predict_arma(const arma::mat& X,
                              const arma::mat& W1,
                              const arma::vec& b1,
                              const arma::mat& W2,
                              const arma::vec& b2,
                              bool classification = false) {

  int n = X.n_rows;

  arma::mat Z1 = X * W1;
  Z1.each_row() += b1.t();
  arma::mat A1 = arma::clamp(Z1, 0.0, arma::datum::inf); // ReLU

  arma::mat Z2 = A1 * W2;
  Z2.each_row() += b2.t();

  NumericVector out(n);

  if (!classification) {
    for (int i = 0; i < n; i++)
      out[i] = Z2(i, 0);
  } else {
    for (int i = 0; i < n; i++)
      out[i] = 1.0 / (1.0 + std::exp(-Z2(i, 0))); // sigmoid
  }

  return out;
}
