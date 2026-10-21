#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector gbm_predict(const NumericMatrix& X,
                          const List& stumps,
                          double learning_rate,
                          double init_value) {

  int n = X.nrow();
  NumericVector preds(n, init_value);

  for (int s = 0; s < stumps.size(); s++) {
    List st = stumps[s];

    int feature     = as<int>(st["feature"]);
    double threshold = as<double>(st["threshold"]);
    double left_val  = as<double>(st["left_val"]);
    double right_val = as<double>(st["right_val"]);

    for (int i = 0; i < n; i++) {
      if (X(i, feature) <= threshold)
        preds[i] += learning_rate * left_val;
      else
        preds[i] += learning_rate * right_val;
    }
  }

  return preds;
}
