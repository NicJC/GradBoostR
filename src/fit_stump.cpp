#include <Rcpp.h>
#include <algorithm>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export]]
List fit_stump(NumericMatrix X, NumericVector residuals) {
  int n = X.nrow();
  int p = X.ncol();

  double best_loss       = R_PosInf;
  int    best_feature    = 0;
  double best_threshold  = 0.0;
  double best_left_pred  = 0.0;
  double best_right_pred = 0.0;

  // Work buffers reused per feature
  std::vector<int> idx(n);
  std::vector<double> vals(n);
  std::vector<double> prefix(n);

  for (int j = 0; j < p; ++j) {
    // 1) Build index + values for feature j
    for (int i = 0; i < n; ++i) {
      idx[i]  = i;
      vals[i] = X(i, j);
    }

    // 2) Sort indices by feature value
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return vals[a] < vals[b]; });

    // 3) Prefix sums of residuals in sorted order
    prefix[0] = residuals[idx[0]];
    for (int i = 1; i < n; ++i)
      prefix[i] = prefix[i - 1] + residuals[idx[i]];

    // 4) Try splits between sorted points
    for (int i = 0; i < n - 1; ++i) {
      int n_left  = i + 1;
      int n_right = n - n_left;

      if (n_left == 0 || n_right == 0) continue;

      double sum_left  = prefix[i];
      double sum_right = prefix[n - 1] - prefix[i];

      double left_pred  = sum_left  / n_left;
      double right_pred = sum_right / n_right;

      // Threshold between current and next value
      double thr = 0.5 * (vals[idx[i]] + vals[idx[i + 1]]);

      // 5) Compute loss in O(n) using sorted partitions
      double loss = 0.0;

      // left side
      for (int k = 0; k <= i; ++k) {
        double diff = residuals[idx[k]] - left_pred;
        loss += diff * diff;
      }
      // right side
      for (int k = i + 1; k < n; ++k) {
        double diff = residuals[idx[k]] - right_pred;
        loss += diff * diff;
      }

      if (loss < best_loss) {
        best_loss       = loss;
        best_feature    = j;
        best_threshold  = thr;
        best_left_pred  = left_pred;
        best_right_pred = right_pred;
      }
    }
  }

  return List::create(
    _["feature"]   = best_feature,
    _["threshold"] = best_threshold,
    _["left_val"]  = best_left_pred,
    _["right_val"] = best_right_pred
  );

}
