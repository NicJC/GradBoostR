#include <Rcpp.h>
#include "tree_utils.h"
using namespace Rcpp;


// [[Rcpp::export]]
NumericVector rf_predict_fast_class(const NumericMatrix& X,
                                    const List& forest_list) {

  int n = X.nrow();
  int T = forest_list.size();
  NumericVector preds(n);

  for (int i = 0; i < n; i++) {
    double sum = 0.0;

    for (int t = 0; t < T; t++) {
      List tree = forest_list[t];
      std::vector<Node> nodes(tree.size());

      for (int j = 0; j < tree.size(); j++) {
        List nd = tree[j];
        nodes[j].feature   = as<int>(nd["feature"]);
        nodes[j].threshold = as<double>(nd["threshold"]);
        nodes[j].is_leaf   = as<bool>(nd["is_leaf"]);
        nodes[j].value     = as<double>(nd["value"]);
        nodes[j].left      = as<int>(nd["left"]);
        nodes[j].right     = as<int>(nd["right"]);
      }

      sum += traverse_tree(X, i, nodes);
    }

    preds[i] = sum / T;
  }

  return preds;
}
