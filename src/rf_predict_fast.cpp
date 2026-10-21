#include <RcppArmadillo.h>
#include "tree_utils.h"
using namespace Rcpp;

// ---------------------------------------------------------------------
// Predict class label for one tree
// ---------------------------------------------------------------------

int predict_tree_class_label(const arma::rowvec& x,
                             const std::vector<Node>& nodes,
                             int idx = 0) {

  if (nodes.empty())
    return 0;

  if (idx < 0 || idx >= static_cast<int>(nodes.size()))
    return 0;

  const Node& node = nodes[idx];

  if (node.is_leaf)
    return static_cast<int>(node.value);

  if (node.feature < 0 || node.feature >= static_cast<int>(x.n_elem))
    return static_cast<int>(node.value);

  if (node.left < 0 || node.left >= static_cast<int>(nodes.size()))
    return static_cast<int>(node.value);

  if (node.right < 0 || node.right >= static_cast<int>(nodes.size()))
    return static_cast<int>(node.value);

  if (x[node.feature] <= node.threshold)
    return predict_tree_class_label(x, nodes, node.left);
  else
    return predict_tree_class_label(x, nodes, node.right);
}


// ---------------------------------------------------------------------
// Majority vote (response)
// ---------------------------------------------------------------------

// [[Rcpp::export]]
IntegerVector rf_predict_fast_class(const arma::mat& X,
                                    const List& forest_list) {

  int n = X.n_rows;
  int T = forest_list.size();

  IntegerVector preds(n);

  for (int i = 0; i < n; i++) {

    std::map<int,int> counts;

    for (int t = 0; t < T; t++) {
      List tree = forest_list[t];
      int tree_size = tree.size();
      if (tree_size == 0)
        continue;

      std::vector<Node> nodes(tree_size);

      for (int j = 0; j < tree_size; j++) {
        List nd = tree[j];
        nodes[j].feature   = as<int>(nd["feature"]);
        nodes[j].threshold = as<double>(nd["threshold"]);
        nodes[j].is_leaf   = as<bool>(nd["is_leaf"]);
        nodes[j].value     = as<double>(nd["value"]);
        nodes[j].left      = as<int>(nd["left"]);
        nodes[j].right     = as<int>(nd["right"]);
      }

      int cls = predict_tree_class_label(X.row(i), nodes);
      counts[cls]++;
    }

    int best_class = 0;
    int best_count = -1;

    for (auto& kv : counts) {
      if (kv.second > best_count) {
        best_class = kv.first;
        best_count = kv.second;
      }
    }

    preds[i] = best_class;
  }

  return preds;
}


// ---------------------------------------------------------------------
// Probability output (prob)
// ---------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix rf_predict_fast_class_prob(const arma::mat& X,
                                         const List& forest_list,
                                         int n_classes) {

  int n = X.n_rows;
  int T = forest_list.size();

  NumericMatrix probs(n, n_classes);

  for (int i = 0; i < n; i++) {

    std::vector<int> counts(n_classes, 0);

    for (int t = 0; t < T; t++) {
      List tree = forest_list[t];
      int tree_size = tree.size();
      if (tree_size == 0)
        continue;

      std::vector<Node> nodes(tree_size);

      for (int j = 0; j < tree_size; j++) {
        List nd = tree[j];
        nodes[j].feature   = as<int>(nd["feature"]);
        nodes[j].threshold = as<double>(nd["threshold"]);
        nodes[j].is_leaf   = as<bool>(nd["is_leaf"]);
        nodes[j].value     = as<double>(nd["value"]);
        nodes[j].left      = as<int>(nd["left"]);
        nodes[j].right     = as<int>(nd["right"]);
      }

      int cls = predict_tree_class_label(X.row(i), nodes);
      if (cls >= 0 && cls < n_classes)
        counts[cls]++;
    }

    for (int c = 0; c < n_classes; c++)
      probs(i, c) = T > 0 ? static_cast<double>(counts[c]) / T : 0.0;
  }

  return probs;
}

