#include <RcppArmadillo.h>
#include "tree_utils.h"
using namespace Rcpp;
using Rcpp::Rcout;

// Helper impurities
double gini_impurity(const arma::vec& y) {
  int n = y.n_elem;
  arma::vec counts = arma::conv_to<arma::vec>::from(
    arma::hist(y, arma::unique(y))
  );
  arma::vec p = counts / n;
  return 1.0 - arma::sum(p % p);
}

double entropy_impurity(const arma::vec& y) {
  int n = y.n_elem;
  arma::vec counts = arma::conv_to<arma::vec>::from(
    arma::hist(y, arma::unique(y))
  );
  arma::vec p = counts / n;
  arma::vec logp = arma::log(p + 1e-12);
  return -arma::sum(p % logp);
}

// Find best split for a single feature
void fast_split_class(const arma::vec& xj,
                      const arma::vec& y,
                      double& best_thr,
                      double& best_imp,
                      bool use_gini) {

  int n = xj.n_elem;
  arma::uvec idx = arma::sort_index(xj);

  arma::vec sx = xj.elem(idx);
  arma::vec sy = y.elem(idx);

  best_imp = arma::datum::inf;

  for (int i = 0; i < n - 1; ++i) {
    arma::uvec left_idx  = idx.head(i + 1);
    arma::uvec right_idx = idx.tail(n - i - 1);

    double imp_left  = use_gini ? gini_impurity(sy.elem(left_idx))
      : entropy_impurity(sy.elem(left_idx));

    double imp_right = use_gini ? gini_impurity(sy.elem(right_idx))
      : entropy_impurity(sy.elem(right_idx));

    double imp = (imp_left * (i + 1) + imp_right * (n - i - 1)) / n;

    double thr = 0.5 * (sx[i] + sx[i + 1]);

    if (imp < best_imp) {
      best_imp = imp;
      best_thr = thr;
    }
  }
}

// Build a single decision tree (recursive)
int build_tree_class(const arma::mat& X,
                     const arma::vec& y,
                     int depth,
                     int max_depth,
                     int mtry,
                     bool use_gini,
                     std::vector<Node>& nodes) {

  Node node;

  if (depth >= max_depth || y.n_elem <= 2) {
    node.is_leaf = true;
    node.value = (y.n_elem == 0) ? 0.0 : arma::as_scalar(arma::median(y)); // class prediction
    node.feature = -1;
    node.threshold = 0.0;
    node.left = node.right = -1;

    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
  }

  int p = X.n_cols;
  arma::uvec feats = arma::randperm(p).head(mtry);

  double best_imp = arma::datum::inf;
  int best_feat = -1;
  double best_thr = 0.0;

  for (unsigned int fi = 0; fi < feats.n_elem; ++fi) {
    int j = feats[fi];
    double thr, imp;
    fast_split_class(X.col(j), y, thr, imp, use_gini);

    if (imp < best_imp) {
      best_imp = imp;
      best_feat = j;
      best_thr = thr;
    }
  }

  arma::uvec left_idx  = arma::find(X.col(best_feat) <= best_thr);
  arma::uvec right_idx = arma::find(X.col(best_feat) > best_thr);

  arma::mat Xl = X.rows(left_idx);
  arma::mat Xr = X.rows(right_idx);
  arma::vec yl = y.elem(left_idx);
  arma::vec yr = y.elem(right_idx);

  node.is_leaf = false;
  node.feature = best_feat;
  node.threshold = best_thr;
  node.value = 0.0;
  node.left = -1;
  node.right = -1;

  nodes.push_back(node);
  int idx = static_cast<int>(nodes.size()) - 1;

  nodes[idx].left  = build_tree_class(Xl, yl, depth + 1, max_depth, mtry, use_gini, nodes);
  nodes[idx].right = build_tree_class(Xr, yr, depth + 1, max_depth, mtry, use_gini, nodes);

  return idx;
}

// [[Rcpp::export]]
Rcpp::List rf_class_fit_fast(const arma::mat& X,
                             const arma::vec& y,
                             int n_trees = 100,
                             int max_depth = 5,
                             int mtry = 3,
                             bool use_gini = true) {

  Rcout << "ENTER rf_class_fit_fast n_trees=" << n_trees
        << " max_depth=" << max_depth
        << " mtry=" << mtry << "\n";

  std::vector<std::vector<Node>> forest(n_trees);

  Rcout << "BEFORE build loop n_trees=" << n_trees
        << " nrows=" << X.n_rows
        << " ncols=" << X.n_cols << "\n";

  for (int t = 0; t < n_trees; ++t) {
    int n = X.n_rows;

    arma::uvec idx = arma::randi<arma::uvec>(n, arma::distr_param(0, n - 1));
    if (idx.n_elem == 0) continue;   // skip empty bootstrap

    arma::mat Xb = X.rows(idx);
    arma::vec yb = y.elem(idx);

    std::vector<Node> nodes;
    build_tree_class(Xb, yb, 0, max_depth, mtry, use_gini, nodes);

    if (nodes.empty()) continue;
    forest[t] = nodes;
  }

  Rcpp::List out_forest(n_trees);

  for (int t = 0; t < n_trees; ++t) {
    const auto& tree = forest[t];
    Rcpp::List tree_list(tree.size());

    for (std::size_t i = 0; i < tree.size(); ++i) {
      const Node& n = tree[i];

      tree_list[i] = Rcpp::List::create(
        Rcpp::Named("feature")   = n.feature,
        Rcpp::Named("threshold") = n.threshold,
        Rcpp::Named("is_leaf")   = n.is_leaf,
        Rcpp::Named("value")     = n.value,
        Rcpp::Named("left")      = n.left,
        Rcpp::Named("right")     = n.right
      );
    }

    out_forest[t] = tree_list;
  }

  return out_forest;
}

// Predict helper (recursive)
double predict_tree_class(const arma::rowvec& x,
                          const std::vector<Node>& nodes,
                          int idx = 0) {

  // Validate index
  if (idx < 0 || idx >= static_cast<int>(nodes.size()))
    return 0.0;

  const Node& node = nodes[idx];

  // Leaf node
  if (node.is_leaf)
    return node.value;

  // Validate feature index
  if (node.feature < 0 || node.feature >= static_cast<int>(x.n_elem))
    return node.value;

  // Validate children
  if (node.left < 0 || node.left >= static_cast<int>(nodes.size()))
    return node.value;

  if (node.right < 0 || node.right >= static_cast<int>(nodes.size()))
    return node.value;

  // Recurse safely
  if (x[node.feature] <= node.threshold)
    return predict_tree_class(x, nodes, node.left);
  else
    return predict_tree_class(x, nodes, node.right);
}

// [[Rcpp::export]]
arma::vec rf_class_predict_fast(const arma::mat& X, Rcpp::List forest_list) {

  Rcout << "ENTER rf_class_predict_fast nrow=" << X.n_rows
        << " ncol=" << X.n_cols << "\n";

  int n = X.n_rows;
  int n_trees = forest_list.size();

  arma::vec out(n, arma::fill::zeros);

  for (int t = 0; t < n_trees; ++t) {
    Rcpp::List tree = forest_list[t];
    std::vector<Node> nodes(tree.size());

    for (int i = 0; i < tree.size(); ++i) {
      Rcpp::List n = tree[i];

      nodes[i].feature   = Rcpp::as<int>(n["feature"]);
      nodes[i].threshold = Rcpp::as<double>(n["threshold"]);
      nodes[i].is_leaf   = Rcpp::as<bool>(n["is_leaf"]);
      nodes[i].value     = Rcpp::as<double>(n["value"]);
      nodes[i].left      = Rcpp::as<int>(n["left"]);
      nodes[i].right     = Rcpp::as<int>(n["right"]);
    }

    for (int i = 0; i < n; ++i)
      out[i] += predict_tree_class(X.row(i), nodes);
  }

  // majority vote
  return arma::round(out / n_trees);
}
