#include <RcppArmadillo.h>
#include "tree_utils.h"
using namespace Rcpp;

// ---------------------------------------------------------------------
// Regression split
// ---------------------------------------------------------------------

void fast_split(const arma::vec& xj,
                const arma::vec& y,
                double& best_thr,
                double& best_loss) {

  int n = xj.n_elem;
  arma::uvec idx = arma::sort_index(xj);

  arma::vec sx = xj.elem(idx);
  arma::vec sy = y.elem(idx);

  arma::vec prefix = arma::cumsum(sy);

  best_loss = arma::datum::inf;

  for (int i = 0; i < n - 1; ++i) {
    int n_left  = i + 1;
    int n_right = n - n_left;

    double sum_left  = prefix[i];
    double sum_right = prefix[n - 1] - prefix[i];

    double lp = sum_left  / n_left;
    double rp = sum_right / n_right;

    double thr = 0.5 * (sx[i] + sx[i + 1]);

    double loss = 0.0;

    for (int k = 0; k <= i; ++k) {
      double diff = sy[k] - lp;
      loss += diff * diff;
    }
    for (int k = i + 1; k < n; ++k) {
      double diff = sy[k] - rp;
      loss += diff * diff;
    }

    if (loss < best_loss) {
      best_loss = loss;
      best_thr  = thr;
    }
  }
}

// ---------------------------------------------------------------------
// Regression tree builder
// ---------------------------------------------------------------------

int build_tree(const arma::mat& X,
               const arma::vec& y,
               int depth,
               int max_depth,
               int mtry,
               std::vector<Node>& nodes) {

  Node node;
  node.feature   = -1;
  node.threshold = 0.0;
  node.value     = 0.0;
  node.left      = -1;
  node.right     = -1;
  node.is_leaf   = false;

  // leaf if too deep, too few points, or empty y
  if (depth >= max_depth || y.n_elem <= 2 || y.n_elem == 0) {
    node.is_leaf   = true;
    node.value     = (y.n_elem == 0) ? 0.0 : arma::mean(y);
    node.feature   = -1;
    node.threshold = 0.0;
    node.left      = -1;
    node.right     = -1;

    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
  }

  int p = X.n_cols;
  arma::uvec feats = arma::randperm(p).head(mtry);

  double best_loss = arma::datum::inf;
  int best_feat    = -1;
  double best_thr  = 0.0;

  for (unsigned int fi = 0; fi < feats.n_elem; ++fi) {
    int j = feats[fi];
    double thr, loss;
    fast_split(X.col(j), y, thr, loss);

    if (loss < best_loss) {
      best_loss = loss;
      best_feat = j;
      best_thr  = thr;
    }
  }

  // if no valid split found, make leaf
  if (best_feat < 0) {
    node.is_leaf   = true;
    node.value     = arma::mean(y);
    node.feature   = -1;
    node.threshold = 0.0;
    node.left      = -1;
    node.right     = -1;

    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
  }

  arma::uvec left_idx  = arma::find(X.col(best_feat) <= best_thr);
  arma::uvec right_idx = arma::find(X.col(best_feat) > best_thr);

  // if one side empty, make leaf
  if (left_idx.n_elem == 0 || right_idx.n_elem == 0) {
    node.is_leaf   = true;
    node.value     = arma::mean(y);
    node.feature   = -1;
    node.threshold = 0.0;
    node.left      = -1;
    node.right     = -1;

    nodes.push_back(node);
    return static_cast<int>(nodes.size()) - 1;
  }

  arma::mat Xl = X.rows(left_idx);
  arma::mat Xr = X.rows(right_idx);
  arma::vec yl = y.elem(left_idx);
  arma::vec yr = y.elem(right_idx);

  node.is_leaf   = false;
  node.feature   = best_feat;
  node.threshold = best_thr;
  node.value     = 0.0;
  node.left      = -1;
  node.right     = -1;

  nodes.push_back(node);
  int idx = static_cast<int>(nodes.size()) - 1;

  nodes[idx].left  = build_tree(Xl, yl, depth + 1, max_depth, mtry, nodes);
  nodes[idx].right = build_tree(Xr, yr, depth + 1, max_depth, mtry, nodes);

  return idx;
}


// ---------------------------------------------------------------------
// RF regression fit
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List rf_fit_fast(const arma::mat& X,
                       const arma::vec& y,
                       int n_trees = 200,
                       int max_depth = 5,
                       int mtry = 3) {

  std::vector< std::vector<Node> > forest(n_trees);

  int n = X.n_rows;

  for (int t = 0; t < n_trees; ++t) {
    arma::uvec idx = arma::randi<arma::uvec>(n, arma::distr_param(0, n - 1));
    if (idx.n_elem == 0) continue;

    arma::mat Xb = X.rows(idx);
    arma::vec yb = y.elem(idx);

    std::vector<Node> nodes;
    build_tree(Xb, yb, 0, max_depth, mtry, nodes);

    if (!nodes.empty())
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

// ---------------------------------------------------------------------
// Safe regression tree prediction
// ---------------------------------------------------------------------

double predict_tree_reg(const arma::rowvec& x,
                        const std::vector<Node>& nodes,
                        int idx = 0) {

  if (idx < 0 || idx >= static_cast<int>(nodes.size()))
    return 0.0;

  const Node& node = nodes[idx];

  if (node.is_leaf)
    return node.value;

  if (node.feature < 0 || node.feature >= static_cast<int>(x.n_elem))
    return node.value;

  if (node.left < 0 || node.left >= static_cast<int>(nodes.size()))
    return node.value;

  if (node.right < 0 || node.right >= static_cast<int>(nodes.size()))
    return node.value;

  if (x[node.feature] <= node.threshold)
    return predict_tree_reg(x, nodes, node.left);
  else
    return predict_tree_reg(x, nodes, node.right);
}

// ---------------------------------------------------------------------
// RF regression predict
// ---------------------------------------------------------------------

// [[Rcpp::export]]
arma::vec rf_predict_fast(const arma::mat& X, Rcpp::List forest_list) {

  int n       = X.n_rows;
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
      out[i] += predict_tree_reg(X.row(i), nodes);
  }

  out /= n_trees;
  return out;
}
