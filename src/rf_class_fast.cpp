#include <RcppArmadillo.h>
#include "tree_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp20)]]

using namespace Rcpp;

// -----------------------------------------------------------------------------
// Impurity functions
// -----------------------------------------------------------------------------

double gini_impurity(const arma::vec& y) {

  if (y.n_elem == 0) {
    return 0.0;
  }

  arma::vec classes = arma::unique(y);
  double squared_probability_sum = 0.0;

  for (arma::uword i = 0; i < classes.n_elem; ++i) {

    double count = static_cast<double>(
      arma::accu(y == classes[i])
    );

    double probability =
      count / static_cast<double>(y.n_elem);

    squared_probability_sum +=
      probability * probability;
  }

  return 1.0 - squared_probability_sum;
}


double entropy_impurity(const arma::vec& y) {

  if (y.n_elem == 0) {
    return 0.0;
  }

  arma::vec classes = arma::unique(y);
  double entropy = 0.0;

  for (arma::uword i = 0; i < classes.n_elem; ++i) {

    double count = static_cast<double>(
      arma::accu(y == classes[i])
    );

    double probability =
      count / static_cast<double>(y.n_elem);

    if (probability > 0.0) {
      entropy -= probability * std::log(probability);
    }
  }

  return entropy;
}


// -----------------------------------------------------------------------------
// Create a classification leaf
// -----------------------------------------------------------------------------

Node make_class_leaf(const arma::vec& y) {

  Node node;

  node.feature = -1;
  node.threshold = 0.0;
  node.is_leaf = true;

  // For binary labels encoded as 0/1, mean(y) is the estimated
  // probability of class 1 within the leaf.
  node.value =
    y.n_elem == 0
  ? 0.0
  : arma::mean(y);

  node.left = -1;
  node.right = -1;

  return node;
}


// -----------------------------------------------------------------------------
// Find the best split for one feature
// -----------------------------------------------------------------------------

void fast_split_class(
    const arma::vec& xj,
    const arma::vec& y,
    double& best_threshold,
    double& best_impurity,
    bool use_gini
) {

  const arma::uword n = xj.n_elem;

  best_threshold = 0.0;
  best_impurity = arma::datum::inf;

  if (
      n < 2 ||
        y.n_elem != n
  ) {
    return;
  }

  arma::uvec order = arma::sort_index(xj);

  arma::vec sorted_x = xj.elem(order);
  arma::vec sorted_y = y.elem(order);

  for (arma::uword i = 0; i < n - 1; ++i) {

    // A threshold cannot separate equal adjacent feature values.
    if (sorted_x[i] == sorted_x[i + 1]) {
      continue;
    }

    arma::vec left_y =
      sorted_y.head(i + 1);

    arma::vec right_y =
      sorted_y.tail(n - i - 1);

    double left_impurity =
      use_gini
      ? gini_impurity(left_y)
        : entropy_impurity(left_y);

    double right_impurity =
      use_gini
      ? gini_impurity(right_y)
        : entropy_impurity(right_y);

    double left_weight =
      static_cast<double>(left_y.n_elem) /
        static_cast<double>(n);

    double right_weight =
      static_cast<double>(right_y.n_elem) /
        static_cast<double>(n);

    double weighted_impurity =
      left_weight * left_impurity +
      right_weight * right_impurity;

    double threshold =
      0.5 * (sorted_x[i] + sorted_x[i + 1]);

    if (weighted_impurity < best_impurity) {
      best_impurity = weighted_impurity;
      best_threshold = threshold;
    }
  }
}


// -----------------------------------------------------------------------------
// Build one classification tree recursively
// -----------------------------------------------------------------------------

int build_tree_class(
    const arma::mat& X,
    const arma::vec& y,
    int depth,
    int max_depth,
    int mtry,
    bool use_gini,
    std::vector<Node>& nodes
) {

  if (
      X.n_rows == 0 ||
        X.n_cols == 0 ||
        y.n_elem == 0 ||
        X.n_rows != y.n_elem ||
        depth >= max_depth ||
        y.n_elem <= 2 ||
        arma::all(y == y[0])
  ) {

    nodes.push_back(
      make_class_leaf(y)
    );

    return static_cast<int>(nodes.size()) - 1;
  }

  const int p =
    static_cast<int>(X.n_cols);

  int effective_mtry = mtry;

  if (effective_mtry <= 0) {

    effective_mtry = std::max(
      1,
      static_cast<int>(
        std::floor(
          std::sqrt(
            static_cast<double>(p)
          )
        )
      )
    );
  }

  // Critical fix: mtry must never exceed the number of columns.
  effective_mtry = std::max(
    1,
    std::min(effective_mtry, p)
  );

  arma::uvec features =
    arma::randperm(
      static_cast<arma::uword>(p)
    ).head(
        static_cast<arma::uword>(effective_mtry)
    );

  double best_impurity = arma::datum::inf;
  double best_threshold = 0.0;
  int best_feature = -1;

  for (
      arma::uword feature_position = 0;
      feature_position < features.n_elem;
      ++feature_position
  ) {

    int feature =
      static_cast<int>(
        features[feature_position]
      );

    double threshold = 0.0;
    double impurity = arma::datum::inf;

    fast_split_class(
      X.col(feature),
      y,
      threshold,
      impurity,
      use_gini
    );

    if (
        std::isfinite(impurity) &&
          impurity < best_impurity
    ) {

      best_impurity = impurity;
      best_feature = feature;
      best_threshold = threshold;
    }
  }

  // No valid split was found.
  if (best_feature < 0) {

    nodes.push_back(
      make_class_leaf(y)
    );

    return static_cast<int>(nodes.size()) - 1;
  }

  arma::uvec left_indices =
    arma::find(
      X.col(best_feature) <= best_threshold
    );

  arma::uvec right_indices =
    arma::find(
      X.col(best_feature) > best_threshold
    );

  // Prevent recursive creation of empty child nodes.
  if (
      left_indices.n_elem == 0 ||
        right_indices.n_elem == 0
  ) {

    nodes.push_back(
      make_class_leaf(y)
    );

    return static_cast<int>(nodes.size()) - 1;
  }

  Node node;

  node.feature = best_feature;
  node.threshold = best_threshold;
  node.is_leaf = false;

  // Store a safe fallback probability for invalid traversal.
  node.value = arma::mean(y);

  node.left = -1;
  node.right = -1;

  nodes.push_back(node);

  int node_index =
    static_cast<int>(nodes.size()) - 1;

  arma::mat left_X =
    X.rows(left_indices);

  arma::mat right_X =
    X.rows(right_indices);

  arma::vec left_y =
    y.elem(left_indices);

  arma::vec right_y =
    y.elem(right_indices);

  int left_child =
    build_tree_class(
      left_X,
      left_y,
      depth + 1,
      max_depth,
      effective_mtry,
      use_gini,
      nodes
    );

  int right_child =
    build_tree_class(
      right_X,
      right_y,
      depth + 1,
      max_depth,
      effective_mtry,
      use_gini,
      nodes
    );

  nodes[node_index].left = left_child;
  nodes[node_index].right = right_child;

  return node_index;
}


// -----------------------------------------------------------------------------
// Deserialize one stored R tree
// -----------------------------------------------------------------------------

std::vector<Node> deserialize_class_tree(
    const Rcpp::List& tree
) {

  const int number_of_nodes =
    tree.size();

  if (number_of_nodes <= 0) {
    return std::vector<Node>();
  }

  std::vector<Node> nodes(
      static_cast<std::size_t>(number_of_nodes)
  );

  for (int i = 0; i < number_of_nodes; ++i) {

    Rcpp::List stored_node =
      Rcpp::as<Rcpp::List>(
        tree[i]
      );

    nodes[i].feature =
      Rcpp::as<int>(
        stored_node["feature"]
      );

    nodes[i].threshold =
      Rcpp::as<double>(
        stored_node["threshold"]
      );

    nodes[i].is_leaf =
      Rcpp::as<bool>(
        stored_node["is_leaf"]
      );

    nodes[i].value =
      Rcpp::as<double>(
        stored_node["value"]
      );

    nodes[i].left =
      Rcpp::as<int>(
        stored_node["left"]
      );

    nodes[i].right =
      Rcpp::as<int>(
        stored_node["right"]
      );
  }

  return nodes;
}


// -----------------------------------------------------------------------------
// Predict one row with one tree
// -----------------------------------------------------------------------------

double predict_tree_class(
    const arma::rowvec& x,
    const std::vector<Node>& nodes,
    int node_index = 0
) {

  if (
      node_index < 0 ||
        node_index >= static_cast<int>(nodes.size())
  ) {
    return 0.0;
  }

  const Node& node =
    nodes[node_index];

  if (node.is_leaf) {
    return std::clamp(
      node.value,
      0.0,
      1.0
    );
  }

  if (
      node.feature < 0 ||
        node.feature >= static_cast<int>(x.n_elem)
  ) {
    return std::clamp(
      node.value,
      0.0,
      1.0
    );
  }

  int next_node =
    x[node.feature] <= node.threshold
  ? node.left
  : node.right;

  if (
      next_node < 0 ||
        next_node >= static_cast<int>(nodes.size())
  ) {
    return std::clamp(
      node.value,
      0.0,
      1.0
    );
  }

  return predict_tree_class(
    x,
    nodes,
    next_node
  );
}


// -----------------------------------------------------------------------------
// Fit the random forest classifier
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List rf_class_fit_fast(
    const arma::mat& X,
    const arma::vec& y,
    int n_trees = 100,
    int max_depth = 5,
    int mtry = 3,
    bool use_gini = true
) {

  if (X.n_rows == 0) {
    Rcpp::stop(
      "X must contain at least one row."
    );
  }

  if (X.n_cols == 0) {
    Rcpp::stop(
      "X must contain at least one column."
    );
  }

  if (y.n_elem != X.n_rows) {
    Rcpp::stop(
      "The length of y must equal the number of rows in X."
    );
  }

  if (n_trees <= 0) {
    Rcpp::stop(
      "n_trees must be greater than zero."
    );
  }

  if (max_depth <= 0) {
    Rcpp::stop(
      "max_depth must be greater than zero."
    );
  }

  if (
      !X.is_finite() ||
        !y.is_finite()
  ) {
    Rcpp::stop(
      "X and y must contain only finite values."
    );
  }

  if (
      arma::any(y < 0.0) ||
        arma::any(y > 1.0)
  ) {
    Rcpp::stop(
      "y must be encoded as numeric binary values 0 and 1."
    );
  }

  arma::vec rounded_y =
    arma::round(y);

  if (
      arma::any(
        arma::abs(y - rounded_y) > 1e-12
      )
  ) {
    Rcpp::stop(
      "y must be encoded as numeric binary values 0 and 1."
    );
  }

  const int number_of_rows =
    static_cast<int>(X.n_rows);

  const int number_of_features =
    static_cast<int>(X.n_cols);

  int effective_mtry = mtry;

  if (effective_mtry <= 0) {

    effective_mtry = std::max(
      1,
      static_cast<int>(
        std::floor(
          std::sqrt(
            static_cast<double>(number_of_features)
          )
        )
      )
    );
  }

  effective_mtry = std::max(
    1,
    std::min(
      effective_mtry,
      number_of_features
    )
  );

  Rcpp::List output_forest(n_trees);

  for (
      int tree_index = 0;
      tree_index < n_trees;
      ++tree_index
  ) {

    arma::uvec bootstrap_indices =
      arma::randi<arma::uvec>(
        static_cast<arma::uword>(number_of_rows),
        arma::distr_param(
          0,
          number_of_rows - 1
        )
      );

    arma::mat bootstrap_X =
      X.rows(bootstrap_indices);

    arma::vec bootstrap_y =
      y.elem(bootstrap_indices);

    std::vector<Node> nodes;

    build_tree_class(
      bootstrap_X,
      bootstrap_y,
      0,
      max_depth,
      effective_mtry,
      use_gini,
      nodes
    );

    if (nodes.empty()) {

      nodes.push_back(
        make_class_leaf(bootstrap_y)
      );
    }

    Rcpp::List tree_list(
        static_cast<R_xlen_t>(nodes.size())
    );

    for (
        std::size_t node_index = 0;
        node_index < nodes.size();
        ++node_index
    ) {

      const Node& node =
        nodes[node_index];

      tree_list[
      static_cast<R_xlen_t>(node_index)
      ] = Rcpp::List::create(
        Rcpp::Named("feature") =
          node.feature,
          Rcpp::Named("threshold") =
            node.threshold,
            Rcpp::Named("is_leaf") =
              node.is_leaf,
              Rcpp::Named("value") =
                node.value,
                Rcpp::Named("left") =
                  node.left,
                  Rcpp::Named("right") =
                    node.right
      );
    }

    output_forest[tree_index] =
      tree_list;
  }

  return output_forest;
}


// -----------------------------------------------------------------------------
// Predict class-1 probabilities
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
arma::vec rf_class_predict_prob_fast(
    const arma::mat& X,
    Rcpp::List forest_list
) {

  if (X.n_cols == 0) {
    Rcpp::stop(
      "X must contain at least one column."
    );
  }

  if (!X.is_finite()) {
    Rcpp::stop(
      "X must contain only finite values."
    );
  }

  const int number_of_trees =
    forest_list.size();

  if (number_of_trees <= 0) {
    Rcpp::stop(
      "forest_list must contain at least one tree."
    );
  }

  arma::vec probabilities(
      X.n_rows,
      arma::fill::zeros
  );

  int valid_trees = 0;

  for (
      int tree_index = 0;
      tree_index < number_of_trees;
      ++tree_index
  ) {

    Rcpp::List tree =
      Rcpp::as<Rcpp::List>(
        forest_list[tree_index]
      );

    if (tree.size() == 0) {
      continue;
    }

    std::vector<Node> nodes =
      deserialize_class_tree(tree);

    if (nodes.empty()) {
      continue;
    }

    ++valid_trees;

    for (
        arma::uword row_index = 0;
        row_index < X.n_rows;
        ++row_index
    ) {

      probabilities[row_index] +=
        predict_tree_class(
          X.row(row_index),
          nodes
        );
    }
  }

  if (valid_trees == 0) {
    Rcpp::stop(
      "forest_list does not contain any valid trees."
    );
  }

  probabilities /=
    static_cast<double>(valid_trees);

  probabilities =
    arma::clamp(
      probabilities,
      0.0,
      1.0
    );

  return probabilities;
}


// -----------------------------------------------------------------------------
// Predict hard classes
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
arma::vec rf_class_predict_fast(
    const arma::mat& X,
    Rcpp::List forest_list
) {

  arma::vec probabilities =
    rf_class_predict_prob_fast(
      X,
      forest_list
    );

  return arma::conv_to<arma::vec>::from(
    probabilities >= 0.5
  );
}

