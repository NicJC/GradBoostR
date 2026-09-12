#ifndef TREE_UTILS_H
#define TREE_UTILS_H

#include <Rcpp.h>
using namespace Rcpp;

struct Node {
  int feature;
  double threshold;
  bool is_leaf;
  double value;
  int left;
  int right;
};

inline double traverse_tree(const NumericMatrix& X, int row,
                            const std::vector<Node>& nodes) {

  int idx = 0;
  while (!nodes[idx].is_leaf) {
    int f = nodes[idx].feature;
    double thr = nodes[idx].threshold;

    if (X(row, f) <= thr)
      idx = nodes[idx].left;
    else
      idx = nodes[idx].right;
  }
  return nodes[idx].value;
}

#endif
