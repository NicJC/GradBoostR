#include <RcppArmadillo.h>
using namespace Rcpp;


// [[Rcpp::depends(RcppArmadillo)]]

struct MLP {
  arma::mat W1;
  arma::vec b1;
  arma::mat W2;
  arma::vec b2;
};

// initialize weights
MLP mlp_init(int p, int h) {
  MLP net;
  net.W1 = 0.01 * arma::randn<arma::mat>(p, h);
  net.b1 = arma::zeros<arma::vec>(h);
  net.W2 = 0.01 * arma::randn<arma::mat>(h, 1);
  net.b2 = arma::zeros<arma::vec>(1);
  return net;
}

// [[Rcpp::export]]
List nn_train_arma(const arma::mat& X,
                   const arma::vec& y,
                   int hidden_units = 32,
                   int epochs = 50,
                   double learning_rate = 0.01) {

  int n = X.n_rows;
  int p = X.n_cols;

  MLP net = mlp_init(p, hidden_units);

  for (int e = 0; e < epochs; ++e) {
    // forward
    arma::mat Z1 = X * net.W1;
    Z1.each_row() += net.b1.t();
    arma::mat A1 = arma::clamp(Z1, 0.0, arma::datum::inf); // ReLU

    arma::vec Z2 = A1 * net.W2;
    Z2 += net.b2(0);
    arma::vec y_hat = Z2; // linear output (regression)

    // loss gradient (MSE)
    arma::vec diff = (y_hat - y);          // n x 1
    arma::vec dZ2 = 2.0 * diff / n;        // n x 1

    // backprop
    arma::mat dW2 = A1.t() * dZ2;          // h x 1
    double db2 = arma::sum(dZ2);
    // 1

    arma::mat dA1 = dZ2 * net.W2.t();     // n x h
    arma::mat dZ1 = dA1;
    dZ1.elem(arma::find(Z1 <= 0.0)).zeros(); // ReLU grad

    arma::mat dW1 = X.t() * dZ1;          // p x h
    arma::vec db1 = arma::sum(dZ1, 0).t();// h

    // SGD update
    net.W1 -= learning_rate * dW1;
    net.b1 -= learning_rate * db1;
    net.W2 -= learning_rate * dW2;
    net.b2 -= learning_rate * db2;
  }

  return List::create(
    _["W1"] = net.W1,
    _["b1"] = net.b1,
    _["W2"] = net.W2,
    _["b2"] = net.b2
  );
}

// [[Rcpp::export]]
arma::vec nn_predict_model(const arma::mat& X, List model) {
  arma::mat W1 = as<arma::mat>(model["W1"]);
  arma::vec b1 = as<arma::vec>(model["b1"]);
  arma::mat W2 = as<arma::mat>(model["W2"]);
  arma::vec b2 = as<arma::vec>(model["b2"]);

  arma::mat Z1 = X * W1;
  Z1.each_row() += b1.t();
  arma::mat A1 = arma::clamp(Z1, 0.0, arma::datum::inf); // ReLU

  arma::vec Z2 = A1 * W2;
  Z2 += b2(0);

  return Z2; // regression output
}
