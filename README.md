<b><u>GradBoostR</b></u>
<hr></hr>

<p align="left">
  <img src="GradBoostR.png" width="200">
</p>

<i>Fast ML engine with boosting, random forests, and neural nets.</i>

# Why <b>GradBoostR</b>, and how does it work ?

<hr></hr>

<b>GradBoostR</b> is a unified machine‑learning toolkit designed to make fast, consistent modeling easy in R.
It provides four high‑performance algorithms implemented in optimized C++:

* Random Forest Regression

* Random Forest Classification

* Gradient Boosting Machines (GBM)

* Neural Network Regression


All models share the same simple interface.


<b>GradBoostR</b> has two main R functions:

* gradboostr_fit() → trains a model

* gradboostr_predict() → makes predictions with that model

Everything else in the package is built around these two ideas.

<B>It is ideal for:</b>

 - tabular regression

 - tabular classification

 - teaching ML algorithms

 - experimenting with tree and boosting logic

 - building your own modeling ecosystem
 
 
## Let’s break things down.
 
### Training a Model
 
Think of <b><i>gradboostr_fit()</i></b> as the “model builder.”
 
 <u>You add:</u>  
 
 * a feature matrix X

 * a target y

 * a method (rf, rf_class, gbm, nn)

 * optional hyperparameters
 
### Your inputs are checked

<b>GradBoostR:</b>

 Converts data frames to matrices

 Ensures X is a matrix

 Ensures classification targets are factors
 
### Choice of engine
<i>Depending on the "method", a call is made to the relevant functions :</i>

| Method     | What happens                                      |
|------------|---------------------------------------------------|
| `rf`       |       Calls C++ random forest regression engine     |
| `rf_class` |       Calls C++ random forest classification engine |
| `gbm`      |       Calls C++ gradient boosting engine                |
| `nn`       |       Calls C++ neural network engine                   |

## How to Tune Neural Networks in GradBoostR

Because the architecture is compact, tuning is straightforward — <u>but the choices matter!</u>



#### Hidden Layer Size

<i>This is the most important hyperparameter.</i>

It controls the number of neurons(computational nodes) in the hidden layer.

```md 
```r
m_nn <- gradboostr_fit(X, y_reg, method = "nn", hidden = 10, epochs = 50)

```

The affect on performance: 

* Too small → underfitting

* Too large → overfitting, slower training

* Just right → smooth nonlinear approximation

<b>A guideline:</b>

| Data complexity                   |    Recommended hidden size      |
|-----------------------------------|---------------------------------|
| Linear-ish                        |       3–10                      |
| Mild nonlinear                    |       10–30                     |
| Strong nonlinear                  |       30–80                     |
| Very complex                      |       80–150                    |

#### Number of Epochs

<i>Controls how long the network trains.</i>

Each epoch runs a full forward + backward pass.

```md
```r
m_nn <- gradboostr_fit(X, y, method = "nn", epochs = 100)
```

The affect on performance:

* Too few → model doesn’t converge

* Too many → overfitting, wasted compute

<b>A guideline:</b>

| Dataset size            |    Recommended epochs     |
|-------------------------|---------------------------|
| < 1,000 rows            |       30–80               |
| 1,000–10,000            |       80–150              |
| > 10,000                |       150–300             |


#### Learning Rate

<i>If your NN engine exposes this, it controls how big each gradient step is.</i>

An optimal learning rate lets the model reach a low error rate , improves the training process, improves accuracy ...

<b>A guideline:</b>

| Learning rate            |    Behavior                    |
|--------------------------|--------------------------------|
| 0.001                    |      slow but stable           |
| 0.01                     |       good default             |
| 0.05                     |   fast but may overshoot       |
| 0.1                      | unstable unless data is simple |
