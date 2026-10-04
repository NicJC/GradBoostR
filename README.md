# GradBoostR <img src="https://raw.githubusercontent.com/nicjc/GradBoostR/main/man/figures/logo.png" align="right" width="120" />

<!-- Badges -->
[![CRAN Status](https://www.r-pkg.org/badges/version/GradBoostR)](https://CRAN.R-project.org/package=GradBoostR)
[![CRAN Downloads](https://cranlogs.r-pkg.org/badges/GradBoostR)](https://cranlogs.r-pkg.org/badges/GradBoostR)
[![R-CMD-check](https://github.com/nicjc/GradBoostR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/nicjc/GradBoostR/actions/workflows/R-CMD-check.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

**GradBoostR** is a fast, lightweight machine learning package for R that implements:

- **Random Forests** (regression + classification)  
- **Gradient Boosting Machines (GBM)**  
- **Neural Networks (NN)** with Armadillo backend  

The package is designed for **speed**, **simplicity**, and **clean R interfaces**, while leveraging optimized C++ code via RcppArmadillo.

---

## Installation

### From CRAN (once published)

```r
install.packages("GradBoostR")
