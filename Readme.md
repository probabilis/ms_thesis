# Magnetic Imaging - Master Thesis
## Maximilian Gschaider 
### *supervised by Prof. Martin Schultze and Prof. Thomas Pock*.

### Adapted Crank Nicolson schematic re-implemented from Nicolas Condette: *Pattern Formation in Magnetic Thin Films: Analysis and Numerics*


## Code Structure:
Parent scripts:
- `env_utils.py`            ... helper functions and overall stuff which is multiple used
- `params.py`               ... global parameter dataclasses       
- `pattern_formation.py`    ... pattern formation methods

Numerical methods:
- `gradient_descent.py`     ... Gradient Descent method
- `gd_proximal.py`          ... Proximal Gradient Descent method
- `gd_nesterov.py`          ... Accelerated Gradient Descent method
- `crank_nicolson.py`       ... Adapted Crank Nicolson method with reference to N.C.

## ToDo's:

### Closed:
- Implement Gradient Descent for Pattern Formation -> X
- Check (2pi)² factor at linear operator at CN -> X
- Re-Implement Fourier Multiplier with safe operation for divergence + vectorized -> X
- Integrate GD Autograd for learning rate backtracking -> X
- Integrate Prox.GD + Nesterov -> X
- Check Sin() stripe-pattern evolution -> check / no parallel-diagonal stripes obtained -> X
- Implement CN STOP_BY_LIMIT -> X
- Check runtimes + convergence for correct implementation -> X
- Implement comparison.py script -> X

### Open:
- Restructure project with modules and classes
- Remove redunant code
- Integrate function docstrings for main methods
- Standardize plotting schematic + LaTex Font integration
- Integrate data-savings for all algorithms (+ concurrent savings for let'say max_it/10 times)
- Check runtimes by multiple runs to get an appropriate estimator
- Check loop efficiency & parallelization for Torch module 
- comparison.py: plot all u's after e.g.: 1000 iterations