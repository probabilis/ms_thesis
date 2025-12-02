# Magnetic Imaging 
### Master Thesis *supervised by Prof. Martin Schultze and Prof. Thomas Pock*.

### Numerical optimization algorithms for simualating magnetic pattern formation
Various Gradient Descent methods + Adapted Crank Nicolson schematic re-implemented from Nicolas Condette: *Pattern Formation in Magnetic Thin Films: Analysis and Numerics*


## Code Structure:
Parent scripts:
- `env_utils.py`            ... helper functions and overall stuff which is multiple used
- `params.py`               ... global parameter dataclasses       
- `pattern_formation.py`    ... pattern formation methods

Numerical methods:
- `gradient_descent.py`     ... Gradient Descent method ()
- `gd_proximal.py`          ... Proximal Gradient Descent method
- `gd_nesterov.py`          ... Accelerated Gradient Descent method
- `crank_nicolson.py`       ... Adapted Crank Nicolson method with reference to N.C.

Evaluation scripts:
- `evaluation.py`           ... Evaluation script 
- `evaluation_add.py`       ... Evaluation script for multiple instances
- `parameter_study.py`      ... Simulation parameters Gamma & Epsilon Parameter study

Auxilliary scripts:
- `preprocessing.py`        ... Raw *.TIF data reader and MCD pre-processing -> to *.CSV
- `read.py`                 ... CSV-Reader Method

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
- Integrate function dataclasses asdict for main methods -> X
- comparison.py: plot all u's after e.g.: 1000 iterations -> X
- Check all scripts for loop efficiency & parallelization for Torch module -> X
- Revisited read.py for data-reading (exp. recordings) -> X
- preprocessing.py script for right MCD calculation -> X

### Open:
#### High Priority:s
- Integrate data-savings for all algorithms (+ concurrent savings for let'say max_it/10 times)

#### Low Priority:
- Check runtimes by multiple runs to get an appropriate estimator -> runtime.py script
- Check GD algorithm once for possible errors (such slow convergence)
- Standardize plotting schematic + LaTex Font integration (still open: axis + latex font)


## Workflow:

- `preprocessing.py` ... preprocess raw data
- `read.py` ... read preprocessed data
- `pattern_formation.py` ... run pattern formation algorithm
- `evaluation_add.py` ... evaluate results


### Information:

- Black ... +1 
- White ... -1