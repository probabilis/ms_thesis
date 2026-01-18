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
- `read.py`    


## Workflow:

- `preprocessing.py` ... preprocess raw data
- `read.py` ... read preprocessed data
- `pattern_formation.py` ... run pattern formation algorithm
- `evaluation_add.py` ... evaluate results             ... CSV-Reader Method


## Information:
Magnetic Image mapping:

- Black ... +1 
- White ... -1


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
- Check all scripts for loop efficiency & parallelization from Torch module -> X
- Revisited read.py for data-reading (exp. recordings) -> X
- preprocessing.py script for right MCD calculation -> X
- Integration of new MCD PreProcessing class        -> X
    - Read PIL Image to Tensor
    - Compute MCD 
    - Save as CSV for Post-Reading Simulations
- Preprocessing -> Reading -> Evaluation (params.json) -> Postprocessing -> X
- Lipschitz constant calculation    ->  X   
- Finite Difference / Fourier method calculation -> X
- Initial File Reading methods (Raw, Standardized, Stand.+ Shift, Clipped)  -> X
- Experimental Data Results             -> X
- Fourier Modes                         -> X
- Master thesis Chapter Allignment      -> X
- Check Energy convergence with exp. data -> X
- Comparison for non-dx / dx simulation -> X
- Make power law estimator plot          -> X
- Implement asymptotic evolution (Condette page 75) -> X
- Check GD algorithm once for possible errors (such slow convergence) -> X




### Open:
#### High Priority:

- Check Lipshitz constant -> implement in GD for rerun + autograd check once again
- Integrate data-savings for all algorithms 
- simulation evolution / concurrent savings for let'say max_it/10 times
- single domain evolution

#### Low Priority:
- Check runtimes by multiple runs to get an appropriate estimator -> runtime.py script
- Standardize plotting schematic + LaTex Font integration (still open: axis + latex font)









