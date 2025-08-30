# Reservoir Computing – Hyperparameter Search and Evaluation

This repository contains experiments and utilities for **Reservoir Computing** with a focus on the **Rössler system**. The main purpose is to explore hyperparameter configurations through systematic search methods and to evaluate their performance.

## Repository Structure

- **`*_eval/` folders**  
  These directories are used for reading tables with multiple hyperparameter combinations. The results of the evaluations are written back into a separate output table, where the computed metrics and comparisons are stored.

- **Other folders (non-eval)**  
  These contain scripts for testing the code with **single hyperparameter combinations**. They were primarily used for debugging and verifying the workflow.

- **Excel files (`.xlsx`)**  
  The spreadsheets contain the **best hyperparameter combinations found so far** for the Rössler system. The results were generated using the GridSearch and RandomSearch scripts and are kept for reference and further analysis.

- **Agent–Edge structure with Docker and ROS2**  
  All folders implement the **Agent–Edge architecture** using **Docker** and **ROS2**. The Agent is responsible for data generation and preprocessing, while the Edge Node executes the reservoir computations and predictions. This setup enables distributed experiments and evaluation of latency and throughput in a realistic edge computing environment.

## Search Methods

- **GridSearch**  
  Designed for exploring a **manually defined, smaller parameter space**. Every possible combination within this restricted search space is evaluated exhaustively.

- **RandomSearch**  
  Suitable for a **very large parameter space**, where testing every combination would be computationally infeasible. Instead, random sampling is applied to cover a broad range of configurations efficiently.  
  Both search methods are implemented in a **parallelized manner** to speed up execution and take advantage of modern hardware.

## Installation & Requirements

The project is written in **Python 3.10+**. You can install the required dependencies via `pip`:

```bash
pip install -r requirements.txt
```

### Key dependencies
- [ReservoirPy 0.3.13](https://github.com/reservoirpy/reservoirpy) (specific version required for compatibility)  
- NumPy (latest version tested)  
- SciPy (latest version tested)  
- scikit-learn (latest version tested)  
- joblib (latest version tested)  
- pandas (for table handling, latest version tested)  
- matplotlib (for visualization, optional)

> Note: Except for ReservoirPy, the latest stable versions of the other packages have been tested without compatibility issues.
