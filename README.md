# VIB Project 2026
Based in a 3D finite element frame program program developed in the BSc project *"Developing a Digital Twin for the Dynamic Analysis of Jacket Foundations for Offshore Wind Turbines."*

## Requirements

- numpy  
- sqlite3  
- os  
- scipy  
- pyplot  
- pickle  

## Features

- **data**: Functions for the collection of data for running the analysis and collecting results (`baseclear`, `baseinsert`, `basestore`, `output`)  
- **Kmat**: Functions for building the system stiffness matrix (`Abeam`, `Bint`, `buildK`, `intpL`, `kbeam`, `kspring`)  
- **Mmat**: Functions for building the system mass matrix, as well as running natural frequency analysis (`buildM`, `mbeam`, `Nint`, `NFA`)  
- **plot**: Functions for plotting both input and output data, including mode shapes (`plotconnectivity`, `plotmodeshapes`, `utils`)  
- **classes**: Function for running the program for NFA and data collection. Manages collaboration between functions (`VIBframe`, `VIBdata`)  

## Usage

The program's main structure is to:  
1. Initialize input data  
2. Build the system stiffness and mass matrices based on the finite element formulation, using Gauss-Legendre quadrature points. These matrices are used to solve the eigenvalue problem and determine natural frequencies and mode shapes.  
3. Print and optionally plot the output using the plotting functions  

Each of these three steps is demonstrated in the example calculations EX1.1–EX1.3, which can be modified for specific purposes. The following describes what can be changed:

### EX1.1: Build input data file
The input data file should be build firstly.


### EX1.2: Run analysis

The program allows for adding boundary conditions and/or spring supports.

- **Numpy arrays**: Both boundary conditions and spring supports have shape `(3, number of supports)`.  
  - Boundary conditions: `[node number, local degree of freedom, displacement]`  
  - Spring supports: `[node number, local degree of freedom, local stiffness]`  
- **List**: The `solve` subset is a list of the eigenvalue indices to solve for  

The results of the analysis are saved to a text file, e.g., `"VIB_results.txt"`.

### EX1.3: Plotting mode shapes

The program can plot the results of the analysis. The `plotmodeshapes` function plots the selected mode shape (from the solved subset) by specifying its mode number.

---
**Authors:** Iven Henrik Meyer (s2238326) & Mia Steen Duus (s223832)
