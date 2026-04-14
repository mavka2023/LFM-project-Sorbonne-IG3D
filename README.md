# Leapfrog Flow Maps — Python Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A Python-based implementation of the **Leapfrog Flow Maps (LFM)** incompressible fluid simulation algorithm. 
This project was developed as part of the **IG3D course at Sorbonne University**.

The algorithm is based on a SIGGRAPH 2025 research paper and has been ported from the original low-level CUDA C++ codebase into a Python environment (using NumPy and SciPy).
## 📺 Demo Animation

<div align="center">
  <video src="animation_IG3D_project_LFM_Svintsitska.mp4" width="100%" autoplay loop muted></video>
  <p><i>Real-time vortical structures and fluid details captured using our LFM implementation.</i></p>
</div>
## 📁 Repository Structure

* `lfm/` – The core mathematical engine
* `simulations/` – Ready-to-run scripts featuring classic physical scenarios
* `visualization/` – Tools for data export (VTK) and rapid 2D/3D prototyping using Matplotlib.
* `tests/` – A comprehensive suite of unit tests
  
## Installation

1. Clone the repository:
```
git clone https://github.com/mavka2023/LFM-project-Sorbonne-IG3D.git
cd LFM-project-Sorbonne-IG3D
```
2. Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```
## Running Simulations
The project includes several pre-configured physical scenarios located in the simulations/ directory. Each script is handles its own initialization.

### Running a 3D Simulation (Example: Smoke Plume)
To run a 3D simulation that generates VTK files for ParaView:
```
python simulations/sim_3d_smoke_plume.py
```
Outputs: Results will be saved in output_3d_smoke_plume/.

## Visualizing Results
ParaView: Open ParaView, click File -> Open, and select the .vtk group (e.g., smoke_..vtk) from the output folder. 
Apply the Volume representation for 3D effects.

For 2D there is already a .gif.

