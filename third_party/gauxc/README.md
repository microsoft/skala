# About

GauXC 

Copyright (c) 2020-2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). 

(c) 2024-2025, Microsoft Corporation

All rights reserved.


NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

# Synopsis

GauXC is a modern, modular C++ library for the evaluation of quantities related
to the exchange-correlation (XC) and exact-exchange (K) energy (e.g. potential, etc) in the Gaussian
basis set discretization of Kohn-Sham density function theory (KS-DFT). GauXC
provides efficient, scalable distributed memory XC and K integrators for both CPU and
accelerator-based (GPU) architectures. Currently, GPU support is provided through
the
[CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) and
[HIP](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
frameworks to target NVIDIA and AMD GPUs, respectively.
Evaluation
of the XC functional CPU/accelerator architectures is provided by the
[ExchCXX](https://github.com/wavefunction91/ExchCXX) library. Quadratures are generated
by the [IntegratorXX](https://github.com/wavefunction91/IntegratorXX) library.

# Design Goals

* Provide a stable, portable and high-performance implementation of numerical
integrators optimized for the evaluation of XC and K related quantities in Gaussian
basis set KS-DFT on CPU and accelerator based architectures.
* Develop a modern, modular, extensible C++ software infrastructure which allows
for flexible and agile development in the field of KS-DFT.

# Dependencies

* CMake (3.20+)
* BLAS (for CPU integrators)
* [ExchCXX](https://github.com/wavefunction91/ExchCXX)
* [IntegratorXX](https://github.com/wavefunction91/IntegratorXX)
* [Gau2Grid](https://github.com/dgasmith/gau2grid) (pregenerated source packaged with GauXC)
* MPI (Optional)
* OpenMP (CPU parallelism, Optional)
* [Cereal](https://github.com/USCiLab/cereal) (Optional)
* [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (Optional)
* [Eigen3](https://eigen.tuxfamily.org/dox/) (Testing Only)
* [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)/[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) (Required only if CUDA enabled)
* [HIP](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)/[ROCm](https://github.com/RadeonOpenCompute/ROCm) (Required only if HIP enabled)
* [MAGMA](https://icl.utk.edu/magma/) (Optional if CUDA/HIP enabled)

# Major Contributors

See CONTRIBUTORS.md for a list of major contributors to GauXC.

# Publications

## GauXC
Please cite the following publications if GauXC was used in your publication:
```
% Relativistic integrals
@article{kovtun2024relativistic,
  author = {Kovtun, Mikael and Lambros, Eleftherios and Liu, Aodong and Tang, Diandong and Williams--Young, David B. and Li, Xiaosong},
  title = {Accelerating Relativistic Exact-Two-Component Density Functional Theory Calculations with Graphical Processing Units},
  journal = {Journal of Chemical Theory and Computation},
  volume = {20},
  number = {18},
  pages = {7694--7699},
  year = {2024},
  doi = {10.1021/acs.jctc.4c00843},
}

% Distributed Memory Seminumerical Exact Exchange implementation
@article{williams2023distributed,
  title = {Distributed memory, GPU accelerated Fock construction for hybrid, Gaussian basis density functional theory},
  author = {Williams--Young, David B. and Asadchev, Andrey and Popovici, Doru Thom and Clark, David and Waldrop, Jonathan and
            Windus, Theresa L. and Valeev, Edward F. and de Jong, Wibe A.},
  journal = {The Journal of Chemical Physics},
  volume = {158},
  number = {23},
  pages = {234104},
  year = {2023},
  doi = {10.1063/5.0151070},
  url = {https://doi.org/10.1063/5.0151070}
}

% Performance Portability (HIP/SYCL implementations)
@article{williams2021achieving,
  title={Achieving performance portability in Gaussian basis set density functional
         theory on accelerator based architectures in NWChemEx},
  author={Williams--Young, David B and Bagusetty, Abhishek and de Jong, Wibe A and
          Doerfler, Douglas and van Dam, Hubertus JJ and V{\'a}zquez-Mayagoitia, {\'A}lvaro and
          Windus, Theresa L and Yang, Chao},
  journal={Parallel Computing},
  volume={108},
  pages={102829},
  year={2021},
  doi={10.1016/j.parco.2021.102829},
  url={https://www.sciencedirect.com/science/article/pii/S0167819121000776?via%3Dihub}
}

% CUDA and distributed memory implementation
@article{williams20on,
  author={David B. Williams--Young and Wibe A. de Jong and Hubertus J.J. van Dam and
          Chao Yang},
  title={On the Efficient Evaluation of the Exchange Correlation Potential on
         Graphics Processing Unit Clusters},
  journal={Frontiers in Chemistry},
  volume={8},
  pages={581058},
  year={2020},
  doi={10.3389/fchem.2020.581058},
  url={https://www.frontiersin.org/articles/10.3389/fchem.2020.581058/abstract},
  preprint={https://arxiv.org/abs/2007.03143}
}

% Algorithm for XC potential assembly and shared-memory CPU implementation
@article{petrone18an,
  author={Alessio Petrone and David B. Williams--Young and Shichao Sun and
          Torin F. Stetina and Xiaosong Li},
  title={An Efficient Implementation of Two-Component Relativistic Density
         Functional Theory with Torque-Free Auxiliary Variables},
  journal={The European Physical Journal B},
  volume={91},
  number={169},
  pages={169},
  year={2018},
  doi={10.1140/epjb/e2018-90170-1},
  url={https://link.springer.com/article/10.1140/epjb/e2018-90170-1}
}
```

## Density functionals

If GauXC was used for the evaluation of exchange-correlation related
quantities in your publication, we request that you also cite
[Libxc](https://libxc.gitlab.io/) which provides the underlying
implementation of the exchange-correlation functionals used in GauXC
via the [ExchCXX](https://github.com/wavefunction91/ExchCXX) library:

```
% Actual Implementations of the Density Functionals
@article{lehtola2018libxc,
  author  = {Lehtola, Susi and Steigemann, Conrad and Oliveira, Micael J. T. and Marques, Miguel A. L.},
  journal = {SoftwareX},
  title   = {Recent developments in {LIBXC}---a comprehensive library of functionals for density functional theory},
  year    = {2018},
  pages   = {1--5},
  volume  = {7},
  doi     = {10.1016/j.softx.2017.11.002},
}
```

# Build Instructions

GauXC provides a CMake build system with automatic dependency management (through [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)).
As such, a simple CMake invocation will often suffice for most purposes
```
cmake -S /path/to/gauxc -B /path/to/build [GauXC configure options]
cmake --build /path/to/build
```


GauXC is linkable both as an installed library as well as a CMake subproject via `FetchContent`
```
# GauXC Discovery
find_package( gauxc REQUIRED )
target_link_libraries( my_target PUBLIC gauxc::gauxc )
```

```
# GauXC as CMake Subproject
include(FetchContent)

# Set GauXC CMake options (see below)

# Pull master branch of GauXC
FetchContent_Declare( gauxc
  GIT_REPOSITORY https://github/com/wavefunction91/GauXC.git
  GIT_TAG master
)
FetchContent_MakeAvailable( gauxc )

# Link to target
target_link_libraries( my_target PUBLIC gauxc::gauxc )
```


## Influential CMake Variables

| Variable Name              | Description                                               | Default  |
|----------------------------|-----------------------------------------------------------|----------|
| `GAUXC_ENABLE_TESTS`       | Enable Testing Framework (Catch2)                         | `ON`     |
| `GAUXC_ENABLE_HOST`        | Enable HOST integrators                                   | `ON`     |
| `GAUXC_ENABLE_CUDA`        | Enable CUDA integrators                                   | `OFF`    |
| `GAUXC_ENABLE_HIP`         | Enable HIP integrators                                    | `OFF`    |
| `GAUXC_ENABLE_MAGMA`       | Enable MAGMA for batched BLAS (No effect if no GPU)       | `ON`     |
| `GAUXC_ENABLE_CUTLASS`     | Enable CUTLASS for batched BLAS (No effect if no CUDA)    | `OFF`    |
| `GAUXC_ENABLE_NCCL`        | Enable NCCL bindings for topology aware GPU reductions    | `OFF`    |
| `GAUXC_ENABLE_MPI`         | Enable MPI Bindings                                       | `ON`     |
| `GAUXC_ENABLE_OPENMP`      | Enable OpenMP Bindings                                    | `ON`     |
| `CMAKE_CUDA_ARCHITECTURES` | CUDA architechtures (e.g. 70 for Volta, 80 for Ampere)    |  --      |
| `BLAS_LIBRARIES`           | Full BLAS linker.                                         |  --      |
| `MAGMA_ROOT_DIR`           | Install prefix for MAGMA.                                 |  --      |




# Example Usage

See `test/standalone_driver.cxx` for an example end-to-end invocation of GauXC for various integrands.


# License

GauXC is made freely available under the terms of a modified 3-Clause BSD license. See
LICENSE.txt for details.

# Acknowledgments

The development of GauXC was previously supported by the Exascale Computing Project
(17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office
of Science and the National Nuclear Security Administration.

##Trademarks 

This project may contain trademarks or logos for projects, products, or
services. Authorized use of Microsoft trademarks or logos is subject to and
must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft
trademarks or logos in modified versions of this project must not cause
confusion or imply Microsoft sponsorship. Any use of third-party trademarks or
logos are subject to those third-party’s policies.
