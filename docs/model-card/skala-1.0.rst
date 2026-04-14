Skala model
===========

Model details
-------------

In pursuit of the universal functional for density functional theory (DFT), the OneDFT team from Microsoft Research AI for Science has developed the Skala-1.0 exchange-correlation functional, as introduced in `Accurate and scalable exchange-correlation with deep learning (arXiv v5), Luise et al. 2025 <https://arxiv.org/abs/2506.14665v5>`__.
This approach departs from the traditional route of incorporating increasingly expensive hand-designed non-local features from Jacob's ladder into functional forms to improve their accuracy.
Instead, we employ a deep learning approach with a scalable neural network that uses only inexpensive input features to learn the necessary non-local representations.

The functional is based on a neural network architecture that takes as input features on a 3D grid describing the electron density and derived meta-generalized-gradient (meta-GGA) quantities.
The architecture performs scalable non-local message-passing on the integration grid via a second, coarser grid, combined with shared local layers that enable representation learning of both local and non-local features.
These representations are then used to predict the exchange-correlation energy in an end-to-end data-driven manner.

To facilitate this learning, the model is trained on a dataset of unprecedented size, containing highly accurate energy labels from coupled cluster theory.
The largest subset focuses on atomization energies and was generated in collaboration with the University of New England.
This subset is released as part of the Microsoft Research Accurate Chemistry Collection (MSR-ACC, `Accurate Chemistry Collection: Coupled cluster atomization energies for broad chemical space, Ehlert et al. 2025 <https://arxiv.org/abs/2506.14492v5>`__).
To broaden coverage of other types of chemistry, the training dataset is further complemented with in-house generated datasets covering conformers, ionization potentials, proton affinities, and elementary reactions, as well as a small amount of publicly available high-accuracy data.

We demonstrate that combining a large-scale high-accuracy dataset with our deep learning architecture yields a functional that predicts atomization energies at chemical accuracy (1 kcal/mol), as measured on the W4-17 benchmark set.
On GMTKN55, which covers general main-group thermochemistry, kinetics, and noncovalent interactions, the Skala-1.0 functional achieves a WTMAD-2 of 3.89 kcal/mol.
This accuracy is competitive with state-of-the-art range-separated hybrid functionals, while only requiring a cost comparable to semi-local DFT.
With this work, we demonstrate the viability of our approach toward the universal density functional across all of chemistry.

Users of this model are expected to have a basic understanding of the field of quantum chemistry and density functional theory. 

:Developed by:
  Chin-Wei Huang, Deniz Gunceler, Derk Kooi, Klaas Giesbertz, Giulia Luise, Jan Hermann, Megan Stanley, Paola Gori Giorgi, Rianne van den Berg, Sebastian Ehlert, Stephanie Lanius, Thijs Vogels, Wessel Bruinsma

:Shared by:
  Microsoft Research AI for Science

:Model type:
  Neural Network Density Functional Theory Exchange Correlation Functional

:License:
  MIT


Direct intended uses
--------------------

#. The Skala-1.0 functional is shared with the research community to facilitate reproduction of the evaluations presented in our paper.
#. Evaluating reaction energy differences by computing the total energy of all compounds in a reaction using a self-consistent field (SCF) calculation with the Skala-1.0 exchange-correlation functional.
#. Evaluating the total energy of a molecule using an SCF calculation with the Skala-1.0 exchange-correlation functional. Note that, as with all density functionals, energy differences are predicted much more reliably than total energies of individual molecules.
#. The SCF implementation provided uses PySCF, which runs the functional on CPU. We also provide a traced version of the Skala-1.0 functional so that other, more optimized open-source SCF codes—including GPU-enabled ones—can integrate it into their pipelines, for instance through GauXC. A compatible fork of GauXC is included in this repository.

Out-of-scope uses
-----------------

#. Evaluating the functional with a single pass given a fixed density as input is not the intended way to evaluate the model. The model's predictions should always be made by using it as part of an SCF procedure. 
#. We do not include a training pipeline for the Skala-1.0 functional in this code base.

Risks and limitations
---------------------

#. Interpretation of results requires expertise in quantum chemistry.
#. The Skala-1.0 functional is trained on atomization energies, conformers, proton affinities, ionization potentials, elementary reaction pathways, and non-covalent interactions, as well as a small amount of electron affinities and total energies of atoms. We have benchmarked performance on W4-17 for atomization energies and on GMTKN55, which covers general main-group thermochemistry, kinetics, and noncovalent interactions, to provide an indication of generalization beyond the training set. We have also evaluated robustness on dipole moment predictions and geometry optimization.
#. The Skala-1.0 functional has been trained on data containing the following elements: H–Ar, Br, Kr, I, Xe. It has been tested on data containing H–Ca, Ge–Kr, Sn–I, Pb, and Bi.
#. Given points 2 and 3 above, this is not a production model. We advise testing the functional further before applying it to your research and welcome any feedback.

Recommendations
---------------

#. In our PySCF-based SCF implementation, the largest system tested contained 180 atoms using the def2-TZVP basis set (:math:`\sim`\ 5000 orbitals) on `Eadsv5 series <https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/eadsv5-series?tabs=sizebasic>`__ virtual machines. Larger systems may run out of memory.
#. For implementations optimized for memory, speed, or GPU support, we recommend integrating the functional with other open-source SCF packages, for instance through GauXC. A compatible fork of GauXC is included in this repository.
#. Skala-1.0 will also be available through `Azure AI Foundry <https://labs.ai.azure.com/projects/skala/>`__, where it is coupled with Microsoft's GPU-accelerated `Accelerated DFT <https://arxiv.org/abs/2406.11185>`__ application.


Training details
----------------

Training data
~~~~~~~~~~~~~

The following data is included in our training set:

- 99% of MSR-ACC:TAE (:math:`\sim`\ 78k reactions) containing atomization energies.
  This data was generated in collaboration with Prof. Amir Karton, University of New England, using the W1-F12 composite protocol based on CCSD(T), and is released as part of the `Microsoft Research Accurate Chemistry Collection <https://arxiv.org/abs/2506.14492v5>`__ (MSR-ACC).
- Total energies, electron affinities, and ionization potentials (up to triple ionization) for atoms from H to Ar (excluding Li and Be due to basis set constraints).
  This data was produced in-house with CCSD(T) by extrapolating to the complete basis set limit from quadruple zeta (QZ) and pentuple zeta (5Z) calculations.
  The basis sets used for H and He were aug-cc-pV(Q+d)Z, aug-cc-pV(5+d), while for the remaining elements B–Ar the basis sets were aug-cc-pCVQZ and aug-cc-pCV5Z.
  All basis sets were obtained from the `Basis Set Exchange (BSE) <https://www.basissetexchange.org/>`__.
  Extrapolation of the correlation energy was performed by fitting a :math:`Z^{-3}` expression, while the Hartree–Fock energy was extrapolated using the two-point scheme of :footcite:`karton2006`.
- Four datasets from the `NCI-Atlas collection of non-covalent interactions <http://www.nciatlas.org/>`__:

  - `D442x10 <http://www.nciatlas.org/D442x10.html>`__, dissociation curves for dispersion-bound van der Waals complexes
  - `SH250x10 <http://www.nciatlas.org/sh250.html>`__, dissociation curves for sigma-hole-bound van der Waals complexes
  - `R739x5 <http://www.nciatlas.org/r739.html>`__, compressed van der Waals complexes
  - `HB300SPXx10 <http://www.nciatlas.org/hb300spx.html>`__, dissociation curves for hydrogen-bound van der Waals complexes

- W4-CC, containing atomization energies of carbon clusters.\ :footcite:`karton2009`

For all training data, input density and derived meta-GGA features were computed from density matrices of converged B3LYP SCF calculations (def2-QZVP and ma-def2-QZVP basis sets) using a modified version of PySCF.

Training procedure
~~~~~~~~~~~~~~~~~~

Preprocessing
^^^^^^^^^^^^^

The training datapoints are preprocessed as follows.

- For each molecule, the density and derived meta-GGA features are computed from the density matrix of a converged B3LYP SCF calculation using a def2-QZVP or ma-def2-QZVP basis set in a modified version of PySCF.
- Density fitting was not applied.
- The density features were evaluated on an atom-centered integration grid of level 2 or level 3.
- The radial quadrature was performed with Treutler-Ahlrichs, Gauss-Chebyshev, Delley, or Mura-Knowles schemes based on Bragg atomic radii with Treutler-based radii adjustment.
- The angular grid points were pruned using the NWChem scheme.
- No density-based cutoff was applied; all grid points were retained for training.

Training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^

The training hyperparameter settings are detailed in the supplementary material of `Accurate and scalable exchange-correlation with deep learning (arXiv v5), Luise et al. 2025 <https://arxiv.org/abs/2506.14665v5>`__.
This repository only includes the code to evaluate the provided checkpoints, not the training code.

Speeds, sizes, times
^^^^^^^^^^^^^^^^^^^^

Training on the dataset described above took approximately 36 hours for 500k steps on an `NC A100 v4 series VM <https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic>`__ with 4 NVIDIA A100 GPUs (80 GB each), 96 CPU cores, 880 GB RAM, and a 256 GB disk.

The model checkpoints have :math:`\sim`\ 276k trainable parameters.

Evaluation
----------

Testing data, factors, and metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have evaluated our functional on several different benchmark sets:

#. W4-17. A diverse and highly accurate dataset of atomization energies.\ :footcite:`karton2017`
#. GMTKN55. A diverse and highly accurate dataset of general main-group thermochemistry, kinetics, and noncovalent interactions.\ :footcite:`goerigk2017`
#. Geometry optimization datasets: (a) CCse21, equilibrium structures, bond lengths, and bond angles;\ :footcite:`piccardo2015` (b) HMGB11, equilibrium structures and bond lengths;\ :footcite:`grimme2015` (c) LMGB35, equilibrium structures and bond lengths.\ :footcite:`grimme2015`
#. The dipole benchmark dataset from :footcite:`hait2018`.
#. Conformer search benchmark dataset of 22 molecules spanning 24 to 176 atoms, used for cost-scaling analysis, from :footcite:`grimme2019`.

These five benchmark types serve to measure different performance aspects of the functional.
Benchmarks 1 and 2 focus on the accuracy of predicted reaction energies, while 3 evaluates geometry optimization and convergence to reference equilibrium structures.
Benchmark 4 measures dipole moments, providing a proxy for the quality of the self-consistent electron density produced by the SCF procedure.
Finally, benchmark 5 assesses computational cost scaling with respect to system size.

The metrics for the different benchmark sets are:

#. Mean Absolute Error (MAE) in kcal/mol for reactions in W4-17 :math:`MAE = \frac{1}{N} \sum_{r=1}^N |\Delta E_r - \Delta E_r^\theta|`. Here *N* is the number of reactions in W4-17, *r* is the index denoting reactions in W4-17, :math:`\Delta E_r` is the energy difference of reaction r as calculated by a high-accuracy method from the W4 family (CCSDT(Q)/CBS to CCSDTQ56/CBS), and :math:`\Delta E_r^\theta` is the prediction of the reaction energy difference using SCF calculations with our functional.
#. Weighted total mean absolute deviations 2 (WTMAD-2) in kcal/mol for the GMTKN55 benchmark set :math:`\text{WTMAD-2} = \frac1{\sum^{55}_{i=1} N_i} \sum_{i=1}^{55} N_i \frac{56.84\text{ kcal/mol}}{\overline{|\Delta E|}_i} \text{MAE}_i` Here :math:`N_i` is the number of reactions in subset *i*, :math:`\overline{|\Delta E|}_i` is the average energy difference in subset *i* in kcal/mol and :math:`\text{MAE}_i` is the mean absolute error in kcal/mol for subset *i*.
#. For the geometry benchmark sets that report bond lengths, we measure the absolute error in bond lengths in Angstrom, averaged over the number of bonds and the number of equilibrium structures in the dataset. For the benchmark that also contains bond angles, we report the absolute error of the angles, averaged over the number of bonds and equilibrium structures in the dataset.
#. For the dipole benchmark, we follow the metrics defined in :footcite:`hait2018`. For molecules (indexed by *i*) for which only the reference magnitude of the dipole moment :math:`\mu_i^{\text{ref}} = |{\vec\mu}_i^{\text{ref}}|` is provided, the error is defined as :math:`\text{Error}_i = \frac{\mu_i^\theta - \mu_i^\text{ref}}{\max(\mu_i^\text{ref}, 1D)} \times 100\%`, where :math:`\mu_i^{\theta} = |{\vec\mu}_i^{\theta}|` is the predicted magnitude and *D* denotes the unit of Debye. For molecules for which the reference dipole vector :math:`\vec{\mu}_i^\text{ref}` is also available, we instead compute :math:`\text{Error}_i = \frac{|\vec{\mu}_i^\theta - \vec{\mu}_i^\text{ref}|}{\max(\mu_i^\text{ref}, 1D)} \times 100\%`. The RMSE is then :math:`\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \text{Error}_i^2}`.
#. We fit a power law of the form :math:`C(M) = \left(\frac{n(M)}{A}\right)^k` to the 22 data points of the test set where *C(M)* and *n(M)* are the computational cost and number of atoms of molecule *M*, respectively, and *A* and *k* are fitted parameters. We report the scaling power *k* as the main metric.

Evaluation results
~~~~~~~~~~~~~~~~~~

On W4-17, the Skala-1.0 functional predicts atomization energies at chemical accuracy (:math:`\sim`\ 1 kcal/mol MAE).
On GMTKN55, it achieves a WTMAD-2 of 3.89 kcal/mol, competitive with state-of-the-art range-separated hybrid functionals while only requiring runtimes typical of semi-local DFT.

On the geometry optimization benchmarks, the functional converges to reference equilibrium structures with errors comparable to a GGA.
On the dipole prediction benchmark, the error in dipole moment predictions is comparable to that of state-of-the-art range-separated hybrid functionals.

Finally, the scaling results show that the Skala-1.0 functional exhibits the asymptotic scaling behavior of a meta-GGA functional, with an approximate prefactor of 3 relative to r2SCAN.

License
-------

.. dropdown:: MIT License

   .. literalinclude:: ../../LICENSE.txt
      :lines: 3-

Citation
--------

When using Skala-1.0 in your research, please reference it including the version number as follows:

    This work uses the Skala-1.0 functional.

.. code:: bibtex

   @misc{luise2025,
      title={Accurate and scalable exchange-correlation with deep learning}, 
      author={Giulia Luise and Chin-Wei Huang and Thijs Vogels and Derk P. Kooi and Sebastian Ehlert and Stephanie Lanius and Klaas J. H. Giesbertz and Amir Karton and Deniz Gunceler and Megan Stanley and Wessel P. Bruinsma and Lin Huang and Xinran Wei and José Garrido Torres and Abylay Katbashev and Rodrigo Chavez Zavaleta and Bálint Máté and Sékou-Oumar Kaba and Roberto Sordillo and Yingrong Chen and David B. Williams-Young and Christopher M. Bishop and Jan Hermann and Rianne van den Berg and Paola Gori-Giorgi},
      year={2025},
      eprint={2506.14665v5},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2506.14665v5}, 
   }

Model card contact
------------------

- Rianne van den Berg, `rvandenberg@microsoft.com <mailto:rvandenberg@microsoft.com>`_
- Paola Gori-Giorgi, `pgorigiorgi@microsoft.com <mailto:pgorigiorgi@microsoft.com>`_
- Jan Hermann, `jan.hermann@microsoft.com <mailto:jan.hermann@microsoft.com>`_
- Sebastian Ehlert, `sehlert@microsoft.com <mailto:sehlert@microsoft.com>`_

References
----------

.. footbibliography::