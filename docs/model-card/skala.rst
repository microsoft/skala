Skala model
===========

Model details
-------------

In pursuit of the universal functional for density functional theory (DFT), the OneDFT team from Microsoft Research AI for Science has developed the Skala exchange-correlation functional, as introduced in `Accurate and scalable exchange-correlation with deep learning, Luise et al. 2025 <https://arxiv.org/abs/2506.14665v5>`__.
The functional is modeled with a neural network architecture that takes as input features on a 3D grid to describe the electron density and derived meta-generalized-gradient (meta-GGA) features.
The architecture performs scalable non-local message-passing on the integration grid via a second coarser grid, combined with shared local layers to allow for representation learning of both local and non-local features.
These are then used to predict the exchange-correlation energy in an end-to-end data-driven manner.
This approach departs from the traditional route of incorporating increasingly expensive hand-designed non-local features from the Jacob's ladder into functional forms to increase their accuracy.
Instead, we use a more modern deep learning approach with a scalable neural network that uses only cheap input features to learn the necessary non-local representations.
To facilitate this learning, the model is trained on a dataset of unprecedented size, containing highly accurate energy labels from coupled cluster theory.
The largest subset focuses on atomization energies and is generated in collaboration with the University of New England.
This subset is released as part of the Microsoft Research Accurate Chemistry Collection (MSR-ACC, `Accurate Chemistry Collection: Coupled cluster atomization energies for broad chemical space, Ehlert et al. 2025 <https://arxiv.org/abs/2506.14492v5>`__).
To increase the coverage of other types of chemistry, the training dataset is further complimented with in-house generated datasets covering conformers, ionization potentials, proton affinities and elementary reactions, as well as a small amount of publicly available highly accurate data.
We demonstrate that the combination of a large-scale high-accuracy dataset combined with our deep learning architecture produces the Skala functional that predicts atomization energies at chemical accuracy (1 kcal/mol), as measured on the public benchmark set W4-17.
On the public benchmark set GMTKN55, which covers general-main group thermochemistry, kinetics and noncovalent interactions, our model makes predictions around 3.89 kcal/mol.
This accuracy is competitive with state-of-the-art range-separated hybrid functionals, while only requiring a cost comparable to semi-local DFT.
With this work we demonstrate the viability of our approach to pursue the universal density functional across all of chemistry.

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

#.	The Skala functional is being shared with the research community to facilitate reproduction of evaluations with our model.
#.	Evaluating the energy difference of a reaction by computing the total energy of all compounds in the reaction using a Self-Consistent-Field (SCF) evaluation with the Skala exchange correlation functional. 
#.	Evaluating the total energy of a molecule using a Self-Consistent-Field (SCF) evaluation with the Skala exchange correlation functional. Note that like all density functionals, energy differences are predicted much more reliably than total energies of single molecules. 
#.	The Self-Consistent-Field evaluation we provide is implemented using PySCF, which runs the functional inference on CPU. We also provide the traced Skala functional such that other open-source implementations that are more optimized in terms of memory and speed and are compatible with other hardware such as GPUs have the option to integrate the functional in their SCF pipelines, for instance through GauXC. We included a fork of GauXC that is compatible with our functional in the repository.

Out-of-scope uses
-----------------

#.	Evaluating the functional with a single pass given a fixed density as input is not the intended way to evaluate the model. The model's predictions should always be made by using it as part of an SCF procedure. 
#.	We do not include a training pipeline for the Skala functional in this code base.

Risks and limitations
---------------------

1.	Interpretation of results requires expertise in quantum chemistry.
2.	The Skala functional is trained on atomization energies, conformers, proton affinities, ionization potentials, elementary reaction pathways, non-covalent interactions, as well as a small amount of electron affinities and total energies of atoms. We have benchmarked the performance on the public benchmark W4-17 for atomization energies, as well as the public benchmark set GMTKN55, which covers general-main group thermochemistry, kinetics and noncovalent interactions, to provide an indication of how it generalizes outside of the training set. We have also measured robustness on dipole moment predictions and geometry optimization.
3.	The Skala functional has been trained on data that contains the following elements of the periodic table: H-Ar, Br, Kr, I, Xe. We have tested it on data containing the elements H-Ca, Ge-Kr, Sn-I, Pb, Bi.
4.	Given the above point 2 and 3, we remind the user that this is not a production model. Therefore, we advise testing the functional further before applying it to your research. We welcome any feedback!

Recommendations
---------------

1.	In our PySCF supported SCF implementation, the largest system that was tested with our functional contained 180 atoms, for which we used the def2-TZVP basis set (~5000 orbitals) and `Eadsv5 series <https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/eadsv5-series?tabs=sizebasic>`__ virtual machines. When testing the functional on larger systems there is a risk of running out of memory.
2.	For more optimized implementations in terms of memory and speed, as well as a GPU supported implementation, we recommend integrating the functional with other more optimized open-source SCF packages, for instance through GauXC. We included a fork of GauXC that is compatible with our functional in the repository.
3.	Skala will also be available through `Azure AI Foundry <https://labs.ai.azure.com/projects/skala/>`__ it is coupled with Microsoft's in-house developed GPU `Accelerated DFT <https://arxiv.org/abs/2406.11185>`__ application.


Training details
----------------

Training data
~~~~~~~~~~~~~

The following data is included in our training set:

- 99% of MSR-ACC:TAE (+-78k reactions) containing atomization energies.\
  This data was generated in collaboration with Prof. Amir Karton, University of New England, with the W1-F12 composite protocol based on CCSD(T) and is released as part of the `Microsoft Research Accurate Chemistry Collection <https://arxiv.org/abs/2506.14492v5>`__ (MSR-ACC).
- Total energies, electron affinities and ionization potentials (up to triple ionization) for atoms, from H to Ar (excluding Li and Be because of basis set constraints).\
  This data was produced in-house with CCSD(T) by extrapolating to the complete basis set limit from quadruple zeta (QZ) and pentuple zeta (5Z) basis set calculations.\
  The basis sets used for H and He were aug-cc-pV(Q+d)Z, aug-cc-pV(5+d), while for the remaining elements B-Ar the basis sets used were aug-cc-pCVQZ and aug-cc-pCV5Z.\
  All basis sets were obtained from the `Basis Set Exchange (BSE) <https://www.basissetexchange.org/>`__.\
  Extrapolation of the correlation energy was performed by fitting a simple Z^(-3) expression, while extrapolation of the Hartree-Fock energy was performed using the two-point extrapolation suggested in :footcite:`karton2006`.
- Four datasets from the `NCI-Atlas collection of non-covalent interactions <http://www.nciatlas.org/>`__: 

  - `D442x10 <http://www.nciatlas.org/D442x10.html>`__, dissociation curves for dispersion bound van-der-Waals complexes
  - `SH250x10 <http://www.nciatlas.org/sh250.html>`__, dissociation curves for sigma-hole bound van-der-Waals complexes
  - `R739x5 <http://www.nciatlas.org/r739.html>`__, compressed van-der-Waals complexes
  - `HB300SPXx10 <http://www.nciatlas.org/hb300spx.html>`__, dissociation curves for hydrogen bound van-der-Waals complexes

- W4-CC, containing atomization energies of carbon clusters provided in `Atomization energies of the carbon clusters Cn (n = 2-10) revisited by means of W4 theory as well as density functional, Gn, and CBS methods, Karton et al., Mol. Phys. 2009 <https://doi.org/10.1080/00268970802708959>`__. 

For all training data we have created input density and derived meta-GGA features using density matrices of converged SCF calculations with the B3LYP functional (def2-QZVP and ma-def2-QZVP basis set) using a modified version of the PySCF software package.  

Training procedure
~~~~~~~~~~~~~~~~~~

Preprocessing
^^^^^^^^^^^^^

The training datapoints are preprocessed as follows.

-	For each molecule the density and derived meta-GGA features are computed based on the density matrix of converged SCF calculations with the B3LYP functional using a def2-QZVP or ma-def2-QZVP basis set using a modified version of the PySCF software package.
-	Density fitting was not applied for the SCF calculation.
-	The density features were evaluated on an atom centered integration grid of level 2 or level 3.
-	The radial integral was performed with the Treutler-Ahlrichs, Gauss-Chebychev, Delley, or Mura-Knowles based on Bragg atomic radii with Treutler based radii adjustment.
-	The angular grid points were pruned using the NWChem scheme.
-	No density based cutoff was applied and all grid points were retained for training. 

Training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^

The training hyperparameter settings are detailed in the supplementary of `Accurate and scalable exchange-correlation with deep learning, Luise et al. 2025 <https://arxiv.org/abs/2506.14492v5>`__.
This repository only includes the code to evaluate the checkpoints provided, not the training code.

Speeds, sizes, times
^^^^^^^^^^^^^^^^^^^^

The training of our functional using the training dataset as detailed in the section "Training data" took approximately 36h for 500k training steps on a `NC A100 v4 series VM <https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic>`__ with 4 NVIDIA A100 GPU with 80 GB memory each, 96 CPU cores, 880 GB RAM,  and a 256 GB disk. 

The model checkpoints have +-276,001 trainable parameters. 

Evaluation
----------

Testing data, factors, and metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have evaluated our functional on several different benchmark sets:

#.	W4-17. A diverse and highly accurate dataset of atomization energies.\ :footcite:`karton2017`
#.	GMTKN55. A diverse and highly accurate dataset of general main group thermochemistry, kinetics and noncovalent interactions.\ :footcite:`goerigk2017`
#.	Geometry optimization datasets: (a) CCse21, equilibrium structures, bond lengths and bond angles.\ :footcite:`piccardo2015` (b) HMGB11, equilibrium structures, bond lengths.\ :footcite:`grimme2015` (c) LMGB35, equilibrium structures, bond lengths.\ :footcite:`grimme2015`
#.	The Dipole benchmark dataset from :footcite:`hait2018`.
#.	Conformer search benchmark dataset of 22 molecules spanning molecular size from 24 to 176 atoms for cost scaling from :footcite:`grimme2019`.

The evaluation of our model using the 5 different types of benchmarks as defined above serve to measure different performance aspects of our functional.
For example, 1 and 2 focus on the accuracy of predicted reaction energies, and 3 focuses on the ability of our functional to perform geometry optimization and to converge to the right equilibrium molecular structure.
Furthermore, 4 measures the dipole moment of the molecules in the benchmark set, which provides a measure for the quality of the self-consistent electron density that a converged SCF procedure produces with our model.
Finally, 5 determines the speed of employing SCF with our model and compares its scaling behavior with respect to system size with the scaling of traditional functionals.

The metrics for the different benchmark sets are:

#.	Mean Absolute Error (MAE) in kcal/mol for reactions in W4-17 :math:`MAE = \frac{1}{N} \sum_{r=1}^N |\Delta E_r - \Delta E_r^\theta|`. Here *N* is the number of reactions in W4-17, *r* is the index denoting reactions in W4-17, :math:`\Delta E_r` is the energy difference of reaction r as calculated by a high-accuracy method from the W4 family (CCSDT(Q)/CBS to CCSDTQ56/CBS), and :math:`\Delta E_r^\theta` is the prediction of the reaction energy difference using SCF calculations with our functional, and
#. Weighted total mean absolute deviations 2 (WTMAD-2) in kcal/mol for the GMTKN55 benchmark set :math:`\text{WTMAD-2} = \frac1{\sum^{55}_{i=1} N_i} \sum_{i=1}^{55} N_i \frac{56.84\text{ kcal/mol}}{\overline{|\Delta E|}_i} \text{MAE}_i` Here :math:`N_i` is the number of reactions in subset *i*, :math:`\overline{|\Delta E|}_i` is the average energy difference in subset *i* in kcal/mol and :math:`\text{MAE}_i` is the mean absolute error in kcal/mol for subset *i*.
#. For the geometry benchmark sets that report bond lengths, we measure the absolute error in bond lengths in Angstrom, averaged over the number of bonds and the number of equilibrium structures in the dataset. For the benchmark that also contains bond angles, we report the absolute error of the angles, averaged over the number of bonds and equilibrium structures in the dataset.
#. We follow the metrics defined in :footcite:`hait2018`. We measure the Root Mean Squared Error (RMSE) of the dipole moment with respect to reference values provided by the benchmark dataset. For those molecules (indexed with *i*) for which only the reference magnitude of the dipole moment :math:`\mu_i^{\text{ref}} = |{\vec\mu}_i^{\text{ref}}|` is provided, we measure the RMSE of the predicted magnitude of the dipole moment :math:`\mu_i^{\theta} = |{\vec\mu}_i^{\theta}|` is available, the error is defined as :math:`\text{Error}_i = \frac{\mu_i^\theta - \mu_i^\text{ref}}{\max(\mu_i^\text{ref}, 1D)} \times 100\%`. Here *D* denotes the unit of Debye. For those molecules for which the reference value of the dipole vector :math:`\vec{\mu}_i^\text{ref}` is also available we instead compute :math:`\text{Error}_i = \frac{|\vec{\mu}_i^\theta - \vec{\mu}_i^\text{ref}|}{max(\mu_i^\text{ref}, 1D)} \times 100\%`. Using these errors we compute the RMSE as follows: :math:`\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \text{Error}_i^2}`
#. We fit a power law of the form :math:`C(M) = \left(\frac{n(M)}{A}\right)^k` to the 22 data points of the test set where *C(M)* and *n(M)* are the computational cost and number of atoms of molecule *M*, respectively, and *A* and *k* are fitted parameters. We report the scaling power *k* as the main metric.

Evaluation results
~~~~~~~~~~~~~~~~~~

We demonstrate that the combination of a large-scale high-accuracy dataset combined with our deep learning architecture produces the Skala functional that predicts atomization energies at chemical accuracy (1 kcal/mol), as measured on the public benchmark set W4-17.
On the public benchmark set GMTKN55, which covers general-main group thermochemistry, kinetics and noncovalent interactions, our model makes predictions around 3.89 kcal/mol.
This accuracy is competitive with state-of-the-art range-separated hybrid functionals while only requiring runtimes typical of semi-local DFT. 

On the geometry optimization benchmarks we demonstrate that we can converge to the reference equilibrium structure with an error that is comparable to a GGA.
On the dipole prediction benchmark test we demonstrate that the error of our dipole moment prediction with respect to reference values is comparable state-of-the-art range-separated hybrid functionals.

Finally, our scaling results demonstrate that our functional shows the asymptotic scaling behavior of a metaGGA functional, with an approximate prefactor of 3 compared to the r2SCAN.

License
-------

.. dropdown:: MIT License

   .. literalinclude:: ../../LICENSE.txt
      :lines: 3-

Citation
--------

.. code:: bibtex

   @misc{luise2025,
      title={Accurate and scalable exchange-correlation with deep learning}, 
      author={Giulia Luise and Chin-Wei Huang and Thijs Vogels and Derk P. Kooi and Sebastian Ehlert and Stephanie Lanius and Klaas J. H. Giesbertz and Amir Karton and Deniz Gunceler and Megan Stanley and Wessel P. Bruinsma and Lin Huang and Xinran Wei and José Garrido Torres and Abylay Katbashev and Rodrigo Chavez Zavaleta and Bálint Máté and Sékou-Oumar Kaba and Roberto Sordillo and Yingrong Chen and David B. Williams-Young and Christopher M. Bishop and Jan Hermann and Rianne van den Berg and Paola Gori-Giorgi},
      year={2025},
      eprint={2506.14665},
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