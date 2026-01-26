Skala: Accurate and scalable exchange-correlation with deep learning
====================================================================

Overview
--------

*Skala* is a deep-learning exchange-correlation functional designed to provide chemical accuracy (less than 1 kcal/mol) for a wide range of chemical systems without using expensive non-local features like exact exchange or hand-crafted density convolutions.
The model is trained on a large dataset of highly accurate total atomization energies, thermochemical properties, like ionization potentials and proton affinities, conformer energies, reaction paths, and non-covalent interactions.

*Skala* is still in active development, we are working on improving the model accuracy and also on integrating it into quantum chemistry packages.
Please stay tuned for updates and new releases.


.. admonition:: Learn more
   :class: important

   To learn more about the Skala functional, check out the `blog post <https://aka.ms/skaladft/blog>`__ or the `preprint <https://aka.ms/skaladft/preprint>`__.


.. toctree::
   :maxdepth: 1
   :caption: User guide
   :hidden:

   installation
   pyscf/singlepoint
   pyscf/scf_settings
   ase
   foundry
   gauxc/index

.. toctree::
   :maxdepth: 1
   :caption: References
   :hidden:

   Skala preprint <https://aka.ms/skaladft/preprint>
   Breaking bonds, breaking ground <https://aka.ms/skaladft/blog>
