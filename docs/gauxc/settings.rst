GauXC settings
==============

GauXC provides different settings for defining the behavior of the library.
Mainly these settings are used to define the molecular integration scheme and the parallelization strategy.

.. _gauxc_molecular_grid_settings:

Molecular grid
--------------

For creating a ``GauXC::MolGrid`` we need to specify the following parameters:

``AtomicGridSizeDefault``
    Defines the number of angular and radial points in the atomic grid.
    Possible options are ``Fine`` (75, 302), ``UltraFine`` (99, 590), ``SuperFine`` (250, 974), ``GM3`` (35, 110) and ``GM5`` (50, 302).

``RadialQuad``
    Defines the radial quadrature scheme to be used for the atomic grid.
    Possible options are ``Becke``,\ :footcite:`becke1988` ``MuraKnowles``,\ :footcite:`mura1996` ``TreutlerAhlrichs``,\ :footcite:`treutler1995` and ``MurrayHandyLaming``.\ :footcite:`murray1993`

``PruningScheme``
    Defines the pruning scheme to construct the molecular grid weights from the atomic grids.
    Possible options are ``Unpruned``, ``Robust``, and ``Treutler``.

``BatchSize``
    Defines the batch size for processing grid points in parallel.
    Default is 512 points per batch, however larger values up around 10000 are recommended for better performance.

References
----------

.. footbibliography::