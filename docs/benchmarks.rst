Expected speed: Skala timing benchmark
======================================

We have run single-point DFT calculations with Skala using (GPU4)PySCF on a dataset of realistic
molecules of increasing size. We compare them with two hybrid functionals and a meta-GGA to assess
their scaling and absolute performance.

See the `interactive report <benchmarks/index.html>`__ with our results on A100 GPUs and on CPU.


What is measured
----------------

The fixed protocol runs the same molecules, basis sets, and functionals in every environment.
Each worker warms process- and system-specific paths before the measured SCF so first-use costs
do not distort recurring timings.

The report separates four directly measured, nested layers: exchange-correlation functional
evaluation, numerical integration, the effective-potential build, and the full SCF iteration.
The stacked timing bands are differences between these measurements rather than attributed
estimates.

The molecule set
----------------

The 58 molecules span two to nearly two thousand atoms and are taken from five published
datasets. Their structures are the property of the authors of those datasets and are not
redistributed here: ``python -m skala.benchmark fetch-dataset`` downloads them from the original
sources, each pinned to a commit so that a rebuild is reproducible. The one source that a commit
cannot pin, the conformer-benchmark supporting information, is checked against a digest instead.

.. list-table::
   :header-rows: 1

   * - Molecules
     - Dataset
     - Source
   * - 14
     - GMTKN55
     - `grimme-lab/GMTKN55 <https://github.com/grimme-lab/GMTKN55>`__
   * - 22
     - Conformer benchmark
     - supporting information of `10.1021/acs.jctc.9b00143 <https://doi.org/10.1021/acs.jctc.9b00143>`__
   * - 16
     - LNCI16
     - `grimme-lab/benchmark-LNCI16 <https://github.com/grimme-lab/benchmark-LNCI16>`__
   * - 4
     - HS13L
     - `grimme-lab/benchmark-HS13L <https://github.com/grimme-lab/benchmark-HS13L>`__
   * - 2
     - S30L
     - `aoterodelaroza/refdata <https://github.com/aoterodelaroza/refdata>`__

The S30L structures are downloaded from a third-party mirror because the Grimme group's own
site rejects scripted downloads; they are the work of the authors cited below either way.

If you use this benchmark set, cite the datasets it is built from:

- **GMTKN55** — L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi, S. Grimme,
  *Phys. Chem. Chem. Phys.* **2017**, *19*, 32184. DOI: `10.1039/C7CP04913G
  <https://doi.org/10.1039/C7CP04913G>`__
- **Conformer benchmark** — S. Grimme, *J. Chem. Theory Comput.* **2019**, *15*, 2847.
  DOI: `10.1021/acs.jctc.9b00143 <https://doi.org/10.1021/acs.jctc.9b00143>`__.
  The supporting information is distributed under CC BY-NC 4.0.
- **HS13L** — J. Gorges, S. Grimme, A. Hansen, *Phys. Chem. Chem. Phys.* **2022**, *24*, 28831.
  DOI: `10.1039/D2CP04049B <https://doi.org/10.1039/D2CP04049B>`__
- **S30L** — R. Sure, S. Grimme, *J. Chem. Theory Comput.* **2015**, *11*, 3785.
  DOI: `10.1021/acs.jctc.5b00296 <https://doi.org/10.1021/acs.jctc.5b00296>`__
- **LNCI16** — J. Gorges, B. Bädorf, A. Hansen, S. Grimme, *Synlett* **2023**, *34*, 1135.
  DOI: `10.1055/s-0042-1753141 <https://doi.org/10.1055/s-0042-1753141>`__

LNCI16 is released under CC-BY-4.0 and asks that both the Synlett paper and the original source
of each individual structure be cited; those are listed in its
`DATA_SOURCES.md <https://github.com/grimme-lab/benchmark-LNCI16/blob/main/DATA_SOURCES.md>`__.

The published files give coordinates in angstrom to seven to ten decimals, which the fetch
converts to bohr. Structures therefore agree with the ones used for the reference report to about
1e-6 bohr — far below any chemically meaningful threshold, but not bit-identical.

Compare a local implementation or machine
-----------------------------------------

Use the same benchmark protocol on your own hardware or software stack, then compare your timings
against our fixed reference measurements in ``benchmarks/reference``. This is the expected workflow
for users who want to validate an implementation, check a new machine, or confirm that a local build
matches our published performance envelope.

Install Git LFS and run ``git lfs install`` once so the source checkout includes the reference
measurements; use ``git lfs pull`` to populate an existing checkout. Then install the benchmark
dependencies and fetch the molecule set once. Set threading variables before starting the run
because worker subprocesses inherit them:

.. code-block:: bash

   git lfs pull
   pixi install --locked -e default
   pixi run -e default python -m skala.benchmark fetch-dataset
   export OMP_NUM_THREADS=16

   pixi run -e default python -m skala.benchmark run benchmark-output \
      --env-id local-cpu \
      --env-label 'Local 16-core CPU' \
      --device cpu \
      --max-orbitals 250 \
      --time-limit 4h

Choose the physical core count appropriate for the machine. GPU runs require a compatible CUDA,
CuPy, and GPU4PySCF installation; requesting ``--device gpu`` fails instead of silently falling
back to the CPU. ``--time-limit`` applies to each DFT calculation, not the whole sweep.

For a quick smoke test, use a separate output directory and restrict both molecule size and basis:

.. code-block:: bash

   pixi run -e default python -m skala.benchmark run benchmark-smoke \
     --env-id cpu-smoke \
     --device cpu \
     --max-atoms 3 \
     --basis def2-svp

Filters are part of the sweep identity, so do not use a restricted run to resume a full sweep.

The fetch takes a few seconds and needs network access; the run itself does not. It writes to a
cache directory, which ``--dataset-dir`` on both commands overrides, as does the
``SKALA_BENCHMARK_DATASET_DIR`` environment variable. On a cluster, fetch once to a shared
location and point every shard at it, so the compute nodes need no network access of their own.

.. note::

   On a cluster, shard the timing sweep by starting one job for each ``--shard-index`` from
   ``0`` through ``N - 1`` and passing the same ``--num-shards N``, environment ID, and shared
   output directory to every job. The shards contain disjoint calculations; wait for all of them
   to finish before collecting the shared output. Rerunning a shard skips completed calculations,
   while incompatible sweep settings are rejected. Skala does not submit these jobs; use your
   cluster's existing scheduler to launch the shard commands.

.. code-block:: bash

   python -m skala.benchmark collect benchmark-output

   # Create a report with both our reference timings and your local timings.
   python -m skala.benchmark report local-report \
     benchmarks/reference \
     benchmark-output/collected

The collected JSON is written to ``benchmark-output/collected`` by default. The local report uses
neutral explanatory text and does not reuse the interpretations written for the official report.
The generated report bundle has no network dependencies.

Serve the generated report over HTTP and open ``http://localhost:8000/`` in your browser:

.. code-block:: bash

   python -m http.server 8000 --directory local-report
