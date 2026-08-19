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

Compare a local implementation or machine
-----------------------------------------

Use the same benchmark protocol on your own hardware or software stack, then compare your timings
against our fixed reference measurements in ``benchmarks/reference``. This is the expected workflow
for users who want to validate an implementation, check a new machine, or confirm that a local build
matches our published performance envelope.

Install the benchmark dependencies from a source checkout. Set threading variables before starting
the run because worker subprocesses inherit them:

.. code-block:: bash

   pip install -e '.[benchmark]'
   export OMP_NUM_THREADS=16

   python -m skala.benchmark run benchmark-output \
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

   python -m skala.benchmark run benchmark-smoke \
     --env-id cpu-smoke \
     --device cpu \
     --max-atoms 3 \
     --basis def2-svp

Filters are part of the sweep identity, so do not use a restricted run to resume a full sweep.

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
