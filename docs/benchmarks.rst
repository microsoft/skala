Expected speed: Skala timing benchmark
=====================================

We have run single-point DFT calculations with Skala using (GPU4)PySCF on a dataset of realistic
molecules of increasing size. We compare them with two hybrid functionals and a meta-GGA to assess
their scaling and absolute performance.

See the `interactive report <benchmarks/index.html>`__ with our results on A100 GPUs and on CPU.


Compare a local implementation or machine
-----------------------------------------

Use the same benchmark protocol on your own hardware or software stack, then compare your timings
against our fixed reference measurements in ``benchmarks/reference``. This is the expected workflow for
users who want to validate an implementation, check a new machine, or confirm that a local build matches
our published performance envelope.

To do this, install the benchmark dependencies, run the benchmark, collect the result, and combine your
local measurements with the public baseline data stored in ``benchmarks/reference``.

.. code-block:: bash

   pip install -e '.[benchmark]'

   python -m skala.benchmark run benchmark-output \
     --env-id local-cpu \
     --env-label 'Your laptop CPU' \
     --device cpu \
     --max-orbitals 250

.. note::

   On a cluster, shard the timing sweep by starting one job for each ``--shard-index`` from
   ``0`` through ``N - 1`` and passing the same ``--num-shards N``, environment ID, and shared
   output directory to every job. The shards contain disjoint calculations; wait for all of them
   to finish before collecting the shared output.

.. code-block:: bash

   python -m skala.benchmark collect benchmark-output

   # Create a report with both our reference timings and your local timings.
   python -m skala.benchmark report local-report \
     benchmarks/reference \
     benchmark-output/collected

Serve the generated report over HTTP and open ``http://localhost:8000/`` in your browser:

.. code-block:: bash

   python -m http.server 8000 --directory local-report
