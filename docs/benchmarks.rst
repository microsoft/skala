Runtime speed benchmarks
========================

We have run single-point DFT calculations with Skala using (GPU4)PySCF on a dataset of realistic
molecules of increasing size. We compare them with two hybrid functionals and a meta-GGA to assess
their scaling and absolute performance.

See the `interactive report <benchmarks/index.html>`__ with our results on A100 GPUs and on CPU.


Compare a local implementation or machine
-----------------------------------------

You can run the same benchmarks we ran on your own infrastructure to validate your implementation.
To do this, install the benchmark dependencies, run the benchmark, collect the result,
and combine them with the baselines from our report that we store in ``benchmarks/reference``.

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
