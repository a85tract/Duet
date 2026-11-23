# Data for the tables and figures of the paper

Everything here is plain text (Markdown tables, JSON, gzip-compressed JSON). The scripts in `../scripts` read these files
and nothing else. Each Markdown file holds only the tables that the scripts parse, with the numbers copied unchanged from
the evaluation runs. Prose and tables that no script reads were removed.

## first_test/

The first evaluation on the held-out test circuits. The operating points (routed fraction f and cap per layer) were
frozen on development data before this test was scored. Benchmark I is the general random Clifford circuits and
Benchmark II is the Lukin group's circuits. "Grace" and "Spark" are the two platforms of the composition.

- `results.md`: per distance d, the pooled failure ratios and failure counts of the network alone and of the
  composition against LoMatching, Cain et al. and BP+LoMatching (Benchmark I), the per-system results with measures (a)
  and (b) (Benchmark II), and the sweep over routed fractions on the Grace platform.
- `choices.md`: the operating point of each benchmark, distance and platform, with its development failures, time per
  layer and the speed bar (the Lukin group's network on an H100) it is judged against.
- `appendix_tables.md`: the same results per point (Benchmark I: block count K, depth L, noise p) and per depth
  (Benchmark II), including the published rates of the Lukin group's network.
- `network_time_batch256/<gpu>/benchmark<1,2>_d<d>.json`: measured time (microseconds per batch of 256 shots) of the
  network alone for each development point, on an RTX PRO 6000 (`rtx_pro_6000`) and on the GB10 of a DGX Spark (`gb10`).
  Keys are `d<d>_K<K>_L<L>` (Benchmark I) and `benchmark2_d<d>_K4_L<L>` (Benchmark II).

## training/

- `ablations.md`: scores of the network variants trained at d = 5 (no exchange, exchange only at the read-out, pooled
  read-out, small trunk, specialists for one block count K), each as a failure ratio against the full network, with 95 %
  intervals and failure counts (Table V).
- `networks.md`: parameter count of every trained network (Table V's second column).

## fresh_test/

The second, independent evaluation on a fresh test set (new circuits and noise samples), scoring every routed fraction at
the cap chosen on development data.

- `development_best.json`: per benchmark, platform, distance and routed fraction, the development result of the best cap
  (failures, time per layer, speed bar). The operating points are picked from it.
- `d<d>/benchmark<1,2>/report.md`: the fresh-test rows for each platform (Basis grace, Basis spark): failures, ratios with
  95 % intervals, time per layer (p99 batch time in ms), routed shots, corrected and introduced failures. Mark B is the
  development-best cap of that f, mark F the first test's operating point.
- `d<d>/benchmark<1,2>/grid.json.gz`: pooled ratios and the routing counts per cell (Tables I and III, Figure 3).
- `d<d>/benchmark2/mle_measures.json.gz`: Benchmark II's measures (a) and (b) per system and depth (Table II).
- `live_rtx_pro_6000.md`, `live_dgx_spark.md`: the live paced stream (development shots) at the operating points, on an
  RTX PRO 6000 with 32 EPYC 9555 cores and on a DGX Spark with 20 Arm cores, compared with the simulated stream
  (Table XX).

Benchmark II covers d = 3, 5, 7 only, so `d9/` has no `benchmark2/`.
