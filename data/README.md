# Data read by the pipeline scripts

- `open_benchmark/MANIFEST.json`: the circuits of Benchmark II, the Lukin family of the open benchmark (the circuits of
  Bonilla Ataides et al. rebuilt from their description): per point (code distance, depth, noise model) 20 circuits with
  their layers, preparation and measurement bases, shots and the prefix decoded by the exact most-likely-error decoder.
  Only the Lukin family is kept; the other families of the open benchmark are not used by the paper.
- `open_benchmark/fig1b_values.csv`: the failure rates of the Lukin group's network and of their exact decoder, read from
  their Fig. 1b (columns `d,decoder,depth,value,std,N_implied,k_implied`).
- `first_test/MANIFEST_bench2.json`, `fresh_test/MANIFEST_bench2.json`: the Benchmark II test points of the first test
  and of the fresh test (new noise samples of the same circuits, new seeds). They are the output of
  `python scripts/eval_pipeline.py manifest --purpose bench2` and `python scripts/fresh_sets.py manifest --purpose bench2`.

The manifests of Benchmark I (`manifest --purpose dev|test`, and `fresh_sets.py manifest --purpose test`) are not stored:
the code regenerates them, and the circuits, seeds and shot counts it produces equal those of the paper's test sets. The
result files behind the paper's tables and figures are in `../paper/data/`.

The raw records of the tests and development stages are published separately (release `data-v1`); `RAW_DATA.md`
indexes them, and `raw/` holds the archive checksums, the per-archive manifests, the list of shot files left out with
their checksums, and the list of exact duplicates shipped once.
