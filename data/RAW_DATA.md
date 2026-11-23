# Raw data

The paper's tables and figures are computed from the result files in `paper/data/`. Those files summarize much larger raw
records: the per-shot output of every decoder we ran, the timing records of the paced stream, the job records of every
test and development stage, and the networks trained on the way to the ones in `networks/`. The raw records are
published as tar archives with the GitHub release `data-v1`:

https://github.com/a85tract/Duet/releases/tag/data-v1

They hold 759,805 files, 73.71 GB unpacked, in 42 archives of 15.98 GB in all. The stored shots
(the sampled detection events) are not in them. The released code regenerates the shot sets of the paper's tests and
development data from their seeds, and `raw/detection_events.tsv` gives the size and SHA-256 of every stored shot file
(section "Detection events" below).

## Download and check

Download the archives from the release page, or with the GitHub command-line tool:

    gh release download data-v1 --repo a85tract/Duet --dir raw_data                          # everything
    gh release download data-v1 --repo a85tract/Duet --dir raw_data --pattern 'duet-raw-fresh-test*'

Check the archives against `raw/SHA256SUMS` (the same list is a release asset), then unpack each one into an empty
folder. Archives ending in `.tar.zst` are zstd-compressed (`zstd` 1.4 or later); `.tar` archives are not compressed.

    cd raw_data
    sha256sum -c --ignore-missing ../data/raw/SHA256SUMS
    mkdir out
    for f in duet-raw-*.tar.zst; do zstd -dc "$f" | tar -xf - -C out; done
    for f in duet-raw-*.tar; do tar -xf "$f" -C out; done

`raw/manifests/<archive>.tsv.gz` (the archive name without `.tar` or `.tar.zst`) lists every member of an archive with
its size in bytes and its SHA-256. To check the unpacked files of one archive:

    zcat ../data/raw/manifests/duet-raw-fresh-test-part01.tsv.gz | tail -n +2 | awk -F'\t' '{print $3 "  " $1}' \
      | (cd out && sha256sum -c --quiet) && echo all files match

## Archives

| archive | holds | size | files | unpacked | SHA-256 |
|---|---|---|---|---|---|
| `duet-raw-fresh-test-part01.tar.zst` | fresh test: Tables I-III, XVII-XIX, Fig. 3 | 89.23 MB | 31,315 | 3.50 GB | `a503c4f407a3d879952ca133171a6c1044f3cdca628a0c829ad9f655f608d3db` |
| `duet-raw-fresh-test-part02.tar.zst` | fresh test: Tables I-III, XVII-XIX, Fig. 3 | 445.42 MB | 7,296 | 3.43 GB | `cb0332a6eab5d2e4a60565d22a5d288c776775d5c99017344c18d46dcf3fc086` |
| `duet-raw-fresh-test-part03.tar.zst` | fresh test: Tables I-III, XVII-XIX, Fig. 3 | 798.88 MB | 10,720 | 3.50 GB | `9fa3bcd0e95fa8b4f46d42c4674521fcd94640362d8fb08f4c039917a3c6b1f6` |
| `duet-raw-fresh-test-part04.tar.zst` | fresh test: Tables I-III, XVII-XIX, Fig. 3 | 310.87 MB | 40,000 | 1.34 GB | `c1d09e9f0fdd0d1da98fa84f36c31d9ec5e39702a84582a775a99b88ab045795` |
| `duet-raw-fresh-test-part05.tar.zst` | fresh test: Tables I-III, XVII-XIX, Fig. 3 | 220.52 MB | 11,676 | 798.66 MB | `c44e163236b2451614d888f5121c617f6bb6f0556f8a4bd40cbe9b8ea42dd8e0` |
| `duet-raw-first-test-part01.tar.zst` | first test: Appendix D, Tables VII-XVI, XXI | 92.39 MB | 40,000 | 3.46 GB | `300ea4f3c21b9b9422a81420a82f067353a268864347797d2d46ae76ed97f09d` |
| `duet-raw-first-test-part02.tar.zst` | first test: Appendix D, Tables VII-XVI, XXI | 454.46 MB | 7,678 | 3.50 GB | `358563cd25ab47b3147cabe41c30d279c024e4ca22feb696d1ad00778373a35d` |
| `duet-raw-first-test-part03.tar.zst` | first test: Appendix D, Tables VII-XVI, XXI | 861.58 MB | 8,252 | 3.50 GB | `f8d4388fc177befde06e7458a564e250fd97773666fc9d39ebc69b163f4c6972` |
| `duet-raw-first-test-part04.tar.zst` | first test: Appendix D, Tables VII-XVI, XXI | 884.73 MB | 40,000 | 3.40 GB | `0b54ad6626fab73967805b2f230346a4b04482ac5d00f56b5ccee7fe52199bb8` |
| `duet-raw-first-test-part05.tar.zst` | first test: Appendix D, Tables VII-XVI, XXI | 642.51 MB | 40,000 | 2.08 GB | `28c86b76caee91ab10b40a542df753d3db40a480609237ffc32f17245e86ffd9` |
| `duet-raw-first-test-part06.tar.zst` | first test: Appendix D, Tables VII-XVI, XXI | 19.10 MB | 1,981 | 79.78 MB | `d981e8b7f24a319c74c55c2876289600499926cc29f730cf6e82480edf7ccad7` |
| `duet-raw-operating-points-part01.tar.zst` | development data of the operating points: Tables XVII, XXI | 599.18 MB | 16,489 | 3.50 GB | `bba078879b5f607224f1fa35aa087acd72fb17c5f8524f062c6d291c89126e4b` |
| `duet-raw-operating-points-part02.tar.zst` | development data of the operating points: Tables XVII, XXI | 1.01 GB | 29,823 | 3.50 GB | `c42ce67ac646e363534684ae442e024525fadbfd49b0a7b93bfcc706ffb0e64e` |
| `duet-raw-operating-points-part03.tar.zst` | development data of the operating points: Tables XVII, XXI | 164.25 MB | 5,903 | 490.76 MB | `13b61d2bdb6a8cfa49dcf31c4a5c430f1aaa84e8af2e630ce836b4835f7799b8` |
| `duet-raw-timing-part01.tar.zst` | speed: Tables IV, XX, Fig. 4, Appendix E | 38.78 MB | 5,527 | 158.50 MB | `8b1dd97f5228821cce8142a341e98d4c019b6757f30833bdc3e1287af5439db1` |
| `duet-raw-timing-part02.tar.zst` | speed: Tables IV, XX, Fig. 4, Appendix E | 33.03 MB | 1,400 | 123.24 MB | `d0972c2fad8a68e63440a39362b6f5da13a385e231ee8c265a5acee4c4bcda5c` |
| `duet-raw-training-part01.tar.zst` | training records of the twelve networks | 57.32 MB | 208 | 64.90 MB | `8e5dd16d20c74b38b79bf9c8e2d7fc4fa9bb6573d4e9df325169ea96761de9a0` |
| `duet-raw-training-part02.tar.zst` | training records of the twelve networks | 22.18 MB | 7 | 24.28 MB | `c65716e4d062578eb367051c69f07d80cc52bfcf23f6ffc324207af47aadf159` |
| `duet-raw-ablations.tar.zst` | variants' reads: Table V | 499.97 MB | 211 | 1.60 GB | `23af97da21930923ae149d486f4baaf86e7c1a0d031611d4196bbe770f4b9ebc` |
| `duet-raw-open-benchmark-part01.tar.zst` | open benchmark, source of Benchmark II | 42.56 MB | 40,000 | 312.76 MB | `bf8b8d58cce2d72e011aa5deafa1fb7014d51ddaa8218b8c9a2a8a9a9da1f668` |
| `duet-raw-open-benchmark-part02.tar.zst` | open benchmark, source of Benchmark II | 357.36 MB | 40,000 | 2.46 GB | `7ddf52c2695969846ff2f0cf1fc4669577e36ddfa766b6bf3a454222768e4a92` |
| `duet-raw-open-benchmark-part03.tar.zst` | open benchmark, source of Benchmark II | 281.27 MB | 40,000 | 1.64 GB | `342c5443489c73a59dc3deecaf9263179f4f77f6b75147557eb8ae8b0882625e` |
| `duet-raw-open-benchmark-part04.tar.zst` | open benchmark, source of Benchmark II | 342.80 MB | 40,000 | 2.31 GB | `ef4ddc2078d6a170095f53acfdfcc0d74a5128b3d6ae087446f6987bc114417a` |
| `duet-raw-open-benchmark-part05.tar.zst` | open benchmark, source of Benchmark II | 204.39 MB | 31,687 | 1.04 GB | `7682f256e2e6dd885af5c3bda3116893635d75072ae29e1827516b20a3742d29` |
| `duet-raw-development-1.tar.zst` | development stage 1 (superseded) | 116.60 MB | 144 | 233.25 MB | `44ce6b8fabbffe8b9ead8f3da103201ed1dad65c72225ec6d2ded2686a9a0256` |
| `duet-raw-development-2-part01.tar.zst` | development stage 2 (superseded) | 131.82 MB | 3,194 | 189.43 MB | `7a8924bc10953d2c23b505034a45af3fed591c5a27fdde053d2a8f2bf52a15e3` |
| `duet-raw-development-2-part02.tar.zst` | development stage 2 (superseded) | 42.61 MB | 6,061 | 80.15 MB | `cb835f91d0ef110a0e13c3a794fe86406f8e385c69ae2b0d054400d9eb8147a3` |
| `duet-raw-development-3.tar.zst` | development stage 3 (superseded) | 204.21 MB | 287 | 528.31 MB | `7b3865a22698054248aa87d08efeb4c129f9ca12f6d29cdc6cdefd931db440a0` |
| `duet-raw-development-4-part01.tar.zst` | development stage 4 (superseded) | 17.79 MB | 73 | 20.12 MB | `1977e6cbf0bae68f9926958b4b7e14eb6bb1cad51dadeee20d4e5afe73283649` |
| `duet-raw-development-4-part02.tar.zst` | development stage 4 (superseded) | 876.14 MB | 24,920 | 3.33 GB | `8e5f09c7d420d1a98707d3300ea2f14be085b51a29f290951bdd512a83a53bbb` |
| `duet-raw-development-4-part03.tar.zst` | development stage 4 (superseded) | 104.59 MB | 1 | 250.00 MB | `b12494eb5f1ffc17e40558e5a29e1f6bbe7d861267dc69cc0b7a69212780f456` |
| `duet-raw-development-5-part01.tar.zst` | development stage 5 (superseded) | 609.81 MB | 40,000 | 2.17 GB | `dece2f3769202d1f9b2233d6c6b254f6689586f13fa9e524bd989c044a286b6a` |
| `duet-raw-development-5-part02.tar.zst` | development stage 5 (superseded) | 432.37 MB | 24,937 | 3.49 GB | `7d2e1d84baf46a90701f3a2f6b6709e47af335299924573402eea5ccc007e809` |
| `duet-raw-development-5-part03.tar.zst` | development stage 5 (superseded) | 546.73 MB | 3,242 | 3.49 GB | `42f1e48851882cb726cf1c7380bc8a661254547bb7826f8aed435bbf4ed4a01c` |
| `duet-raw-development-5-part04.tar.zst` | development stage 5 (superseded) | 624.42 MB | 2,626 | 3.50 GB | `f248fa4d54421e582b6c1aecfb2984d0aafa03a1b669a74a2b252be946885610` |
| `duet-raw-development-5-part05.tar.zst` | development stage 5 (superseded) | 357.95 MB | 18,868 | 1.39 GB | `c54b008e070826f9c7d341b08d098a27dea2f2ad238517bb5307e1738f6a4d65` |
| `duet-raw-development-5-part06.tar.zst` | development stage 5 (superseded) | 12.18 MB | 4 | 13.20 MB | `f68f4268d7e7b0203d5fb2fd55bc94f7327d543a8a4e6b205cbb63fe83de0dc3` |
| `duet-raw-development-6.tar.zst` | development stage 6 (superseded) | 681.43 MB | 34,346 | 2.09 GB | `8a5f09ac26f098f9585e2e3d457db3a3b4badc398e3300466f75994d4f14d43f` |
| `duet-raw-development-7-part01.tar.zst` | development stage 7 (superseded) | 1.28 GB | 40,000 | 3.17 GB | `8646fa30d9d594665966464024bd7eb57fabc168fb2af78091588de2b97a1c2f` |
| `duet-raw-development-7-part02.tar.zst` | development stage 7 (superseded) | 171.90 MB | 40,000 | 677.62 MB | `9d6057b0631ef58792df5984bd65ac461d2d1043b078b41b0199afb3ccbc9af5` |
| `duet-raw-development-7-part03.tar.zst` | development stage 7 (superseded) | 724.35 MB | 29,722 | 2.64 GB | `f6bef6ad37b56be68669b713fd4339586a713fc75c8ff62e035231b35e59a1c9` |
| `duet-raw-reproduction.tar.zst` | reproduction of the Lukin group's training | 580.89 MB | 1,207 | 644.04 MB | `af8e096211d96d4fa98ca8fde41330f08e0007ec801ac2d4c804d202af8c149d` |

Each archive unpacks into one top folder named after its group, so parts of a group unpack into the same tree.

## What each group holds

### fresh-test

The fresh test is the second, independent evaluation on new test circuits and noise samples, scored at
the operating points chosen on development data. It backs Tables I, II, III, XVII, XVIII and XIX and Figure 3 (the
summaries are in `paper/data/fresh_test/`). `benchmark1/`: the network's per-shot logits (`network/`), the 21 long-beam
Tesseract trials of every routed shot with each trial's time (`tesseract/`), Tesseract alone on subsets
(`tesseract-alone/`) and its budget-matched presets (`tesseract-presets/`), the subset lists (`subsets/`), the per-shot
predictions of LoMatching, Cain et al. and BP+LoMatching (`lomatching/`, `cain/`, `bp-lomatching/`) and the score outputs
(`scores/`: per-cell grids, curves, reports). `benchmark2/`: the network, the Tesseract trials, the exact
most-likely-error decoder (`exact-mle/`, SCIP, solution status and time per shot), LoMatching, the shots that have an
exact answer (`mle-prefix/`) and the score outputs. `jobs/` and `job-logs/` hold the commands, node descriptions and
logs of the jobs.

### first-test

The first test ran on held-out test circuits at operating points frozen before it was scored (Appendix D,
Tables VII to XVI and XXI; summaries in `paper/data/first_test/`). Same layout as the fresh test, plus the manifests of
its sets (`manifests/`), the scoring and calibration test runs (`scoring-tests/`) and `benchmark1/sets/sha256.txt`, the
SHA-256 of the Benchmark I test sets as they were sampled.

### operating-points

The development data the operating points were chosen on (Tables XVII and XXI,
`paper/data/fresh_test/development_best.json`, `paper/data/first_test/choices.md`). `benchmark1/`: the network's logits
on the development shots of Benchmark I (the final networks and their starts), the Tesseract trials, LoMatching and
Cain et al., the selected shots and the routing records. `benchmark2/`: the same for the development bank of
Benchmark II, with the operating-point choice records (`choice/`: thresholds, curves, simulated streams).

### timing

The speed records behind Table IV, Table XX, Figure 4 and Appendix E: the live paced stream on an RTX PRO 6000 with
32 EPYC 9555 cores and on a DGX Spark (per-shot records, timing, logs, and the comparison with the simulated stream),
the calibration of Tesseract trial times per CPU model (`calibration*/`), the network time per layer at batch 256 on
both GPUs (`first-test/network-time-*`), and the earlier end-to-end runs of the parallel long beam
(`development-runs/`).

### training

The training records of the twelve networks trained for the paper, the four networks of the main results
and the eight variants of Table V: arguments (`run.json`), monitor rows (`log.jsonl`), training logs and commands. The
networks themselves are `networks/*.pt` in this repository.

### ablations

The variants' reads at d = 5 behind Table V (per-shot outputs of each variant and of the full network on
the development data, with their manifests).

### open-benchmark

The open benchmark of transversal-gate circuits (seven circuit families, 1,328 points) that
Benchmark II's circuits come from: the per-shot predictions of LoMatching, Cain et al., BP+LoMatching, the exact
decoder and the networks on every point (`predictions/`), the jobs, and the README, manifest and checksum list of its
export (`export/`). The export's shots, circuits and re-encoded predictions are left out (see below).

### development-1 to development-7

These are the development stages before the paper's design, in time order. They are
superseded; no number in the paper comes from them, except that development-5 holds the training of the start networks
(`networks/start_d*.pt`) and their ancestors.

| group | stage |
|---|---|
| development-1 | first networks for two blocks joined by a transversal CNOT (exploratory runs; the pilot's eight networks at d = 5, 7, 9) and the pilot test (per-shot outputs on calibration and evaluation shots) |
| development-2 | the Lukin group's benchmark circuits: the design loop at d = 5, the routed composition at d = 5, the production networks at d = 5 and 7 and their test reads (network passes, long-beam Tesseract reference, scoring) |
| development-3 | the first main study: ablations at d = 7 and 9, stronger opponents (BP+LoMatching, a Tesseract curve) and the carry-over to other circuit families (networks and per-shot outputs) |
| development-4 | networks with a per-layer read-out on deeper CNOT circuits: the evaluation sets' choices and per-shot predictions at d = 7, 9 and 11, the read jobs, the large-trunk networks for d = 3 to 11, read-out and depth studies, Cain et al. and BP+LoMatching on those sets, the exact decoder at d = 7 and 9 |
| development-5 | Clifford circuits (H, S, CNOT) on two blocks: networks, test reads of the network, LoMatching, Cain et al., BP+LoMatching and the exact decoder, robustness reads; then two to four blocks (zero-shot reads and the fine-tunes that gave the start networks) |
| development-6 | single-layer circuits under uniform noise (diagnosis and repaired networks) and a second test of those networks on new sets (opponents, exact decoder, network reads) |
| development-7 | the Lukin-noise development bank (LoMatching, Tesseract, exact decoder and network reads), the fine-tune arms, test-set reads, a fresh d = 7 set and end-to-end timing packs |

### reproduction

Our reproduction of the Lukin group's network training with their published recipe: run records,
configurations, logs, the accuracy jobs under their noise model, and the two checkpoints kept from the longest run.

## Names inside the archives

The top folders and the folders below them have the plain names used above. Deeper folders and files keep the names
the records use, with one exception: folders and files named after a point of a Benchmark I set (the first test's, the
fresh test's and the development set) carry the point names of the released code, `b1t_`, `fresht_` and `b1d_` followed
by `K`, `d`, `L` and `p`. Inside the record files the same points appear under the names the runs used, which differ
only in the prefix before `_K`. The records themselves are unchanged, byte for byte, so they also contain the absolute
paths, host names and user names of the machines the jobs ran on.

## Left out

| what | files | why |
|---|---|---|
| stored shots (detection events) | 89,484 (184.78 GB) | sampled from recorded seeds (see below); `raw/detection_events.tsv` lists them with their checksums |
| exact duplicates | 6,788 | the same bytes are shipped once; `raw/duplicates.tsv` maps each left-out path to the shipped copy, or to `networks/` for the 16 networks already in this repository |
| symbolic links | 255 | links to other files of the records (one to the folder of a shot set); `raw/duplicates.tsv` lists each with its target |
| the open benchmark's export of circuits and re-encoded predictions | 389,086 (2.90 GB) | the circuits are regenerated by `duet.bench_sets`; the predictions are the per-shot outputs in `open-benchmark/predictions/` written in another format; the export's own checksum list (`open-benchmark/export/SHA256SUMS`) covers every file |
| training resume states (`ckpt.pt`, `last.pt`) | 107 (1.54 GB) | optimizer state for resuming a run; each run's network weights (EMA) are shipped |
| code snapshots, environments, caches, decoder builds | 94,669 | copies of the code and its dependencies at the time; the code is this repository |
| scheduler logs without a result | 179 | logs holding only the cluster's login banner and the job wrapper's start and exit lines |
| cluster archive and cleanup jobs, notes and checksum lists of our backup copies, third-party data | 200 | file housekeeping; the published data of other groups |

Duplicates were found by SHA-256 before packing. Files that had no checksum at that point were hashed while being
packed; 958 of them (0.61 GB) are byte-identical to another shipped file (identical outputs of two decoders or two runs,
identical job scripts and test outputs) and are shipped as well. The backup copies of three training logs of
`development-4` differ from the cluster copies; the cluster copies are shipped.

## Detection events

`raw/detection_events.tsv` has one row per stored shot file that is not in the archives (89,484 files,
184.78 GB): its name, size in bytes and SHA-256.

The shot sets of the paper's tests and of the development data are listed as the released code writes them from its
own manifests (folders `first-test/`, `fresh-test/` and `operating-points/` in the list). To regenerate one set and check
it, here the fresh test's Benchmark I set (96 files, 9.8 GB):

    python scripts/fresh_sets.py manifest --purpose test --out fresh_b1.json
    python scripts/eval_pipeline.py gen --manifest fresh_b1.json --dir fresh-test/benchmark1/sets --workers 16
    grep '^fresh-test/benchmark1/sets/.*npz' data/raw/detection_events.tsv | awk -F'\t' '{print $3 "  " $1}' | sha256sum -c --quiet

| rows of the list | regenerate with |
|---|---|
| `first-test/benchmark1/sets/` | `scripts/eval_pipeline.py manifest --purpose test`, then `gen` |
| `operating-points/benchmark1/sets/` | `scripts/eval_pipeline.py manifest --purpose dev`, then `gen` |
| `fresh-test/benchmark1/sets/` | `scripts/fresh_sets.py manifest --purpose test`, then `scripts/eval_pipeline.py gen` |
| `first-test/benchmark2/sets/` | `scripts/eval_pipeline.py gen --manifest data/first_test/MANIFEST_bench2.json` |
| `fresh-test/benchmark2/sets/` | `scripts/eval_pipeline.py gen --manifest data/fresh_test/MANIFEST_bench2.json` |
| `operating-points/benchmark2/sets/` | `scripts/dev_bank.py gen --bank operating-points/benchmark2` (writes `sets/`) |

We regenerated all six sets (378 files, 27.1 GB) with the released code. In every file the detection events, the
observables and their sizes equal those of the stored file that the paper's runs read. The files differ from the stored
ones only in the point entry each file embeds: its name and a family or purpose label, which this release renamed. Given
the manifests the runs recorded, the released code writes the stored files byte for byte. We checked that on all 96 files
of the first test's Benchmark I set, against the checksums taken when it was sampled
(`first-test/benchmark1/sets/sha256.txt` in the archives), and on one file of each of the other five sets.

The other rows are shots as they were stored. `open-benchmark/export/points/<point>/c<i>.dets.b8` and `.obs.b8` are the
open benchmark's shots in stim's `b8` format, one file per circuit, sampled from the seeds in the export's
`MANIFEST.json` (in the `open-benchmark` archive) by the sampler that `duet.bench_sets` keeps; the export step that
wrote the `b8` files is not part of this repository, and the export's `SHA256SUMS` lists every file it wrote. The
`development-5` and `development-7` rows are test shots of those stages, sampled from the seeds in their records by the
code of the time.
