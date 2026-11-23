# Trained networks

The weight files are in this directory. Check them with

    cd networks && sha256sum -c SHA256SUMS

Each file is the final exponential-moving-average snapshot (`final_ema.pt`) of one training run: a `torch.save` dictionary
with the keys `format`, `cfg` (the constructor arguments), `model` (the weights), `args` (the training arguments) and
`step`. Load one with

    import torch
    from duet.net import model_cl
    ck = torch.load("networks/main_d5.pt", map_location="cpu", weights_only=False)
    net = model_cl.NetCL(**ck["cfg"])
    net.load_state_dict(ck["model"])

Files written before the code was renamed carry an earlier name in `format`; `duet.net.train_cl.format_ok` accepts it.

## The networks of the paper's results

All use the Lukin group's noise model at p = 0.3 %. Width 128, bottleneck 32, 4 heads, one network per distance for 2 to 4
blocks. The distance sets the checkpoint's `cfg["d"]`.

| file | distance | what it is | where the paper uses it |
|---|---|---|---|
| `main_d3.pt` | 3 | main network, 7 residual blocks (grown from the 3-block start, 911,877 parameters) | every Benchmark I and II result at d = 3, the speed tables |
| `main_d5.pt` | 5 | main network, 7 residual blocks | every result at d = 5 |
| `main_d7.pt` | 7 | main network, 7 residual blocks | every result at d = 7 |
| `main_d9.pt` | 9 | main network, 7 residual blocks (912,843 parameters) | every Benchmark I result at d = 9 |
| `start_d3.pt` | 3 | the start that `main_d3` was trained from: 3 residual blocks, 523,397 parameters | Appendix B comparison with the start; the first step of training at d = 3 |
| `start_d5.pt`, `start_d7.pt`, `start_d9.pt` | 5, 7, 9 | the starts of the other main networks, 7 residual blocks | Appendix B comparison with the start; the first step of training |
| `variant_full_d5.pt` | 5 | ablation reference: trained from scratch on 10^8 samples, 7 blocks | the ablation table (Table V) |
| `variant_no_exchange_d5.pt` | 5 | no exchange between blocks in the trunk or the read-out (`--couple none`) | Table V |
| `variant_late_exchange_d5.pt` | 5 | blocks meet only in the read-out (`--couple late`) | Table V |
| `variant_pooled_readout_d5.pt` | 5 | one pooled read-out instead of the per-layer one (`--head pool`) | Table V |
| `variant_small_d5.pt` | 5 | 3 residual blocks (`--blocks 3`) | Table V |
| `variant_k2_d5.pt`, `variant_k3_d5.pt`, `variant_k4_d5.pt` | 5 | specialists trained on 2, 3 or 4 blocks only | Table V |

The checksums in `SHA256SUMS` are the ones the experiment records state for the files that produced the results. The four
`start_*` checksums are those of the starts that the experiments read.

## Training the networks

`scripts/train.sh` and the command lines in the main README train the main networks from the starts and the variants from
scratch. The starts themselves came from a longer chain (two-block training at learning rate 10^-3, then two to four blocks
at 3 x 10^-4; Table VI of the paper). That chain began from a two-block network of an earlier design that this release does not
contain, so the `start_*.pt` files are listed here but cannot be rebuilt from this release. `train_cl` trains the same architecture from scratch
without `--init`.
