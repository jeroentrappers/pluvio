# Lightning multimodal vs radar-only nowcast — 2026-06-17

**Question:** does adding the MTG-LI lightning channel (`li_flash`) improve
heavy-rain nowcast skill vs a radar-only model, on the *same* summer-2025 window?

## Setup (identical except the lightning channel)
- Grid 256² ≈ 3.0×2.7 km; SeamlessNet base-32, `--history-steps 6`, 10 epochs (`--max-minutes 90`).
- **mm**: 7 channels (radar + history + static + `li_flash`), 665,475 params, best val 0.0266 @ epoch 3.
- **ro**: 6 channels (no lightning), 665,187 params, best val **0.0261** @ epoch 3.
- Both overfit by epoch 3 (val climbs after); train=58804 / val=14952 windows; eval 4000 samples vs **real pysteps** optical-flow.
- Instance: Runcrate RTX-4090, ~2.37 h, **~$1.56**.

## mm − ro delta (model rows; bar = optical-flow)
| lead | mm CSI1 | ro CSI1 | ΔCSI1 | mm CSI.1 | ro CSI.1 | ΔCSI.1 | ΔMAE | OF CSI1 |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 10 | 0.474 | 0.490 | **−0.016** | 0.537 | 0.492 | +0.045 | −0.004 | 0.518 |
| 20 | 0.499 | 0.501 | −0.002 | 0.481 | 0.439 | +0.042 | −0.006 | 0.525 |
| 30 | 0.423 | 0.425 | −0.002 | 0.422 | 0.395 | +0.027 | −0.009 | 0.404 |
| 40 | 0.340 | 0.346 | −0.006 | 0.357 | 0.337 | +0.020 | −0.012 | 0.318 |
| 50 | 0.330 | 0.342 | −0.012 | 0.336 | 0.311 | +0.025 | −0.014 | 0.316 |
| 60 | 0.281 | 0.292 | −0.011 | 0.305 | 0.273 | +0.032 | −0.018 | 0.266 |
| 70 | 0.232 | 0.245 | −0.013 | 0.278 | 0.242 | +0.036 | −0.022 | 0.227 |
| 80 | 0.237 | 0.257 | −0.020 | 0.269 | 0.219 | +0.050 | −0.027 | 0.239 |
| 90 | 0.169 | 0.198 | −0.029 | 0.249 | 0.165 | +0.084 | −0.028 | 0.194 |
| 100 | 0.127 | 0.162 | −0.035 | 0.217 | 0.114 | +0.103 | −0.034 | 0.175 |
| 110 | 0.133 | 0.169 | −0.036 | 0.213 | 0.114 | +0.099 | −0.034 | 0.183 |
| 120 | 0.107 | 0.148 | −0.041 | 0.201 | 0.095 | +0.106 | −0.038 | 0.164 |

Means (10–120 min): **ΔCSI@1 = −0.019**, ΔCSI@0.1 = +0.056, ΔMAE = −0.021 (mm lower).

## Verdict (honest)
Adding lightning **did NOT improve heavy-rain skill** — ΔCSI@1 is negative at
*every* lead and worsens with lead (−0.002 → −0.041). It **improved light-rain
footprint** (ΔCSI@0.1 +0.02→+0.11) and MAE (smoother, broader field).
Interpretation: the lightning channel pushes the model to paint *broader, smoother*
convective footprints — better any-rain detection, but it **smears the peaks**,
which is exactly what CSI@1 penalises.

Against optical-flow on heavy-rain CSI@1, **radar-only wins over a wider band
(30–90 min, 7 leads) than mm (30–70 min, 5 leads)** — so lightning *narrowed* the
nowcast win. On CSI@0.1 both models still lose to optical-flow at every lead
(no defensible "beats nowcasting" claim).

**Caveats:** lightning-only (no GII / cloud-phase / IR yet); summer-2025 only
(convective regime favours pure advection); undertrained (~10 epochs, overfit by
epoch 3); 256² grid. A fairer test needs the richer aux channels, more epochs with
regularisation/early-stop, and a multi-season window — future work on asusprime (free).
