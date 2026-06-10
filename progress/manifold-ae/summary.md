# Toy manifold-AE sweep — summary

297 runs. Metrics averaged over seeds.


## FVU vs clean manifold (embedded+noise), flat@k vs matched

| manifold | tier | flat@k | matched | AE/PCA flat | AE/PCA matched |
|---|---|---|---|---|---|
| helix | easy | 0.0140 | 0.0140 | 18.2303 | 18.2303 |
| swiss_roll | easy | 0.0588 | 0.0588 | 4.9709 | 4.9709 |
| s_curve | easy | 0.1237 | 0.1237 | 0.9982 | 0.9982 |
| wavy_sheet | easy | 0.0067 | 0.0067 | 6.1703 | 6.1703 |
| circle | closed | 0.2478 | 0.0017 | 3.5697 | 298.2910 |
| sphere | closed | 0.0322 | 0.0021 | 11.2047 | 159.2373 |
| torus | closed | 0.1511 | 0.0944 | 0.9989 | 6.3800 |

## Membership AUROC & denoising (embedded+noise)

| manifold | mode | AUROC | projection_ratio | participation_ratio |
|---|---|---|---|---|
| helix | flat@k | 1.000 | 0.226 | 1.00 |
| helix | matched | 1.000 | 0.226 | 1.00 |
| swiss_roll | flat@k | 1.000 | 0.464 | 1.86 |
| swiss_roll | matched | 1.000 | 0.464 | 1.86 |
| s_curve | flat@k | 1.000 | 0.671 | 1.59 |
| s_curve | matched | 1.000 | 0.671 | 1.59 |
| wavy_sheet | flat@k | 1.000 | 0.161 | 1.87 |
| wavy_sheet | matched | 1.000 | 0.161 | 1.87 |
| circle | flat@k | 0.949 | 0.616 | 1.00 |
| circle | matched | 1.000 | 0.065 | 2.00 |
| sphere | flat@k | 0.997 | 0.224 | 1.97 |
| sphere | matched | 1.000 | 0.092 | 3.00 |
| torus | flat@k | 1.000 | 0.770 | 1.92 |
| torus | matched | 1.000 | 0.537 | 2.68 |