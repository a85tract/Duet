# First test: the operating points

## Choices

| result | d | basis | f | c (s x L) | failures / shots, development (network alone) | geo. mean failure rate (network alone) | routed shots | us per layer, paced stream (batch p99) | bar | qualifying | rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | spark (N = 20) | 10 % | none | 30057 / 10958400 (network alone 31315) | 0.004492 (0.004683) | 1095928 | 5.9 (p99 181 ms) | 47.4 | 56 / 56 | most accurate below the bar |
| 2 | 3 | spark (N = 20) | 10 % | none | 6300 / 688000 (network alone 6393) | 0.01463 (0.01485) | 68801 | 7.7 (p99 220 ms) | 47.4 | 56 / 56 | most accurate below the bar |
| 1 | 3 | grace (N = 64) | 10 % | 0.001 | 30057 / 10958400 (network alone 31315) | 0.004492 (0.004683) | 1095928 | 1.6 (p99 38 ms) | 47.4 | 56 / 56 | most accurate below the bar |
| 2 | 3 | grace (N = 64) | 10 % | none | 6300 / 688000 (network alone 6393) | 0.01463 (0.01485) | 68801 | 2.6 (p99 160 ms) | 47.4 | 56 / 56 | most accurate below the bar |
| 1 | 5 | spark (N = 20) | 10 % | 0.05 | 2874 / 10958400 (network alone 4218) | 0.0003873 (0.0005417) | 1095846 | 54.3 (p99 2343 ms) | 103.3 | 56 / 56 | most accurate below the bar |
| 2 | 5 | spark (N = 20) | 10 % | 0.02 | 4485 / 5712000 (network alone 6213) | 0.001617 (0.002258) | 571200 | 93.9 (p99 2099 ms) | 103.3 | 47 / 56 | most accurate below the bar |
| 1 | 5 | grace (N = 64) | 10 % | 0.05 | 2874 / 10958400 (network alone 4218) | 0.0003873 (0.0005417) | 1095846 | 15.4 (p99 1125 ms) | 103.3 | 56 / 56 | most accurate below the bar |
| 2 | 5 | grace (N = 64) | 10 % | 0.02 | 4484 / 5712000 (network alone 6213) | 0.001617 (0.002258) | 571200 | 26.6 (p99 859 ms) | 103.3 | 56 / 56 | most accurate below the bar |
| 1 | 7 | spark (N = 20) | 5 % | none | 262 / 10958400 (network alone 695) | 3.054e-05 (7.225e-05) | 547920 | 157.6 (p99 8612 ms) | 185.2 | 49 / 56 | most accurate below the bar |
| 2 | 7 | spark (N = 20) | 2 % | none | 372 / 3720000 (network alone 876) | 0.0001551 (0.0003606) | 74400 | 176.5 (p99 7147 ms) | 185.2 | 41 / 56 | most accurate below the bar |
| 1 | 7 | grace (N = 64) | 5 % | none | 262 / 10958400 (network alone 695) | 3.054e-05 (7.225e-05) | 547920 | 44.9 (p99 4535 ms) | 185.2 | 56 / 56 | most accurate below the bar |
| 2 | 7 | grace (N = 64) | 20 % | 0.1 | 357 / 3720000 (network alone 876) | 0.0001502 (0.0003606) | 744000 | 183.5 (p99 6100 ms) | 185.2 | 55 / 56 | most accurate below the bar |
| 1 | 9 | spark (N = 20) | 2 % | none | 34 / 10958400 (network alone 204) | 4.916e-06 (1.927e-05) | 219168 | 231.2 (p99 15639 ms) | 302.2 | 47 / 56 | most accurate below the bar |
| 1 | 9 | grace (N = 64) | 2 % | none | 34 / 10958400 (network alone 204) | 4.916e-06 (1.927e-05) | 219168 | 65.2 (p99 7648 ms) | 302.2 | 56 / 56 | most accurate below the bar |
