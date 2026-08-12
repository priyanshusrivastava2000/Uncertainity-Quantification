# Baseline feature build - audit report

Reference t=0: DXSUM baseline visit, 3685 patients (3685 baseline-coded).

Gap filter applied: none

## Record selection and timing

| modality | patients | baseline_coded | fallback_used | median_gap_months | min_gap | max_gap | within_6mo | within_12mo | after_baseline_gt_6mo | after_baseline_gt_12mo | after_baseline_gt_36mo | multi_record_patients |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mmse | 4668 | 4261 | 407 | -0.13 | -23.7 | 59.9 | 3657 | 3672 | 7 | 3 | 2 | 2350 |
| mri | 3175 | 3108 | 67 | 0.2 | -13.1 | 60.3 | 3082 | 3097 | 49 | 38 | 7 | 2223 |
| csf | 1660 | 1621 | 39 | 1.05 | -10.7 | 133.4 | 1599 | 1626 | 60 | 34 | 19 | 839 |

## Notes

- **mmse**: 4668 patients, 37 features. 407 used the earliest-record fallback; 3 sit more than 12 months after t=0 and are candidates for exclusion.
- **mri**: 3175 patients, 325 features. 67 used the earliest-record fallback; 38 sit more than 12 months after t=0 and are candidates for exclusion.
- **csf**: 1660 patients, 11 features. 39 used the earliest-record fallback; 34 sit more than 12 months after t=0 and are candidates for exclusion.
- **mmse coalesce window**: median 0 days, max 0 days across the baseline-coded rows being merged.
- **mri field strength**: {'3T': 2333, '1.5T': 842}. This tracks ADNI study era (1.5T = ADNI-1, 3T = ADNI-GO/2/3) and is a known confound for any outcome that differs by era - carried here so it can be controlled downstream.

## Lowest-coverage features (bottom 15)

| modality | feature | non_missing | n_patients | coverage_pct |
|---|---|---|---|---|
| mri | ST8SV | 32 | 3175 | 1.0 |
| mri | ST68SV | 245 | 3175 | 7.7 |
| csf | ELE_ABETA40 | 474 | 1660 | 28.6 |
| csf | ELE_ABETA42_40 | 474 | 1660 | 28.6 |
| mmse | MMD | 2812 | 4668 | 60.2 |
| mmse | attention | 2812 | 4668 | 60.2 |
| mmse | MMR | 2812 | 4668 | 60.2 |
| mmse | MML | 2812 | 4668 | 60.2 |
| mmse | MMW | 2812 | 4668 | 60.2 |
| mmse | MMO | 2812 | 4668 | 60.2 |
| mri | ST97SA | 2998 | 3175 | 94.4 |
| mri | ST97TA | 2998 | 3175 | 94.4 |
| mri | ST107SA | 2998 | 3175 | 94.4 |
| mri | ST107CV | 2998 | 3175 | 94.4 |
| mri | ST107TS | 2998 | 3175 | 94.4 |
