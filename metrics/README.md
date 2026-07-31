# Metrics

Long-term numbers about this repository, as tidy CSVs. GitHub itself keeps only
14 days of traffic and no download history at all, so these files exist to hold
what the API forgets.

| File | What it is | Updated by |
|---|---|---|
| `download_counts.csv` | Release-asset download totals, one row per asset per snapshot | `.github/workflows/download-metrics.yml`, weekly |
| `traffic.csv` | Daily views and clones, 2022-08-29 to 2026-07-26 | nothing, frozen |
| `stars.csv` | Cumulative stars, 2022-02-13 to 2026-07-17 | nothing, frozen |
| `forks.csv` | Cumulative forks, 2022-04-13 to 2026-07-17 | nothing, frozen |

## Where the frozen three came from

A `jgehrcke/github-repo-stats` action collected them on a `github-repo-stats`
branch from 2022 until it stopped on 2026-07-27. That branch also carried a
report it regenerated three times a day, which grew to 2 GB of near-identical
copies, so the branch was deleted and only these aggregates were kept.

Two things to know when reading them:

- `traffic.csv` spans the rename. This repo was called EcoAssist until February
  2025, and the action recorded the two names separately. The series is joined
  here into one continuous file; on the 13 days both covered, the AddaxAI
  numbers are used.
- `traffic.csv` has one missing day, 2022-09-02. That gap is in the source.

Nothing appends to these three any more. Traffic older than 14 days cannot be
fetched again, so treat them as an archive.
