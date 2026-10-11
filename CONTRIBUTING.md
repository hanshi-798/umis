# Contributing

Use a branch, describe the concrete behavior you change, and run
`python -m pytest -q` before submitting a pull request. For probe or scoring
changes, include a small synthetic regression fixture. Do not commit downloaded
datasets, model weights, private reports, or credentials.

Model/GPU experiments should state the exact model revisions, environment,
split policy, feature configuration, and failures as well as metric values.
