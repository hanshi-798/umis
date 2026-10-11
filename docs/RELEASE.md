# Publishing v1.3 to GitHub

The repository root is the directory containing `pyproject.toml` and this README.
Upload that directory's contents, not its containing ZIP. Publication was not
performed as part of preparing this artifact.

## Local verification

```bash
python -m pip install -e ".[dev,bench,judge]"
python -m pytest -q
python -m build
umis --version
```

## GitHub repository and tag

Create an empty GitHub repository, then replace `YOUR_ACCOUNT` below with your
actual account. If the repository already has history, integrate these files on
a new branch rather than replacing that history.

```bash
git init
git add .
git commit -m "Prepare UMIS v1.3.0 release"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/umis.git
git push -u origin main
git tag -a v1.3.0 -m "UMIS v1.3.0"
git push origin v1.3.0
```

After GitHub CI passes, create a release using tag `v1.3.0` and title
`UMIS v1.3.0`. Paste the text from `docs/RELEASE_NOTES_v1.3.md`. GitHub supplies
the source archives; optionally attach locally built wheels and source distributions.
No PyPI credentials, publishing automation, fake repository URL, or author identity
has been added to this package.
