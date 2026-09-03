# Release Checklist

Use this checklist before tagging a toolbox or core-package release.

## Local Verification

```bash
python3 -m py_compile HRpQCTTools/*/*.py IOTools/*/*.py CTTools/*/*.py Setup/*/*.py
python3 -m pytest tests/test_package_status.py tests/test_batch_processor_module.py -q
```

Run focused tests for touched modules and the relevant core package test suites.

## Documentation

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs build --strict
```

## Versioning

- Bump core package versions when PyPI behavior changes.
- Update Slicer Setup minimum package versions.
- Tag core packages with semantic version tags.
- Tag Slicer wrapper releases when the extension changed.

## Publishing

Trusted publishing workflows should build wheels and source distributions, run Twine checks, and publish from protected PyPI environments.

After publishing, verify that the new version appears on PyPI and that GitHub Actions are green.
