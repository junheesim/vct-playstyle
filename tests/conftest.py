import pathlib, sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import paths

# The raw Kaggle dump is 1.3 GB and is not committed, so a fresh clone -- and CI --
# cannot run the end-to-end checks. Those skip rather than error, which leaves the
# checks that need no data running everywhere: that the site is in sync with the
# parts it is built from, that every decision cites a diagnostic that exists, and
# that every referenced figure is present. Those are the ones that caught the real
# bugs in this repository.
NEEDS_DATA_MODULES = {"test_pipeline", "test_pca", "test_roles"}
NEEDS_DATA_TESTS = {"test_the_site_quotes_the_numbers_the_pipeline_computes"}


def pytest_collection_modifyitems(config, items):
    if paths.RAW.is_dir():
        return
    skip = pytest.mark.skip(reason=f"raw data absent at {paths.RAW}; this check reads it")
    for item in items:
        if item.module.__name__ in NEEDS_DATA_MODULES or item.name in NEEDS_DATA_TESTS:
            item.add_marker(skip)
