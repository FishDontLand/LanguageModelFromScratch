def pytest_addoption(parser):
    parser.addoption("--snapshot-exact", action="store_true", help="Compare snapshots exactly")
