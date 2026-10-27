def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test that needs more than 5 seconds")
