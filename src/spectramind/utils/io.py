def read_yaml(path):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(path).read_text())
