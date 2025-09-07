import json, jsonschema, pathlib

def validate_json(obj, schema_path):
    schema = json.loads(pathlib.Path(schema_path).read_text())
    jsonschema.validate(obj, schema)
