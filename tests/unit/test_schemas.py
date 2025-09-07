import json, pathlib
def test_submission_schema_loads():
    schema = json.loads(pathlib.Path("schemas/submission.schema.json").read_text())
    assert schema["properties"]["mu"]["minItems"] == 283
