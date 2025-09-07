from spectramind.utils.hashing import hash_config_snapshot
def test_hash_len():
    h = hash_config_snapshot({"a":1})
    assert len(h) == 12
