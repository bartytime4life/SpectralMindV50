import zipfile, pathlib, json, time
def pack(csv_path: str, out_zip: str):
    z = zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED)
    z.write(csv_path, arcname="submission.csv")
    meta = {"created": time.time(), "tool": "spectramind-v50"}
    z.writestr("meta.json", json.dumps(meta))
    z.close()
