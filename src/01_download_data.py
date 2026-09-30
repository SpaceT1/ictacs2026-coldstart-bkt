"""Step 1: download the public ASSISTments 2009-2010 skill-builder data (~83 MB)."""
import os, urllib.request
URL = "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv"
os.makedirs("data", exist_ok=True)
dst = "data/as.csv"
if not os.path.exists(dst):
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, dst)
print("OK:", dst, os.path.getsize(dst) // 1_000_000, "MB")
