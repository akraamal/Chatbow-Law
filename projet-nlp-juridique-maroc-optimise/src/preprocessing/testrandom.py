import re

text = open(
    "data/processed/ar/BO_7517_Ar.txt",
    encoding="utf-8"
).read()

for m in re.finditer("مادة", text):
    print(text[m.start()-20:m.start()+50])
    break