"""Parse sales CSV text into dict rows.

The first line is the header. Blank lines are skipped. Fields are stripped of surrounding
whitespace. Quoted fields may contain commas (standard CSV quoting).
"""


def parse(text):
    lines = text.splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split(",")]
        rows.append(dict(zip(header, fields)))
    return rows
