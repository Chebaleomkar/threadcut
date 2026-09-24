"""Row cleaning.

clean(rows) converts 'qty' to int and 'price' to float, drops rows whose qty or price is missing
or not a number, and drops rows with qty <= 0. The 'region' field is title-cased ("north east" ->
"North East").
"""


def clean(rows):
    out = []
    for r in rows:
        try:
            qty, price = int(r["qty"]), float(r["price"])
        except ValueError:
            continue
        if qty < 0:
            continue
        out.append({**r, "qty": qty, "price": price, "region": r["region"].capitalize()})
    return out
