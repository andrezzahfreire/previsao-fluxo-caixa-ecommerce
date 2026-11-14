"""Gera CSVs sintéticos com o MESMO esquema do Olist e um cache das séries SGS.

Serve só para testar que o pipeline executa e que a versão modular reproduz o
notebook original. Os números NÃO têm significado substantivo.

Uso (da raiz do projeto): python ferramentas/gerar_dados_sinteticos.py dados_brutos/olist dados_brutos/macro_cache
NUNCA rode isso na pasta com os dados reais: os arquivos seriam sobrescritos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
data_dir = Path(sys.argv[1]); data_dir.mkdir(parents=True, exist_ok=True)
macro_dir = Path(sys.argv[2]); macro_dir.mkdir(parents=True, exist_ok=True)

N_SELLERS = 260
sellers = [f"s{idx:05d}{rng.integers(1e9):010d}" for idx in range(N_SELLERS)]
# entrada dos vendedores: maioria em 2017, alguns em 2018 (fora da coorte)
entry = pd.to_datetime("2017-01-02") + pd.to_timedelta(rng.integers(0, 300, N_SELLERS), unit="D")
entry = entry.where(rng.random(N_SELLERS) > 0.15, pd.to_datetime("2018-01-15") + pd.to_timedelta(rng.integers(0, 150, N_SELLERS), unit="D"))
churn = entry + pd.to_timedelta(rng.integers(120, 700, N_SELLERS), unit="D")
rate = rng.lognormal(mean=-0.6, sigma=1.0, size=N_SELLERS)          # pedidos por semana
ticket = rng.lognormal(mean=4.6, sigma=0.6, size=N_SELLERS)

orders, items, payments = [], [], []
oid = 0


def add_order(ts, seller_list, approved=True):
    global oid
    oid += 1
    o = f"o{oid:08d}"
    status = rng.choice(["delivered", "shipped", "canceled", "invoiced"], p=[0.95, 0.02, 0.02, 0.01])
    appr = ts + pd.Timedelta(minutes=int(rng.integers(5, 600))) if approved else pd.NaT
    orders.append({"order_id": o, "customer_id": f"c{oid:08d}", "order_status": status,
                   "order_purchase_timestamp": ts, "order_approved_at": appr,
                   "order_delivered_carrier_date": pd.NaT, "order_delivered_customer_date": pd.NaT,
                   "order_estimated_delivery_date": ts + pd.Timedelta(days=15)})
    total = 0.0
    for k, s in enumerate(seller_list, start=1):
        price = round(float(ticket[s] * rng.lognormal(0, 0.3)), 2)
        freight = round(float(rng.uniform(5, 30)), 2)
        items.append({"order_id": o, "order_item_id": k, "product_id": f"p{rng.integers(1e6)}",
                      "seller_id": sellers[s], "shipping_limit_date": ts + pd.Timedelta(days=3),
                      "price": price, "freight_value": freight})
        total += price + freight
    n_pay = 1 if rng.random() < 0.9 else 2
    parts = np.full(n_pay, total / n_pay)
    for j, v in enumerate(parts, start=1):
        payments.append({"order_id": o, "payment_sequential": j, "payment_type": "credit_card",
                         "payment_installments": 1, "payment_value": round(float(v), 2)})
    return o


# lote isolado em set/out 2016 e lacuna até jan/2017 (como no Olist)
for d in pd.date_range("2016-09-15", "2016-10-10", freq="D"):
    for _ in range(rng.poisson(3)):
        add_order(d + pd.Timedelta(hours=int(rng.integers(0, 23))), [int(rng.integers(N_SELLERS))])

weeks = pd.date_range("2017-01-02", "2018-08-27", freq="7D")
for w in weeks:
    growth = 1.0 + 0.6 * (w - weeks[0]).days / 600
    bf = 2.5 if (w.month == 11 and 20 <= w.day <= 27) else 1.0
    for s in range(N_SELLERS):
        if not (entry[s] <= w <= churn[s]):
            continue
        n = rng.poisson(rate[s] * growth * bf)
        for _ in range(n):
            ts = w + pd.Timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 23)))
            if ts > pd.Timestamp("2018-08-29 15:00"):
                continue
            sl = [s] if rng.random() > 0.03 else [s, int(rng.integers(N_SELLERS))]
            add_order(ts, sl, approved=rng.random() > 0.002)
# resíduo de extração: 1 pedido depois de um intervalo
add_order(pd.Timestamp("2018-09-03 17:40"), [0])
# pedido com pagamento e sem itens
oid += 1
orders.append({"order_id": f"o{oid:08d}", "customer_id": "cx", "order_status": "unavailable",
               "order_purchase_timestamp": pd.Timestamp("2017-05-05"), "order_approved_at": pd.Timestamp("2017-05-05 10:00"),
               "order_delivered_carrier_date": pd.NaT, "order_delivered_customer_date": pd.NaT,
               "order_estimated_delivery_date": pd.Timestamp("2017-05-20")})
payments.append({"order_id": f"o{oid:08d}", "payment_sequential": 1, "payment_type": "boleto",
                 "payment_installments": 1, "payment_value": 99.9})

pd.DataFrame(orders).to_csv(data_dir / "olist_orders_dataset.csv", index=False)
pd.DataFrame(payments).to_csv(data_dir / "olist_order_payments_dataset.csv", index=False)
pd.DataFrame(items).to_csv(data_dir / "olist_order_items_dataset.csv", index=False)
print(f"pedidos {len(orders):,} | itens {len(items):,} | pagamentos {len(payments):,}")

# cache SGS no formato gravado por fetch_sgs_monthly
months = pd.date_range("2016-01-01", "2018-12-01", freq="MS")
series = {"macro_ipca": 433, "macro_ibcbr": 24363, "macro_icc": 4393, "macro_selic": 4390}
base = {"macro_ipca": 0.4, "macro_ibcbr": 135.0, "macro_icc": 85.0, "macro_selic": 0.8}
for var, code in series.items():
    v = base[var] + np.cumsum(rng.normal(0, base[var] * 0.02, len(months)))
    pd.DataFrame({"date": months, "month": months, "variavel": var, "codigo_sgs": code, "value": np.round(v, 5)}) \
        .to_csv(macro_dir / f"sgs_{code}_{var}.csv", index=False)
print("cache macro gravado em", macro_dir)
