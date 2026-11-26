"""Seeded synthetic card-transaction simulator.

Real card data is private, so this generates behaviour with known fraud patterns
and deliberately confusing legitimate behaviour (travel, new phones, shopping
sprees, large purchases) so that no single rule separates the classes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from fraudshield.config import DataConfig
from fraudshield.schema import CATEGORIES

START = datetime(2026, 1, 1, tzinfo=UTC)
COUNTRIES = np.array(["IN", "US", "GB", "SG", "AE", "DE"])
CATEGORY_AMOUNT = np.array([25.0, 20.0, 35.0, 50.0, 180.0, 220.0, 15.0, 40.0])
ONLINE_PROB = np.array([0.1, 0.1, 0.0, 0.4, 0.5, 0.8, 1.0, 0.6])
MERCHANTS_PER_CATEGORY = 150


def simulate(config: DataConfig) -> pd.DataFrame:
    """Return a time-sorted frame of transactions with `is_fraud` and `scenario`."""
    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []

    def add(customer, seconds, amount, category, merchant, online, country, device, scenario):
        rows.append(
            {
                "customer_id": f"C{customer:05d}",
                "timestamp": START + timedelta(seconds=int(seconds)),
                "amount": round(float(max(amount, 0.5)), 2),
                "merchant_id": f"M-{CATEGORIES[category]}-{merchant:03d}",
                "category": CATEGORIES[category],
                "channel": "online" if online else "pos",
                "country": str(country),
                "device_id": device if online else "",
                "is_fraud": int(scenario != "legit"),
                "scenario": scenario,
            }
        )

    def daytime(day: int) -> float:
        hour = rng.normal(14, 4.5) % 24
        return day * 86400 + hour * 3600

    horizon = config.n_days * 86400
    profiles = []
    for c in range(config.n_customers):
        home = rng.choice(COUNTRIES, p=[0.5, 0.2, 0.1, 0.08, 0.07, 0.05])
        spend = rng.lognormal(0, 0.45)
        prefs = rng.dirichlet(np.array([5, 3, 2, 2, 0.6, 0.4, 1, 0.3]))
        favourites = rng.integers(0, MERCHANTS_PER_CATEGORY, size=(len(CATEGORIES), 4))
        rate = rng.gamma(2.5, 0.28)
        device, phone_change = f"D{c}-a", rng.integers(0, config.n_days * 4)
        profiles.append((home, spend, prefs, favourites, phone_change))

        trip_start = rng.integers(0, config.n_days * 2)
        trip_end = trip_start + rng.integers(3, 12)
        abroad = rng.choice(COUNTRIES[COUNTRIES != home])

        for day in range(config.n_days):
            if day == phone_change:
                device = f"D{c}-b"
            country = abroad if trip_start <= day < trip_end else home
            n = rng.poisson(rate)
            spree = rng.random() < 0.003
            if spree:
                n += int(rng.integers(4, 9))
            spree_start = daytime(day)
            for _ in range(n):
                cat = int(rng.choice(len(CATEGORIES), p=prefs))
                explore = rng.random() < 0.12
                merchant = (
                    rng.integers(0, MERCHANTS_PER_CATEGORY)
                    if explore
                    else rng.choice(favourites[cat])
                )
                amount = CATEGORY_AMOUNT[cat] * spend * rng.lognormal(0, 0.6)
                if rng.random() < 0.004:
                    amount *= rng.uniform(4, 12)
                when = spree_start + rng.uniform(0, 5400) if spree else daytime(day)
                online = rng.random() < ONLINE_PROB[cat]
                add(c, when, amount, cat, int(merchant), online, country, device, "legit")

    risky = [CATEGORIES.index(x) for x in ("electronics", "gift_cards", "digital_goods", "retail")]
    for day in range(config.n_days):
        for _ in range(rng.poisson(config.fraud_episodes_per_day)):
            c = int(rng.integers(config.n_customers))
            home, spend, prefs, favourites, phone_change = profiles[c]
            start = day * 86400 + rng.uniform(0, 86400)
            foreign = rng.choice(COUNTRIES[COUNTRIES != home])
            # Some attacks run through the victim's own (compromised) device.
            if rng.random() < 0.35:
                attacker = f"D{c}-b" if day >= phone_change else f"D{c}-a"
            else:
                attacker = f"X{day}-{c}"
            kind = rng.choice(["card_testing", "stolen_card", "account_takeover"])
            t = start
            if kind == "card_testing":
                cat = CATEGORIES.index("digital_goods")
                country = foreign if rng.random() < 0.5 else home
                for _ in range(int(rng.integers(4, 12))):
                    t += rng.exponential(60)
                    merchant = int(rng.integers(MERCHANTS_PER_CATEGORY))
                    add(c, t, rng.uniform(0.5, 6), cat, merchant, True, country, attacker, kind)
            elif kind == "stolen_card":
                country = foreign if rng.random() < 0.4 else home
                for _ in range(int(rng.integers(2, 6))):
                    t += rng.exponential(1200)
                    cat = int(rng.choice(risky))
                    online = rng.random() < 0.4
                    amount = CATEGORY_AMOUNT[cat] * spend * rng.uniform(1.2, 5)
                    merchant = int(rng.integers(MERCHANTS_PER_CATEGORY))
                    add(c, t, amount, cat, merchant, online, country, attacker, kind)
            else:
                # Quiet takeover: home country, sometimes a familiar merchant.
                for _ in range(int(rng.integers(1, 4))):
                    t += rng.exponential(7200)
                    cat = int(rng.choice(risky))
                    amount = CATEGORY_AMOUNT[cat] * spend * rng.uniform(0.8, 3.5)
                    familiar = rng.random() < 0.3
                    merchant = int(
                        rng.choice(favourites[cat])
                        if familiar
                        else rng.integers(MERCHANTS_PER_CATEGORY)
                    )
                    add(c, t, amount, cat, merchant, True, home, attacker, kind)

    frame = pd.DataFrame(rows)
    frame = frame[frame["timestamp"] < START + timedelta(seconds=horizon)]
    frame = frame.sort_values(["timestamp", "customer_id"], kind="stable").reset_index(drop=True)
    frame.insert(0, "transaction_id", [f"T{i:09d}" for i in range(len(frame))])
    return frame
