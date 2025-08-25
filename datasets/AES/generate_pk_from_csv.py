#!/usr/bin/env python3
"""
Generate AES dataset split files (train.pk, dev.pk, test.pk) for each prompt_id
from the existing hand_crafted_v3.csv. This creates minimal records expected by
AESProcessor.get_examples: 'prompt_id', 'essay_id', 'content_text', 'score'.

Note: Since raw essay texts are unavailable, content_text is a simple placeholder
string so the pipeline can run end-to-end. Scores come from the CSV 'score' column.
"""
from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CSV_PATH = os.path.join(ROOT, 'hand_crafted_v3.csv')


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_record(row: pd.Series) -> Dict:
    return {
        'prompt_id': int(row['prompt_id']),
        'essay_id': int(row['item_id']),
        # Placeholder content since raw essays are not present in repo
        'content_text': f"Essay {int(row['item_id'])} for prompt {int(row['prompt_id'])}.",
        'score': int(row['score']),
        # Additional attributes (e.g., analytic trait scores) are optional and
        # not present in this CSV; AESProcessor will handle missing keys.
    }


def split_list(items: List, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(items)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    # Ensure non-empty splits if possible
    if n >= 3:
        n_train = max(n_train, 1)
        n_dev = max(n_dev, 1)
    n_test = n - n_train - n_dev
    return items[:n_train], items[n_train:n_train + n_dev], items[n_train + n_dev:]


def main(seed: int = 42) -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if not {'item_id', 'prompt_id', 'score'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: item_id, prompt_id, score")

    # Group by prompt_id and create per-domain splits
    rng = random.Random(seed)

    for prompt_id, gdf in df.groupby('prompt_id'):
        domain_dir = os.path.join(ROOT, str(int(prompt_id)))
        ensure_dir(domain_dir)

        # Deduplicate by essay_id in case CSV has repeats
        gdf = gdf.drop_duplicates(subset=['item_id'])
        records = [to_record(row) for _, row in gdf.iterrows()]
        rng.shuffle(records)

        train, dev, test = split_list(records, ratios=(0.8, 0.1, 0.1))

        for split, data in [('train', train), ('dev', dev), ('test', test)]:
            out_path = os.path.join(domain_dir, f'{split}.pk')
            with open(out_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Wrote {split} ({len(data)} items) -> {out_path}")


if __name__ == '__main__':
    main()
