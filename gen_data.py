
"""
Генератор тестовых данных для лабораторной работы по NumPy.
по variant.json.
"""
import numpy as np
import csv
import json
import argparse
import re
import os
import sys
from pathlib import Path

def parse_variant_fields(field_strings: list[str]) -> list[dict]:
    fields = []
    pattern = re.compile(r"^(\w+)\s*\(([\w\d]+)\)")
    for fs in field_strings:
        match = pattern.match(fs.strip())
        if match:
            fields.append({"name": match.group(1), "dtype": match.group(2)})
        else:
            parts = fs.split(":")[0].strip().split()
            fields.append({"name": parts[0], "dtype": parts[1].strip("()")})
    return fields

def generate_field_data(name: str, dtype: str, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """столбец данных на основе имени и типа, имитируя предметную область."""
    n_lower = name.lower()
    dtype_np = np.dtype(dtype)
    # Временные метки
    if any(k in n_lower for k in ["ts", "time", "timestamp"]):
        return np.sort(rng.integers(1_600_000_000, 1_750_000_000, n_rows)).astype(dtype_np)
    #Идентификаторы и коды
    if any(k in n_lower for k in ["id", "code", "type", "flag", "status", "zone", "err", "alarm", "fault", "sleep", "wcode", "pay_type", "art", "fall"]):
        limit = 500 if any(x in n_lower for x in ["user", "patient", "student", "acc", "hotel", "airline"]) else 100
        return rng.integers(0, limit, n_rows).astype(dtype_np)
    # Проценты, доли, рейтинги (0-100 или 0-1)
    if any(k in n_lower for k in ["hum", "occ", "load", "batt", "fill", "cap", "eff", "mois", "succ", "ctr", "comp", "leak", "churn", "err_r", "viol_r", "cong", "drops", "fid", "qerr"]):
        if "ph" in n_lower:
            return rng.uniform(6.5, 8.5, n_rows).astype(np.float32)
        if "rating" in n_lower or n_lower.startswith("rat"):
            return rng.uniform(1.0, 5.0, n_rows).astype(np.float32)
        # Доли 0..1
        if any(k in n_lower for k in ["succ", "ctr", "comp", "leak", "churn", "err_r", "viol_r", "fid", "qerr"]):
            return rng.uniform(0.05, 0.95, n_rows).astype(np.float32)
        # Проценты 0..100
        return rng.uniform(5.0, 95.0, n_rows).astype(np.float32)
    #  Цены, тарифы, объёмы (положительные, с правым хвостом)
    if any(k in n_lower for k in ["price", "fare", "rate", "amt", "rev", "mrr", "royal", "avg_bill", "cpc", "gain", "power", "bit", "latency", "ping", "alt", "km", "dist", "area", "weight", "mass", "flow", "volt", "cur", "tvol", "bpm", "glu", "hgb", "wbc", "plt", "eos", "hist", "ige", "symp", "iop", "thick", "acu", "dp", "ery", "mel", "hours", "proc_t", "door_t", "vib", "wait", "cpu", "pos_err", "comf", "uv", "ozone", "pm25", "aqi", "temp", "press", "strain", "wind", "sp", "vel", "drag", "reverb", "db", "laeq", "la10", "la90", "freq", "wave", "intensity", "coh", "noise_db", "bright", "depth", "fps", "layer", "tnoz", "gps", "lat", "long"]):
        # Физические величины: нормальное распределение с отсечкой отрицательных
        if any(k in n_lower for k in ["temp", "press", "volt", "alt", "strain", "wind", "sp", "vel"]):
            return rng.normal(20.0, 10.0, n_rows).astype(np.float32)
        # Финансы/ресурсы: экспоненциальное/логнормальное
        return np.abs(rng.normal(50.0, 25.0, n_rows)).astype(np.float32)
    #  Счётчики, количества (целые положительные)
    if any(k in n_lower for k in ["qty", "items", "steps", "likes", "orders", "cycl", "veh", "tix", "shows", "nights", "players", "turns", "gates", "imgs", "unlocks", "dl", "plays", "covers", "pick", "entry", "evt", "awak", "attempts", "crash", "fall"]):
        return rng.integers(1, 150, n_rows).astype(dtype_np)

    if "float" in dtype:
        return rng.normal(0, 10, n_rows).astype(dtype_np)
    return rng.integers(0, 50, n_rows).astype(dtype_np)

def inject_anomalies(data: dict, fields: list[dict], n_rows: int, rng: np.random.Generator):
    """ ~3% пропусков, ~3% отрицательных значений и ~3% выбросов в float-поля"""
    for f in fields:
        if "float" not in f["dtype"]:
            continue            
        arr = data[f["name"]]
        dtype_np = np.dtype(f["dtype"])
        # 1. Пропуски (NaN) ~3%
        nan_mask = rng.random(n_rows) < 0.03
        arr[nan_mask] = np.nan
        # 2. Отрицательные значения ~3% (только где они физически невозможны)
        neg_mask = rng.random(n_rows) < 0.03
        valid_neg = ~np.isnan(arr) & neg_mask
        if valid_neg.any():
            arr[valid_neg] = -np.abs(arr[valid_neg])
        # 3. Выбросы за пределы ~3%
        out_mask = rng.random(n_rows) < 0.03
        valid_out = ~np.isnan(arr) & out_mask
        if valid_out.any():
            lower, upper = np.nanpercentile(arr, [5, 95])
            span = upper - lower if upper > lower else 10.0
            arr[valid_out] = upper + span * rng.uniform(1.5, 4.0, valid_out.sum())

def main():
    parser = argparse.ArgumentParser(description="Генератор CSV данных для лабораторной NumPy")
    parser.add_argument("-variant", type=int, required=True, help="Номер варианта (1-60)")
    parser.add_argument("-rows", type=int, default=2_000_000, help="Количество строк по умолчанию 2000k")
    parser.add_argument("-seed", type=int, default=77, help="Сид генератора случайных чисел")
    parser.add_argument("-output", type=str, default=None, help="Имя выходного файла CSV")
    parser.add_argument("-config", type=str, default="variant.json", help="Путь к variant.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f" Файл конфигурации не найден: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not (1 <= args.variant <= len(config["variants"])):
        print(f" Вариант должен быть от 1 до {len(config['variants'])}")
        sys.exit(1)

    variant = config["variants"][args.variant - 1]
    output_file = args.output or f"data_variant_{args.variant}.csv"
    rng = np.random.default_rng(args.seed)
    fields = parse_variant_fields(variant["fields"])
    n_rows = args.rows

    print(f" Генерация данных для Варианта {variant['id']}: {variant['title']}")
    print(f" Строк: {n_rows:,} | Seed: {args.seed} | Полей: {len(fields)}")  
    data = {}
    for f in fields:
        data[f["name"]] = generate_field_data(f["name"], f["dtype"], n_rows, rng)
    inject_anomalies(data, fields, n_rows, rng)  
    print(" Запись в CSV...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f["name"] for f in fields])      
        chunk_size = 50_000
        for i in range(0, n_rows, chunk_size):
            end = min(i + chunk_size, n_rows)
      
            rows = zip(*(data[f["name"]][i:end] for f in fields))
            writer.writerows(rows)

    size_mb = os.path.getsize(output_file) / 1024**2
    print(f" Готово: {output_file} ({size_mb:.1f} MB)")
  
main()