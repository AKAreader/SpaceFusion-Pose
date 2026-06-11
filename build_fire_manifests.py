#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build same-domain fire manifests for downstream pose-judge training and hard-case test-only evaluation.

Outputs:
  - fire_train.csv
  - fire_val.csv
  - fire_test_hard.csv

Rules:
  - All paired satellites are mixed into fire_train / fire_val / fire_test_hard.
  - Splitting is stratified by (satellite_name, subset_name).
  - Samples are paired by frame_idx first, then image_name fallback.
  - Metadata from per-folder csv/xlsx is used when available.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_FIELDS = [
    "satellite_name",
    "frame_idx",
    "visible_path",
    "infrared_path",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "difficulty_reason_visible",
    "difficulty_reason_infrared",
    "subset_name",
]


@dataclass
class FolderPair:
    satellite_name: str
    satellite_key: str
    visible_dir: str
    infrared_dir: str


@dataclass
class ImageEntry:
    path: str
    image_name: str
    frame_idx: Optional[int]


@dataclass
class MetaEntry:
    frame_idx: Optional[int]
    image_name: str
    quat: Optional[Tuple[float, float, float, float]]
    difficulty_reason: str


def _norm_text(x: object) -> str:
    s = "" if x is None else str(x).strip()
    return s


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm_text(name).lower())


def _satellite_key(name: str) -> str:
    s = _norm_text(name).lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_int(x: object) -> Optional[int]:
    if x is None:
        return None
    s = _norm_text(x)
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _stem(path_or_name: object) -> str:
    s = _norm_text(path_or_name)
    if not s:
        return ""
    b = os.path.basename(s)
    return os.path.splitext(b)[0]


def _name_aliases(name: str) -> List[str]:
    base = os.path.basename(_norm_text(name))
    stem = os.path.splitext(base)[0]
    out: List[str] = []

    for token in [base, stem]:
        t = _norm_text(token)
        if t:
            out.append(t)

    digit = _extract_int(stem)
    if digit is not None:
        out.append(str(digit))
        out.append(f"{digit:04d}")
        out.append(f"{digit:05d}")
        out.append(f"{digit:06d}")

    dedup: List[str] = []
    seen = set()
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(x)
    return dedup


def _lookup_keys(frame_idx: Optional[int], image_name: str) -> List[str]:
    keys: List[str] = []
    if frame_idx is not None:
        keys.append(f"frame:{frame_idx}")
    for alias in _name_aliases(image_name):
        keys.append(f"name:{alias.lower()}")
    # Stable dedup while keeping priority order.
    out: List[str] = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _subset_name(frame_idx: Optional[int]) -> str:
    if frame_idx is None:
        return ""
    if 1 <= frame_idx <= 50:
        return "subset_1"
    if 51 <= frame_idx <= 100:
        return "subset_2"
    if 101 <= frame_idx <= 150:
        return "subset_3"
    if 151 <= frame_idx <= 200:
        return "subset_4"
    if 201 <= frame_idx <= 250:
        return "subset_5"
    return ""


def _discover_folder_pairs(data_root: str) -> List[FolderPair]:
    visible_by_sat: Dict[str, Tuple[str, str]] = {}
    infrared_by_sat: Dict[str, Tuple[str, str]] = {}
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    for name in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, name)
        if not os.path.isdir(folder):
            continue

        m_vis = re.match(r"^(.+)_VisibleBatch\d+$", name, flags=re.IGNORECASE)
        m_ir = re.match(r"^(.+)_IRBatch\d+$", name, flags=re.IGNORECASE)
        if m_vis:
            sat = _norm_text(m_vis.group(1))
            key = _satellite_key(sat)
            visible_by_sat[key] = (sat, os.path.abspath(folder))
        elif m_ir:
            sat = _norm_text(m_ir.group(1))
            key = _satellite_key(sat)
            infrared_by_sat[key] = (sat, os.path.abspath(folder))

    common = sorted(set(visible_by_sat.keys()) & set(infrared_by_sat.keys()))
    if not common:
        raise RuntimeError(f"No visible/infrared folder pairs found under: {data_root}")

    pairs: List[FolderPair] = []
    for key in common:
        sat_vis, vis_dir = visible_by_sat[key]
        sat_ir, ir_dir = infrared_by_sat[key]
        satellite_name = sat_vis if sat_vis else sat_ir
        pairs.append(
            FolderPair(
                satellite_name=satellite_name,
                satellite_key=key,
                visible_dir=vis_dir,
                infrared_dir=ir_dir,
            )
        )
    return pairs


def _find_metadata_files(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(folder):
        for name in names:
            low = name.lower()
            if low.startswith("~$"):
                continue
            if low.endswith(".csv") or low.endswith(".xlsx"):
                files.append(os.path.abspath(os.path.join(root, name)))
    files.sort()
    return files


def _read_csv_rows(path: str) -> List[Dict[str, object]]:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return [dict(r) for r in reader]
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return []


def _xlsx_col_to_index(ref: str) -> int:
    letters = []
    for ch in str(ref):
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    out = 0
    for ch in letters:
        out = out * 26 + (ord(ch) - ord("A") + 1)
    return max(0, out - 1)


def _read_xlsx_rows(path: str) -> List[Dict[str, object]]:
    with zipfile.ZipFile(path, "r") as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall(".//{*}si"):
                texts = [t.text or "" for t in si.findall(".//{*}t")]
                shared.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.find("{*}sheets")
        if sheets is None or len(list(sheets)) == 0:
            return []

        first_sheet = list(sheets)[0]
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = None
        if rel_id and "xl/_rels/workbook.xml.rels" in zf.namelist():
            rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            for rel in rels.findall("{*}Relationship"):
                if rel.attrib.get("Id") == rel_id:
                    target = rel.attrib.get("Target")
                    break
        if not target:
            target = "worksheets/sheet1.xml"
        target = str(target).replace("\\", "/").lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"

        sheet_xml = ET.fromstring(zf.read(sheet_path))
        grid: List[List[object]] = []
        for row in sheet_xml.findall(".//{*}sheetData/{*}row"):
            vals: List[object] = []
            for cell in row.findall("{*}c"):
                col_idx = _xlsx_col_to_index(cell.attrib.get("r", "A1"))
                while len(vals) < col_idx:
                    vals.append("")

                cell_type = cell.attrib.get("t", "")
                value: object = ""
                if cell_type == "inlineStr":
                    node = cell.find("{*}is")
                    if node is not None:
                        value = "".join(t.text or "" for t in node.findall(".//{*}t"))
                else:
                    node = cell.find("{*}v")
                    if node is not None:
                        raw = node.text or ""
                        if cell_type == "s":
                            idx = int(raw)
                            value = shared[idx] if 0 <= idx < len(shared) else raw
                        elif cell_type == "b":
                            value = "1" if raw == "1" else "0"
                        else:
                            value = raw
                vals.append(value)
            grid.append(vals)

        if not grid:
            return []

        width = max(len(r) for r in grid)
        grid = [r + [""] * (width - len(r)) for r in grid]
        header = [_norm_text(x) for x in grid[0]]
        rows: List[Dict[str, object]] = []
        for values in grid[1:]:
            row: Dict[str, object] = {}
            for i, key in enumerate(header):
                if not key:
                    key = f"col_{i+1}"
                row[key] = values[i]
            rows.append(row)
        return rows


def _read_table_rows(path: str) -> List[Dict[str, object]]:
    low = path.lower()
    if low.endswith(".csv"):
        return _read_csv_rows(path)
    if low.endswith(".xlsx"):
        return _read_xlsx_rows(path)
    return []


def _find_frame_col(cols: Sequence[str]) -> Optional[str]:
    if not cols:
        return None
    norm = {_norm_col(c): c for c in cols}
    for k in ["frameidx", "frameindex", "frameid", "idx", "frame"]:
        if k in norm:
            return norm[k]
    for c in cols:
        nc = _norm_col(c)
        if "frame" in nc and ("idx" in nc or "id" in nc):
            return c
    return None


def _find_image_col(cols: Sequence[str]) -> Optional[str]:
    if not cols:
        return None
    norm = {_norm_col(c): c for c in cols}
    for k in ["imagename", "filename", "filepath", "imagepath", "imgname", "img", "image", "name", "pairid"]:
        if k in norm:
            return norm[k]
    for c in cols:
        nc = _norm_col(c)
        if any(tag in nc for tag in ["image", "img", "file", "name", "path"]):
            return c
    return None


def _find_difficulty_col(cols: Sequence[str], modality: str) -> Optional[str]:
    if not cols:
        return None
    norm = {_norm_col(c): c for c in cols}
    preferred = []
    if modality == "visible":
        preferred = [
            "difficultyreasonvisible",
            "visibledifficultyreason",
            "difficultyreasonrgb",
            "rgbdifficultyreason",
        ]
    else:
        preferred = [
            "difficultyreasoninfrared",
            "infrareddifficultyreason",
            "irdifficultyreason",
            "difficultyreasonir",
        ]
    preferred.extend(["difficultyreason", "hardreason", "difficulty"])
    for k in preferred:
        if k in norm:
            return norm[k]
    for c in cols:
        nc = _norm_col(c)
        if "difficulty" in nc or "hard" in nc:
            return c
    return None


def _find_quat_cols(cols: Sequence[str]) -> Optional[Tuple[str, str, str, str]]:
    if not cols:
        return None
    norm = {_norm_col(c): c for c in cols}

    direct_sets = [
        ("quatw", "quatx", "quaty", "quatz"),
        ("quaternionw", "quaternionx", "quaterniony", "quaternionz"),
        ("qw", "qx", "qy", "qz"),
    ]
    for ds in direct_sets:
        if all(k in norm for k in ds):
            return (norm[ds[0]], norm[ds[1]], norm[ds[2]], norm[ds[3]])

    qcols = [c for c in cols if "quat" in _norm_col(c) or _norm_col(c).startswith("q")]
    if len(qcols) >= 4:
        chosen: Dict[str, str] = {}
        for c in qcols:
            nc = _norm_col(c)
            for axis in ["w", "x", "y", "z"]:
                if nc.endswith(axis) and axis not in chosen:
                    chosen[axis] = c
        if all(a in chosen for a in ["w", "x", "y", "z"]):
            return (chosen["w"], chosen["x"], chosen["y"], chosen["z"])
    return None


def _to_float(x: object) -> Optional[float]:
    s = _norm_text(x)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _meta_score(entry: MetaEntry) -> int:
    score = 0
    if entry.frame_idx is not None:
        score += 1
    if entry.image_name:
        score += 1
    if entry.quat is not None:
        score += 2
    if entry.difficulty_reason:
        score += 1
    return score


def _build_metadata_index(folder: str, modality: str) -> Dict[str, MetaEntry]:
    index: Dict[str, MetaEntry] = {}
    files = _find_metadata_files(folder)
    for path in files:
        rows = _read_table_rows(path)
        if not rows:
            continue
        cols = list(rows[0].keys())
        frame_col = _find_frame_col(cols)
        image_col = _find_image_col(cols)
        diff_col = _find_difficulty_col(cols, modality=modality)
        quat_cols = _find_quat_cols(cols)

        for row in rows:
            frame_idx = _extract_int(row.get(frame_col)) if frame_col else None
            image_name = _norm_text(row.get(image_col)) if image_col else ""
            if not image_name and frame_idx is None:
                continue

            quat = None
            if quat_cols is not None:
                q_vals = [_to_float(row.get(c)) for c in quat_cols]
                if all(v is not None for v in q_vals):
                    quat = (float(q_vals[0]), float(q_vals[1]), float(q_vals[2]), float(q_vals[3]))

            difficulty_reason = _norm_text(row.get(diff_col)) if diff_col else ""
            entry = MetaEntry(
                frame_idx=frame_idx,
                image_name=image_name,
                quat=quat,
                difficulty_reason=difficulty_reason,
            )
            score = _meta_score(entry)
            for key in _lookup_keys(frame_idx, image_name):
                prev = index.get(key)
                if prev is None or score > _meta_score(prev):
                    index[key] = entry
    return index


def _scan_images(folder: str) -> List[ImageEntry]:
    out: List[ImageEntry] = []
    for root, _, names in os.walk(folder):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            abspath = os.path.abspath(os.path.join(root, name))
            frame_idx = _extract_int(os.path.splitext(name)[0])
            out.append(ImageEntry(path=abspath, image_name=name, frame_idx=frame_idx))
    out.sort(key=lambda x: x.path.lower())
    return out


def _index_unique(entries: Sequence[ImageEntry], by_frame: bool) -> Dict[str, ImageEntry]:
    idx: Dict[str, ImageEntry] = {}
    for e in entries:
        if by_frame:
            if e.frame_idx is None:
                continue
            key = f"frame:{e.frame_idx}"
        else:
            key = f"name:{_stem(e.image_name).lower()}"
        # keep first to stay deterministic
        if key not in idx:
            idx[key] = e
    return idx


def _pair_visible_infrared(visible: Sequence[ImageEntry], infrared: Sequence[ImageEntry]) -> List[Tuple[ImageEntry, ImageEntry]]:
    pairs: List[Tuple[ImageEntry, ImageEntry]] = []
    used_vis = set()
    used_ir = set()

    vis_by_frame = _index_unique(visible, by_frame=True)
    ir_by_frame = _index_unique(infrared, by_frame=True)
    for key in sorted(set(vis_by_frame.keys()) & set(ir_by_frame.keys())):
        v = vis_by_frame[key]
        i = ir_by_frame[key]
        pairs.append((v, i))
        used_vis.add(v.path)
        used_ir.add(i.path)

    vis_left = [e for e in visible if e.path not in used_vis]
    ir_left = [e for e in infrared if e.path not in used_ir]
    vis_by_name = _index_unique(vis_left, by_frame=False)
    ir_by_name = _index_unique(ir_left, by_frame=False)
    for key in sorted(set(vis_by_name.keys()) & set(ir_by_name.keys())):
        v = vis_by_name[key]
        i = ir_by_name[key]
        pairs.append((v, i))
        used_vis.add(v.path)
        used_ir.add(i.path)

    pairs.sort(key=lambda x: (x[0].frame_idx if x[0].frame_idx is not None else 10**9, x[0].image_name.lower()))
    return pairs


def _pick_meta(meta_index: Dict[str, MetaEntry], frame_idx: Optional[int], image_name: str) -> Optional[MetaEntry]:
    for key in _lookup_keys(frame_idx, image_name):
        if key in meta_index:
            return meta_index[key]
    return None


def _build_rows_for_pair(pair: FolderPair) -> List[Dict[str, object]]:
    vis_images = _scan_images(pair.visible_dir)
    ir_images = _scan_images(pair.infrared_dir)
    image_pairs = _pair_visible_infrared(vis_images, ir_images)

    vis_meta = _build_metadata_index(pair.visible_dir, modality="visible")
    ir_meta = _build_metadata_index(pair.infrared_dir, modality="infrared")

    rows: List[Dict[str, object]] = []
    for vis, ir in image_pairs:
        vis_m = _pick_meta(vis_meta, vis.frame_idx, vis.image_name)
        ir_m = _pick_meta(ir_meta, ir.frame_idx, ir.image_name)

        frame_idx = vis.frame_idx
        if frame_idx is None:
            frame_idx = ir.frame_idx
        if frame_idx is None and vis_m is not None and vis_m.frame_idx is not None:
            frame_idx = vis_m.frame_idx
        if frame_idx is None and ir_m is not None and ir_m.frame_idx is not None:
            frame_idx = ir_m.frame_idx
        if frame_idx is None:
            frame_idx = _extract_int(_stem(vis.image_name))
        if frame_idx is None:
            frame_idx = _extract_int(_stem(ir.image_name))

        quat: Optional[Tuple[float, float, float, float]] = None
        if vis_m is not None and vis_m.quat is not None:
            quat = vis_m.quat
        elif ir_m is not None and ir_m.quat is not None:
            quat = ir_m.quat

        row = {
            "satellite_name": pair.satellite_name,
            "frame_idx": frame_idx if frame_idx is not None else "",
            "visible_path": vis.path,
            "infrared_path": ir.path,
            "quat_w": quat[0] if quat is not None else "",
            "quat_x": quat[1] if quat is not None else "",
            "quat_y": quat[2] if quat is not None else "",
            "quat_z": quat[3] if quat is not None else "",
            "difficulty_reason_visible": (vis_m.difficulty_reason if vis_m is not None else ""),
            "difficulty_reason_infrared": (ir_m.difficulty_reason if ir_m is not None else ""),
            "subset_name": _subset_name(frame_idx),
        }
        rows.append(row)
    return rows


def _split_train_val_test(
    rows: Sequence[Dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total <= 0.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
    train_r = float(train_ratio) / total
    val_r = float(val_ratio) / total

    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        sat = _norm_text(row.get("satellite_name"))
        subset = _norm_text(row.get("subset_name"))
        key = (sat, subset)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []
    for key in sorted(groups.keys()):
        grp = groups[key]
        idx = list(range(len(grp)))
        group_seed = f"{seed}|{key[0]}|{key[1]}"
        rng = random.Random(group_seed)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train.extend(grp[i] for i in idx[:n_train])
        val.extend(grp[i] for i in idx[n_train : n_train + n_val])
        test.extend(grp[i] for i in idx[n_train + n_val :])
    return train, val, test


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in MANIFEST_FIELDS}
            w.writerow(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fire manifests (train/val/test_hard) from paired visible/infrared folders.")
    ap.add_argument("--data-root", default=r"D:\BaiduNetdiskDownload\fire", help="Root folder containing *_VisibleBatch*/ *_IRBatch* folders.")
    ap.add_argument("--out-dir", default=os.path.abspath("."), help="Output directory for fire_train.csv, fire_val.csv, fire_test_hard.csv.")
    ap.add_argument("--train-ratio", type=float, default=0.60, help="Train split ratio (stratified by satellite_name x subset_name).")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (stratified by satellite_name x subset_name).")
    ap.add_argument("--test-ratio", type=float, default=0.25, help="Test split ratio (stratified by satellite_name x subset_name).")
    ap.add_argument("--split-seed", type=int, default=3407, help="Random seed for train/val/test stratified split.")
    ap.add_argument(
        "--test-satellite",
        default="",
        help="DEPRECATED: ignored in mixed-satellite default flow.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _discover_folder_pairs(args.data_root)

    rows_all: List[Dict[str, object]] = []

    for pair in pairs:
        rows = _build_rows_for_pair(pair)
        rows_all.extend(rows)

    if not rows_all:
        raise RuntimeError("No paired rows found from discovered visible/infrared folder pairs.")

    rows_train, rows_val, rows_test_hard = _split_train_val_test(
        rows_all,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    if not rows_train or not rows_val or not rows_test_hard:
        raise RuntimeError(
            "One or more output splits are empty. Adjust --train-ratio/--val-ratio/--test-ratio or verify dataset size."
        )

    rows_train.sort(
        key=lambda r: (
            _satellite_key(_norm_text(r.get("satellite_name"))),
            int(r["frame_idx"]) if _norm_text(r.get("frame_idx")).isdigit() else 10**9,
            _norm_text(r.get("visible_path")).lower(),
        )
    )
    rows_val.sort(
        key=lambda r: (
            _satellite_key(_norm_text(r.get("satellite_name"))),
            int(r["frame_idx"]) if _norm_text(r.get("frame_idx")).isdigit() else 10**9,
            _norm_text(r.get("visible_path")).lower(),
        )
    )
    rows_test_hard.sort(
        key=lambda r: (
            _satellite_key(_norm_text(r.get("satellite_name"))),
            int(r["frame_idx"]) if _norm_text(r.get("frame_idx")).isdigit() else 10**9,
            _norm_text(r.get("visible_path")).lower(),
        )
    )

    out_train = os.path.abspath(os.path.join(args.out_dir, "fire_train.csv"))
    out_val = os.path.abspath(os.path.join(args.out_dir, "fire_val.csv"))
    out_test = os.path.abspath(os.path.join(args.out_dir, "fire_test_hard.csv"))

    _write_csv(out_train, rows_train)
    _write_csv(out_val, rows_val)
    _write_csv(out_test, rows_test_hard)

    print(out_train)
    print(out_val)
    print(out_test)


if __name__ == "__main__":
    main()
