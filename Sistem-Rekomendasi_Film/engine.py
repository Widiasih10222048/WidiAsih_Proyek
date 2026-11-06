import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Filters:
    year_range: Optional[tuple] = None
    types: list = field(default_factory=list)
    genres: list = field(default_factory=list)
    genre_mode: str = "any"
    min_rating: float = 0.0
    min_votes: int = 0
    countries: list = field(default_factory=list)


def muat_paket(path):
    with open(path, "rb") as f:
        paket = pickle.load(f)
    paket["titles"] = paket["data"]["title"].tolist()
    paket["titles_proc"] = [utils.default_process(t) for t in paket["titles"]]
    paket["genre_sets"] = [set(g) for g in paket["data"]["genres"]]
    return paket


def minmax(x):
    rentang = x.max() - x.min()
    if rentang == 0:
        return np.zeros_like(x, dtype=float)
    return (x - x.min()) / rentang


def cari_judul(paket, query, limit=5, score_cutoff=50):
    query = (query or "").strip()
    if not query:
        return []
    data = paket["data"]
    hasil_mentah = process.extract(
        utils.default_process(query),
        paket["titles_proc"],
        scorer=fuzz.WRatio,
        limit=limit * 3,
        score_cutoff=score_cutoff,
    )
    hasil = []
    sudah = set()
    for _, skor, idx in hasil_mentah:
        if idx in sudah:
            continue
        sudah.add(idx)
        baris = data.iloc[idx]
        hasil.append({
            "idx": int(idx),
            "title": baris["title"],
            "year": int(baris["year"]),
            "label": baris["label"],
            "score": float(skor),
        })
        if len(hasil) >= limit:
            break
    return hasil


def buat_mask(paket, f):
    data = paket["data"]
    mask = np.ones(len(data), dtype=bool)
    if f is None:
        return mask

    if f.year_range:
        mask &= data["year"].between(*f.year_range).to_numpy()

    if f.types:
        mask &= data["type"].isin(f.types).to_numpy()

    if f.genres:
        agg = all if f.genre_mode == "all" else any
        mask &= np.array([agg(g in s for g in f.genres) for s in paket["genre_sets"]])

    if f.min_rating:
        mask &= (data["rating"] >= f.min_rating).to_numpy()

    if f.min_votes:
        mask &= (data["votes"] >= f.min_votes).to_numpy()

    if f.countries:
        mask &= data["country"].isin(f.countries).to_numpy()

    return mask


def jelaskan(paket, asal, tujuan, sim_teks):
    alasan = []

    genre_sama = [g for g in asal["genres"] if g in tujuan["genres"]]
    if genre_sama:
        alasan.append("Genre sama: " + ", ".join(genre_sama))

    sutradara_sama = [d for d in asal["directors"] if d in tujuan["directors"]]
    if sutradara_sama:
        alasan.append("Sutradara sama: " + ", ".join(sutradara_sama))

    pemeran_sama = [c for c in asal["cast"] if c in tujuan["cast"]]
    if pemeran_sama:
        alasan.append("Pemeran sama: " + ", ".join(pemeran_sama[:3]))

    if sim_teks >= 0.15:
        alasan.append("Sinopsis bernuansa mirip")

    if asal["country"] == tujuan["country"] and tujuan["country"] != "Unknown":
        alasan.append(f"Sama-sama produksi {tujuan['country']}")

    if tujuan["votes"] >= paket["m"] and tujuan["rating"] >= 7.0:
        alasan.append(
            f"Berkualitas: rating {tujuan['rating']:.1f} dari {tujuan['votes']:,} voter"
        )

    return alasan or ["Profil konten secara keseluruhan mirip"]


def rekomendasi(paket, idx, top_n=10, alpha=0.7, w_meta=0.7, filters=None, pool=150):
    data = paket["data"]

    sim_meta = cosine_similarity(paket["meta_matrix"][idx], paket["meta_matrix"]).ravel()
    sim_teks = cosine_similarity(paket["teks_matrix"][idx], paket["teks_matrix"]).ravel()
    sim = w_meta * sim_meta + (1 - w_meta) * sim_teks

    mask = buat_mask(paket, filters)
    mask[idx] = False
    kandidat = np.where(mask)[0]
    if len(kandidat) == 0:
        return pd.DataFrame()

    kandidat = kandidat[np.argsort(sim[kandidat])[::-1][:pool]]
    sim_n = minmax(sim[kandidat])
    wr_n = minmax(data["wr"].to_numpy()[kandidat])
    skor = alpha * sim_n + (1 - alpha) * wr_n

    urutan = np.argsort(skor)[::-1][:top_n]
    asal = data.iloc[idx]
    baris_hasil = []
    for o in urutan:
        j = int(kandidat[o])
        tujuan = data.iloc[j]
        alasan = jelaskan(paket, asal, tujuan, sim_teks[j])
        baris_hasil.append({
            "idx": j,
            "title": tujuan["title"],
            "year": int(tujuan["year"]),
            "type": tujuan["type"],
            "country": tujuan["country"],
            "genres": ", ".join(tujuan["genres"]),
            "director": ", ".join(tujuan["directors"]),
            "episodes": tujuan["episodes"],
            "platform": tujuan["platform"],
            "rating": float(tujuan["rating"]),
            "votes": int(tujuan["votes"]),
            "weighted_rating": round(float(tujuan["wr"]), 2),
            "similarity": round(float(sim[j]), 3),
            "score": round(float(skor[o]), 3),
            "reasons": alasan,
            "explanation": " · ".join(alasan),
            "synopsis": tujuan["synopsis"],
        })
    return pd.DataFrame(baris_hasil)


def terbaik(paket, filters=None, top_n=20):
    data = paket["data"]
    mask = buat_mask(paket, filters)
    d = data[mask].sort_values("wr", ascending=False).head(top_n)
    out = d[["title", "year", "type", "country", "genres", "rating", "votes", "wr"]].copy()
    out["genres"] = out["genres"].apply(", ".join)
    out = out.rename(columns={"wr": "weighted_rating"})
    return out.round({"weighted_rating": 2}).reset_index(drop=True)
