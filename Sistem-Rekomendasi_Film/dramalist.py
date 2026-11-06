import html
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, utils
from engine import Filters, buat_mask, cari_judul, muat_paket, rekomendasi

BASE = Path(__file__).parent
PKL_PATH = BASE / "movie_marger.joblib"
LANGKAH = 12
MAKS_REKOMENDASI = 150
MAKS_DAFTAR = 300
NAV = ["Beranda", "Rekomendasi", "Daftar Film", "Tentang"]
URUTAN = ["Rekomendasi", "Rating tertinggi", "Terbaru", "Terpopuler", "Judul A–Z"]

st.set_page_config(page_title="Filmora — Find Your Next Favorite Movie", page_icon="🎬",
                   layout="wide", initial_sidebar_state="auto")
st.markdown(f"<style>{(BASE / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat data film…")
def load_paket():
    paket = muat_paket(str(PKL_PATH))
    d = paket["data"]
    paket["orang"] = (d["cast"].apply(", ".join) + " | " + d["directors"].apply(", ".join)).str.lower()
    paket["genre_str"] = d["genres"].apply(lambda g: " | ".join(g)).str.lower()
    return paket


if not PKL_PATH.exists():
    st.error(f"File **{PKL_PATH.name}** tidak ditemukan di folder `{BASE}`. "
             "Pastikan file `movie_marger.joblib` ada di folder yang sama dengan dramalist.py.")
    st.stop()

paket = load_paket()
DATA = paket["data"]


@st.cache_data(show_spinner=False)
def opsi_filter(_paket):
    d = _paket["data"]
    genre = [g for g, _ in Counter(g for lst in d["genres"] for g in lst).most_common(14)]
    negara = [c for c, _ in d.loc[d["country"] != "Unknown", "country"].value_counts().head(30).items()]
    return {"genre": genre, "negara": negara, "tipe": sorted(d["type"].unique().tolist()),
            "tahun_min": int(d["year"].min()), "tahun_max": int(d["year"].max())}


OPSI = opsi_filter(paket)


def esc(x):
    return html.escape(str(x), quote=True).replace("$", "&#36;")


def potong(teks, n):
    teks = re.sub(r"\s+", " ", str(teks or "")).strip()
    return teks if len(teks) <= n else teks[:n].rsplit(" ", 1)[0] + "…"


def angka(n):
    return f"{int(n):,}".replace(",", ".")


ICON_BM = ('<svg class="bm" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#b9736e" '
           'stroke-width="1.8" stroke-linejoin="round"><path d="M6 3h12v18l-6-4.5L6 21z"/></svg>')

ICON_LOGO = ('<svg width="46" height="46" viewBox="0 0 48 48" fill="none"><rect x="6" y="18" width="36" height="22" rx="3" '
             'fill="#d79a94"/><path d="M6 18l32-8 3 9-32 8z" fill="#c4817b"/><path d="M12 15l5 6M21 13l5 6M30 11l5 6" '
             'stroke="#fbf6f1" stroke-width="3" stroke-linecap="round"/><path d="M20 26l9 5-9 5z" fill="#fbf6f1"/></svg>')

ICON_USER = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6f5a55" stroke-width="1.8" '
             'stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c1-4 4-6 8-6s7 2 8 6"/></svg>')

ICON_FUNNEL = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c4817b" stroke-width="1.8" '
               'stroke-linejoin="round"><path d="M4 5h16l-6 8v6l-4-2v-4z"/></svg>')

ICON_LEAF = ('<svg class="hero-leaf" width="110" height="150" viewBox="0 0 110 150" fill="none" stroke="#a98b84" '
             'stroke-width="1.4"><path d="M55 148V40"/><path d="M55 90c-26-4-36-26-32-48 22 2 36 22 32 48z" fill="#e9cfc8"/>'
             '<path d="M55 70c22-2 34-20 32-42-20 2-34 18-32 42z" fill="#eed8d1"/><path d="M55 40c-12-6-16-20-10-34 12 6 16 20 10 34z" '
             'fill="#e4c7c0"/></svg>')


def kartu(r, buka=False):
    meta = [str(r["year"]), r["type"]] + ([r["country"]] if r["country"] != "Unknown" else [])
    chips = "".join(f'<span class="chip">{esc(g)}</span>' for g in r["genres"][:3])
    rating = f'{r["rating"]:.1f}' if r["rating"] > 0 else "–"
    voter = f'<span class="vt">· {angka(r["votes"])} voter</span>' if r["votes"] > 0 else ""
    sim = f'<span class="sim">Kemiripan {r["similarity"] * 100:.0f}%</span>' if r.get("similarity") is not None else ""
    sinopsis = esc(potong(r["synopsis"], 170)) or "<i>Sinopsis belum tersedia.</i>"

    info = []
    if r["directors"]:
        info.append(f'<b>Sutradara:</b> {esc(", ".join(r["directors"][:2]))}')
    if r["cast"]:
        info.append(f'<b>Pemeran:</b> {esc(", ".join(r["cast"][:4]))}')
    if pd.notna(r["episodes"]) and r["episodes"]:
        info.append(f'<b>Episode:</b> {int(r["episodes"])}')
    if isinstance(r["platform"], str) and r["platform"].strip():
        info.append(f'<b>Tayang di:</b> {esc(potong(r["platform"], 60))}')
    alasan = ""
    if r.get("reasons"):
        alasan = "<ul class='why'>" + "".join(f"<li>{esc(a)}</li>" for a in r["reasons"]) + "</ul>"
    judul_detail = "Kenapa direkomendasikan?" if r.get("reasons") else "Detail"

    return (
        '<div class="card">' + ICON_BM
        + f'<div class="card-title">{esc(r["title"])}</div>'
        + f'<div class="card-meta">{esc(" · ".join(meta))}</div>'
        + f'<div class="chips">{chips}</div>'
        + f'<div class="rate"><span class="star">★</span>{rating}{voter}{sim}</div>'
        + f'<p class="syn">{sinopsis}</p>'
        + f'<details class="more"{" open" if buka else ""}><summary>{judul_detail}</summary>{alasan}'
        + f'<div class="info">{"<br>".join(info)}</div></details>'
        + "</div>"
    )


def tampil_grid(rows, buka=False):
    if not rows:
        st.markdown('<div class="empty">😕 Belum ada film yang cocok. Coba longgarkan filter '
                    '(genre, tahun, atau rating minimum).</div>', unsafe_allow_html=True)
        return
    st.markdown('<div class="grid">' + "".join(kartu(r, buka) for r in rows) + "</div>", unsafe_allow_html=True)


def kunci_filter(f):
    return (f.year_range, tuple(f.types), tuple(f.genres), f.genre_mode, f.min_rating, f.min_votes, tuple(f.countries))


def dari_kunci(k):
    return Filters(year_range=k[0], types=list(k[1]), genres=list(k[2]), genre_mode=k[3],
                   min_rating=k[4], min_votes=k[5], countries=list(k[6]))


def baris_dari_idx(idx, ekstra=None):
    d = DATA.iloc[idx]
    r = {"idx": int(idx), "title": d["title"], "year": int(d["year"]), "type": d["type"], "country": d["country"],
         "genres": list(d["genres"]), "cast": list(d["cast"]), "directors": list(d["directors"]),
         "synopsis": d["synopsis"], "rating": float(d["rating"]), "votes": int(d["votes"]),
         "episodes": d["episodes"], "platform": d["platform"], "wr": float(d["wr"]),
         "similarity": None, "reasons": None}
    if ekstra:
        r.update(ekstra)
    return r


@st.cache_data(show_spinner="Mencari film yang cocok…", max_entries=64)
def ambil_rekomendasi(_paket, idx, jumlah, kf):
    hasil = rekomendasi(_paket, idx, top_n=jumlah, filters=dari_kunci(kf))
    return [baris_dari_idx(r["idx"], {"similarity": r["similarity"], "reasons": r["reasons"]})
            for r in hasil.to_dict("records")]


@st.cache_data(show_spinner=False, max_entries=64)
def ambil_daftar(_paket, jumlah, kf, kata=""):
    mask = buat_mask(_paket, dari_kunci(kf))
    if kata:
        ql = re.escape(kata.lower())
        cocok = (_paket["orang"].str.contains(rf"\b{ql}\b", regex=True)
                 | _paket["genre_str"].str.contains(rf"\b{ql}\b", regex=True)
                 | _paket["data"]["title"].str.lower().str.contains(ql, regex=True))
        mask &= cocok.to_numpy()
    d = _paket["data"][mask].sort_values("wr", ascending=False).head(jumlah)
    return [baris_dari_idx(i) for i in d.index]


@st.cache_data(show_spinner=False)
def cari_orang_genre(_paket, q):
    ql = re.escape(q.lower())
    cocok = (_paket["orang"].str.contains(rf"\b{ql}\b", regex=True)
             | _paket["genre_str"].str.contains(rf"\b{ql}\b", regex=True))
    return int(cocok.sum())


def cari_film(q, limit=5):
    qp = utils.default_process(q)
    hasil = []
    for c in cari_judul(paket, q, limit=20, score_cutoff=50):
        tp = utils.default_process(c["title"])
        skor = max(fuzz.ratio(qp, tp), 0.9 * fuzz.token_set_ratio(qp, tp))
        if skor >= 60:
            hasil.append({**c, "score": int(skor), "votes": int(DATA.at[c["idx"], "votes"])})
    hasil.sort(key=lambda c: (-c["score"], -c["votes"]))
    return hasil[:limit]


def urutkan(rows, opsi):
    if opsi == "Rekomendasi" or not rows:
        return rows
    kolom, naik = {"Rating tertinggi": ("wr", False), "Terbaru": ("year", False),
                   "Terpopuler": ("votes", False), "Judul A–Z": ("title", True)}[opsi]
    return sorted(rows, key=lambda r: (r[kolom].lower() if kolom == "title" else r[kolom]), reverse=not naik)


c_logo, c_nav, c_user = st.columns([2.3, 4.7, 0.6], vertical_alignment="center")
with c_logo:
    st.markdown(f'<div class="brand">{ICON_LOGO}<div><div class="brand-name">Filmora</div>'
                '<div class="brand-tag">Find Your Next Favorite Movie</div></div></div>', unsafe_allow_html=True)
with c_nav:
    nav = st.segmented_control("Menu", NAV, default="Beranda", key="nav", label_visibility="collapsed") or "Beranda"
with c_user:
    st.markdown(f'<div class="hdr-avatar">{ICON_USER}</div>', unsafe_allow_html=True)
st.markdown('<hr class="sep">', unsafe_allow_html=True)


if nav == "Tentang":
    st.markdown('<div class="sec-title">Tentang Filmora</div>'
                '<p class="sec-sub">Sistem rekomendasi film &amp; drama berbasis konten (content-based filtering).</p>',
                unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("Judul tersedia", angka(len(DATA)))
    k2.metric("Rentang tahun", f'{OPSI["tahun_min"]}–{OPSI["tahun_max"]}')
    k3.metric("Negara", angka(DATA.loc[DATA["country"] != "Unknown", "country"].nunique()))
    st.markdown(
        """
**Cara kerja**
1. **Pencarian judul** toleran salah ketik (fuzzy matching dengan RapidFuzz).
2. **Kemiripan metadata** — genre, sutradara, pemeran, dan negara diubah menjadi vektor TF-IDF.
3. **Kemiripan teks** — sinopsis dan tag diubah menjadi vektor TF-IDF (unigram + bigram).
4. Keduanya digabung dengan **cosine similarity**, lalu dikombinasikan dengan **weighted rating** (rumus IMDB)
   supaya film berkualitas dengan banyak voter lebih diutamakan daripada film dengan 1–2 voter.
5. Setiap rekomendasi disertai **alasan** (genre/sutradara/pemeran sama, dsb).

**Catatan:** urutan *Rating tertinggi* memakai weighted rating, bukan rating mentah.
Rekomendasi terbatas pada data yang ada di dataset.
        """
    )
    st.stop()


with st.sidebar:
    st.markdown(f'<div class="flt-head">{ICON_FUNNEL}Filter</div>', unsafe_allow_html=True)
    with st.form("form_filter", border=False):
        with st.expander("Genre", expanded=True):
            f_genre = st.pills("Genre", OPSI["genre"], selection_mode="multi", key="f_genre",
                               label_visibility="collapsed") or []
            st.caption("Kosongkan = semua genre")
        with st.expander("Tahun Rilis", expanded=False):
            tahun = list(range(OPSI["tahun_min"], OPSI["tahun_max"] + 1))
            f_dari = st.selectbox("Dari tahun", tahun, index=None, placeholder="Dari tahun",
                                  key="f_dari", label_visibility="collapsed")
            f_sampai = st.selectbox("Sampai tahun", tahun[::-1], index=None, placeholder="Sampai tahun",
                                    key="f_sampai", label_visibility="collapsed")
        with st.expander("Rating Minimum", expanded=False):
            f_rating = st.selectbox("Rating minimum", ["Semua Rating", "5.0+", "6.0+", "7.0+", "8.0+", "9.0+"],
                                    key="f_rating", label_visibility="collapsed")
            f_votes = st.selectbox("Voter minimum", ["Semua voter", "10+", "50+", "100+", "500+", "1.000+"],
                                   key="f_votes", label_visibility="collapsed")
        with st.expander("Tipe & Negara", expanded=False):
            f_tipe = st.pills("Tipe", OPSI["tipe"], selection_mode="multi", key="f_tipe",
                              label_visibility="collapsed") or []
            f_negara = st.multiselect("Negara", OPSI["negara"], key="f_negara", placeholder="Pilih negara",
                                      label_visibility="collapsed")
        st.form_submit_button("Terapkan Filter", type="primary", icon=":material/filter_alt:",
                              use_container_width=True)

rentang = None
if f_dari or f_sampai:
    a, b = f_dari or OPSI["tahun_min"], f_sampai or OPSI["tahun_max"]
    rentang = (min(a, b), max(a, b))

FILTERS = Filters(
    year_range=rentang,
    types=list(f_tipe),
    genres=list(f_genre),
    genre_mode="any",
    min_rating=0.0 if f_rating == "Semua Rating" else float(f_rating.replace("+", "")),
    min_votes=0 if f_votes == "Semua voter" else int(f_votes.replace("+", "").replace(".", "")),
    countries=list(f_negara),
)
KF = kunci_filter(FILTERS)


if nav == "Beranda":
    st.markdown(
        '<div class="hero"><div class="hero-eyebrow">Sistem Rekomendasi Film</div>'
        '<div class="hero-title">Temukan film yang sesuai<br>dengan selera kamu</div>'
        '<p class="hero-text">Dapatkan rekomendasi film berdasarkan kemiripan genre, aktor, dan tema favoritmu.</p>'
        '<div class="hero-deco">Film yang tepat,<br>untuk setiap suasana<br>kamu.</div>' + ICON_LEAF + "</div>",
        unsafe_allow_html=True)
else:
    judul = "Rekomendasi Film" if nav == "Rekomendasi" else "Daftar Film"
    sub = ("Ketik judul film favoritmu, lalu lihat film-film yang paling mirip."
           if nav == "Rekomendasi" else "Jelajahi seluruh koleksi film dan drama.")
    st.markdown(f'<div class="sec-title">{judul}</div><p class="sec-sub">{sub}</p>', unsafe_allow_html=True)

st.markdown('<div class="hint-filter">☰ Buka menu di pojok kiri atas untuk memakai filter.</div>', unsafe_allow_html=True)
q = st.text_input("Cari", key="q", placeholder="Cari judul film, aktor, atau genre…",
                  label_visibility="collapsed").strip()


mode, idx_acuan, kata, judul_hasil, sub_hasil = "terbaik", None, "", "Rekomendasi Untukmu", \
    "Film berkualitas terbaik sesuai filter kamu. Ketik judul favoritmu di atas untuk rekomendasi yang lebih personal."

if nav == "Daftar Film":
    mode, judul_hasil = "daftar", "Semua Film"
    sub_hasil = "Diurutkan menurut kualitas (weighted rating)." if not q else f"Hasil pencarian untuk “{q}”."
    kata = q
elif q:
    kandidat = cari_film(q)
    persis = any(c["title"].lower() == q.lower() for c in kandidat)
    if not persis and len(q) >= 3 and cari_orang_genre(paket, q) >= 3:
        mode, kata = "kata", q
        judul_hasil = f"Hasil untuk “{q}”"
        sub_hasil = "Film dengan aktor, sutradara, atau genre yang cocok. Ketik judul film untuk mendapat rekomendasi."
    elif kandidat:
        pilih = st.selectbox("Menampilkan rekomendasi untuk:", list(range(len(kandidat))), key=f"acuan::{q}",
                             format_func=lambda i: f'{kandidat[i]["label"]}  ·  kecocokan {kandidat[i]["score"]:.0f}%')
        mode, idx_acuan = "rekomendasi", kandidat[pilih]["idx"]
        judul_hasil = f'Karena kamu suka {kandidat[pilih]["title"]}'
        sub_hasil = "Film yang paling mirip dalam genre, tema, pemeran, dan kualitasnya."
    else:
        mode, judul_hasil, sub_hasil = "kosong", "Film tidak ditemukan", \
            "Coba periksa ejaan judul, atau cari dengan nama aktor atau genre."


c_t, c_s = st.columns([2.3, 1.2], vertical_alignment="bottom")
with c_t:
    st.markdown(f'<div class="sec-title">{esc(judul_hasil)}</div><p class="sec-sub">{esc(sub_hasil)}</p>',
                unsafe_allow_html=True)
with c_s:
    urut = st.selectbox("Urutkan", URUTAN, key="urut", label_visibility="collapsed")

sig = (nav, q, idx_acuan, KF)
if st.session_state.get("sig") != sig:
    st.session_state["sig"] = sig
    st.session_state["jumlah"] = LANGKAH
jumlah = st.session_state.get("jumlah", LANGKAH)


if mode == "kosong":
    rows, maks = [], 0
elif mode == "rekomendasi":
    maks = MAKS_REKOMENDASI
    rows = ambil_rekomendasi(paket, idx_acuan, min(jumlah, maks), KF)
else:
    maks = MAKS_DAFTAR
    rows = ambil_daftar(paket, min(jumlah, maks), KF, kata)

tampil_grid(urutkan(rows, urut), buka=(mode == "rekomendasi" and nav == "Rekomendasi"))

if rows and len(rows) >= jumlah and jumlah < maks:
    _, tengah, _ = st.columns([1, 1, 1])
    if tengah.button("Tampilkan lebih banyak", use_container_width=True):
        st.session_state["jumlah"] = jumlah + LANGKAH
        st.rerun()
if rows:
    st.caption(f"Menampilkan {len(rows)} film")