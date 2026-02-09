import streamlit as st
import numpy as np
import pandas as pd

# ==============================
# DATA DASAR
# ==============================
criteria = ["HP","P","KP","HO","KR","KM","KK"]

criteria_full = {
    "HP": "Harga Produk",
    "HO": "Harga Ongkir",
    "P":  "Promo",
    "KP": "Kecepatan Pengantaran",
    "KR": "Kelengkapan Restoran dan Menu",
    "KM": "Keadaan Makanan",
    "KK": "Keramahan Kurir"
}

questions = [
    ("HP","P"), ("HP","KP"), ("HP","HO"), ("HP","KR"), ("HP","KM"), ("HP","KK"),
    ("KP","P"), ("KP","HO"), ("KP","KR"), ("KP","KM"), ("KP","KK"),
    ("P","HO"), ("P","KR"), ("P","KM"), ("P","KK"),
    ("HO","KR"), ("HO","KM"), ("HO","KK"),
    ("KR","KM"), ("KR","KK"),
    ("KM","KK")
]

app_name = {
    "GO": "GoFood (Gojek)",
    "GR": "GrabFood (Grab)",
    "SF": "ShopeeFood (Shopee)"
}

scale_map = {
    "Sama penting": 1,
    "Sedikit lebih penting": 3,
    "Lebih penting": 5,
    "Jauh lebih penting": 7,
    "Sangat jauh lebih penting": 9
}

# ==============================
# AHP FUNCTION
# ==============================
def run_ahp(user_input):
    n = len(criteria)
    ahp = pd.DataFrame(np.ones((n,n)), index=criteria, columns=criteria)

    for (A,B), val in user_input.items():
        ahp.loc[A,B] = val
        ahp.loc[B,A] = 1/val

    norm = ahp / ahp.sum()
    weights = norm.mean(axis=1)

    

    return weights.to_dict()

# ==============================
# FUZZY TOPSIS FUNCTION
# ==============================
def run_fuzzy_topsis(weights):
    alternatives = ["GO","GR","SF"]

    fuzzy_data = {
        "HP": [(0.54,0.77,0.88),(0.54,0.76,0.89),(0.55,0.77,0.88)],
        "P":  [(0.55,0.77,0.88),(0.53,0.75,0.87),(0.53,0.76,0.88)],
        "KP": [(0.57,0.79,0.88),(0.56,0.78,0.88),(0.56,0.79,0.89)],
        "HO": [(0.65,0.80,0.91),(0.62,0.79,0.90),(0.64,0.80,0.90)],
        "KR": [(0.79,0.79,0.90),(0.53,0.75,0.89),(0.52,0.74,0.86)],
        "KM": [(0.69,0.84,0.92),(0.67,0.82,0.92),(0.68,0.83,0.93)],
        "KK": [(0.50,0.72,0.87),(0.47,0.69,0.84),(0.48,0.70,0.85)]
    }

    df = pd.DataFrame(fuzzy_data, index=alternatives)

    weighted = {}
    for c in df.columns:
        w = weights[c]
        weighted[c] = [(l*w,m*w,u*w) for l,m,u in df[c]]

    wdf = pd.DataFrame(weighted, index=alternatives)
    crisp = wdf.applymap(lambda x:(x[0]+4*x[1]+x[2])/6)

    cost = ["HP","HO"]
    benefit = ["P","KP","KR","KM","KK"]

    FPIS, FNIS = {},{}
    for c in crisp.columns:
        if c in benefit:
            FPIS[c]=crisp[c].max()
            FNIS[c]=crisp[c].min()
        else:
            FPIS[c]=crisp[c].min()
            FNIS[c]=crisp[c].max()

    Dp,Dn={},{}
    for a in alternatives:
        Dp[a]=np.sqrt(((crisp.loc[a]-pd.Series(FPIS))**2).sum())
        Dn[a]=np.sqrt(((crisp.loc[a]-pd.Series(FNIS))**2).sum())

    CC = {a:Dn[a]/(Dp[a]+Dn[a]) for a in alternatives}
    best = max(CC, key=CC.get)

    return best, CC, crisp

def format_result(best, scores, crisp):
    app_full_name = {
        "GO": "GoFood (Gojek)",
        "GR": "GrabFood (Grab)",
        "SF": "ShopeeFood (Shopee)"
    }

    criteria_full_name = {
        "HP": "Harga Produk",
        "HO": "Harga Ongkir",
        "P":  "Promo",
        "KP": "Kecepatan Pengantaran",
        "KR": "Kelengkapan Restoran dan Menu",
        "KM": "Keadaan Makanan",
        "KK": "Keramahan Kurir"
    }

    # Nilai preferensi akhir
    best_score = scores[best]

    # Kontribusi relatif
    raw_scores = crisp.loc[best]
    contribution_ratio = raw_scores / raw_scores.sum()

    top_criteria = (
        contribution_ratio
        .sort_values(ascending=False)
        .head(3)
    )

    return {
        "best_app": app_full_name[best],
        "best_score": best_score,
        "top_criteria": {
            criteria_full_name[k]: v * 100
            for k, v in top_criteria.items()
        }
    }

# ==============================
# STREAMLIT UI
# ==============================
st.title("🍽️ Sistem Rekomendasi Aplikasi Pemesanan Makanan")

st.write("Silakan isi preferensi Anda berdasarkan perbandingan berikut:")

user_input = {}

for i, (A, B) in enumerate(questions, start=1):

    st.markdown(f"### Pertanyaan {i}")
    st.write("Menurut Anda, mana yang lebih penting?")
    st.markdown(f"**{criteria_full[A]}** dibandingkan **{criteria_full[B]}**")

    options = [
        f"{A} dan {B} sama penting",
        f"{A} sedikit lebih penting dari {B}",
        f"{A} lebih penting dari {B}",
        f"{A} jauh lebih penting dari {B}",
        f"{A} sangat jauh lebih penting dari {B}",
        f"{B} sedikit lebih penting dari {A}",
        f"{B} lebih penting dari {A}",
        f"{B} jauh lebih penting dari {A}",
        f"{B} sangat jauh lebih penting dari {A}"
    ]

    choice = st.selectbox(
        label="Jawaban:",
        options=options,
        key=f"q_{i}"
    )

    if choice == f"{A} dan {B} sama penting":
        val = 1
    elif choice == f"{A} sedikit lebih penting dari {B}":
        val = 3
    elif choice == f"{A} lebih penting dari {B}":
        val = 5
    elif choice == f"{A} jauh lebih penting dari {B}":
        val = 7
    elif choice == f"{A} sangat jauh lebih penting dari {B}":
        val = 9
    elif choice == f"{B} sedikit lebih penting dari {A}":
        val = 1 / 3
    elif choice == f"{B} lebih penting dari {A}":
        val = 1 / 5
    elif choice == f"{B} jauh lebih penting dari {A}":
        val = 1 / 7
    elif choice == f"{B} sangat jauh lebih penting dari {A}":
        val = 1 / 9

    user_input[(A, B)] = val


if st.button("🔍 Lihat Rekomendasi"):
    weights = run_ahp(user_input)
    best, scores, crisp = run_fuzzy_topsis(weights)

    result = format_result(best, scores, crisp)

    st.markdown("## 🍽️ HASIL REKOMENDASI APLIKASI PEMESANAN MAKANAN")
    st.markdown("---")

    st.success("✅ **Rekomendasi terbaik untuk Anda adalah:**")
    st.markdown(f"### 👉 {result['best_app']}")

    st.markdown(f"📊 **Nilai preferensi akhir:** `{result['best_score']:.4f}`")

    st.markdown("### 🔍 Alasan utama rekomendasi ini:")
    for crit, val in result["top_criteria"].items():
        st.markdown(f"- **{crit}**, berkontribusi sekitar **{val:.1f}%**")

    st.info(
        "💡 **Catatan:**\n\n"
        "Persentase menunjukkan kontribusi relatif setiap kriteria "
        "terhadap skor akhir alternatif yang direkomendasikan."
    )