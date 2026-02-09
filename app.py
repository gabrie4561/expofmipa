import streamlit as st
import numpy as np
import pandas as pd

# ==============================
# DATA DASAR & KONFIGURASI
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

    # Data Fuzzy yang sudah dirapikan
    fuzzy_data = {
        "HP": [(0.544414,0.765,0.8812), (0.53831,0.76335,0.89047), (0.54761,0.7683,0.87581)],
        "P":  [(0.55789,0.77971,0.88227), (0.54298,0.76459,0.87977), (0.55344,0.78012,0.894)],
        "KP": [(0.49492,0.712,0.8528), (0.49237,0.71289,0.86127), (0.50876,0.73038,0.86712)],
        "HO": [(0.55334,0.77443,0.88013), (0.52525,0.74663,0.87220), (0.53063,0.75545,0.88478)],
        "KR": [(0.56464,0.79017,0.89541), (0.52824,0.7521,0.88549), (0.52284,0.74196,0.86457)],
        "KM": [(0.57075,0.79256,0.88192), (0.56015,0.78093,0.87761), (0.56296,0.78897,0.89365)],
        "KK": [(0.50063,0.72037,0.86675), (0.47474,0.69039,0.84050), (0.48461,0.70308,0.85206)]
    }

    df = pd.DataFrame(fuzzy_data, index=alternatives)

    # Pembobotan Matriks Fuzzy
    weighted = {}
    for c in df.columns:
        w = weights[c]
        weighted[c] = [(l*w,m*w,u*w) for l,m,u in df[c]]

    wdf = pd.DataFrame(weighted, index=alternatives)
    
    # Defuzzifikasi (Crisp Value)
    crisp = wdf.applymap(lambda x:(x[0]+4*x[1]+x[2])/6)

    cost = ["HP","HO"]
    benefit = ["P","KP","KR","KM","KK"]

    # Menentukan FPIS dan FNIS
    FPIS, FNIS = {},{}
    for c in crisp.columns:
        if c in benefit:
            FPIS[c]=crisp[c].max()
            FNIS[c]=crisp[c].min()
        else:
            FPIS[c]=crisp[c].min()
            FNIS[c]=crisp[c].max()

    # Menghitung Jarak (Distance)
    Dp,Dn={},{}
    for a in alternatives:
        Dp[a]=np.sqrt(((crisp.loc[a]-pd.Series(FPIS))**2).sum())
        Dn[a]=np.sqrt(((crisp.loc[a]-pd.Series(FNIS))**2).sum())

    # Menghitung Closeness Coefficient (CC)
    CC = {a:Dn[a]/(Dp[a]+Dn[a]) for a in alternatives}
    
    # Mengurutkan ranking dari nilai CC tertinggi ke terendah
    # Hasil berupa list of tuples: [('GO', 0.8), ('SF', 0.7), ...]
    sorted_ranking = sorted(CC.items(), key=lambda item: item[1], reverse=True)

    return sorted_ranking, crisp

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="SPK Rekomendasi Makanan", page_icon="🍽️")

st.title("🍽️ Sistem Rekomendasi Aplikasi Pemesanan Makanan")
st.write("Silakan isi preferensi Anda berdasarkan perbandingan berikut:")

user_input = {}

# Form Input User
with st.form("ahp_form"):
    for i, (A, B) in enumerate(questions, start=1):
        st.markdown(f"**{i}. {criteria_full[A]}** vs **{criteria_full[B]}**")
        
        # Pilihan disederhanakan agar UI lebih rapi
        options = [
            f"{A} sama penting dengan {B}",
            f"{A} sedikit lebih penting",
            f"{A} lebih penting",
            f"{A} jauh lebih penting",
            f"{A} mutlak lebih penting",
            f"{B} sedikit lebih penting",
            f"{B} lebih penting",
            f"{B} jauh lebih penting",
            f"{B} mutlak lebih penting"
        ]
        
        choice = st.selectbox(f"Mana yang lebih penting?", options, key=f"q_{i}", label_visibility="collapsed")
        st.divider()

        # Mapping Nilai AHP
        if choice == options[0]: val = 1
        elif choice == options[1]: val = 3
        elif choice == options[2]: val = 5
        elif choice == options[3]: val = 7
        elif choice == options[4]: val = 9
        elif choice == options[5]: val = 1/3
        elif choice == options[6]: val = 1/5
        elif choice == options[7]: val = 1/7
        elif choice == options[8]: val = 1/9
        
        user_input[(A, B)] = val
    
    submitted = st.form_submit_button("🔍 Lihat Rekomendasi")

if submitted:
    weights = run_ahp(user_input)
    sorted_ranking, crisp = run_fuzzy_topsis(weights)

    # Ambil Peringkat 1, 2, dan 3
    best_code, best_score = sorted_ranking[0]
    second_code, second_score = sorted_ranking[1]
    third_code, third_score = sorted_ranking[2]

    # Hitung Kontribusi Kriteria untuk Peringkat 1
    raw_scores = crisp.loc[best_code]
    contribution_ratio = raw_scores / raw_scores.sum()
    top_criteria = contribution_ratio.sort_values(ascending=False).head(3)

    st.markdown("## 🏆 HASIL REKOMENDASI")
    st.markdown("---")

    # === TAMPILAN PERINGKAT 1 ===
    st.success(f"🥇 **Rekomendasi Utama: {app_name[best_code]}**")
    
    col_score, col_detail = st.columns([1, 2])
    
    with col_score:
        st.metric(label="Skor Preferensi", value=f"{best_score:.4f}")
    
    with col_detail:
        st.write("🔍 **Alasan rekomendasi ini (Kontribusi Kriteria):**")
        for crit, val in top_criteria.items():
             st.write(f"- {criteria_full[crit]}: **{val*100:.1f}%**")

    st.markdown("---")
    
    # === TAMPILAN PERINGKAT 2 & 3 ===
    st.subheader("📊 Peringkat Alternatif Lainnya")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"🥈 **Peringkat 2**")
        st.markdown(f"### {app_name[second_code]}")
        st.write(f"Skor: **{second_score:.4f}**")
        
    with col2:
        st.warning(f"🥉 **Peringkat 3**")
        st.markdown(f"### {app_name[third_code]}")
        st.write(f"Skor: **{third_score:.4f}**")

    # Catatan Kaki
    st.markdown("---")
    st.caption("💡 Skor preferensi dihitung menggunakan metode Fuzzy TOPSIS berdasarkan bobot AHP dari input Anda.")
