# Model deteksi anomali saya kalah dari z-score, dan itu hasil terbaik yang pernah saya dapat

Saya menghabiskan beberapa hari membangun ulang model deteksi anomali dari proyek capstone saya.
Autoencoder, 1.839 parameter, fitur kausal, export ONNX, API, dashboard. Lalu saya mengukurnya
dengan benar untuk pertama kalinya, dan hasilnya:

| Detektor | PR-AUC | ROC-AUC | Recall @ FPR 1% |
|---|---|---|---|
| Autoencoder | 0,154 | 0,570 | 0,025 |
| z-score | 0,296 | 0,791 | 0,138 |
| IsolationForest | 0,318 | 0,762 | 0,037 |

Garis acak di 0,105. Model yang saya bangga-banggakan kalah dari `max(|z|) > 3`.

Saya menulis ini bukan untuk mengeluh, tapi karena momen ini yang membedakan orang yang membangun
model dari orang yang mengukur model. Dan karena dua tahun sebelumnya, proyek ini "jalan" tanpa
satu pun metrik.

## Cara saya mengukur (dan kenapa angkanya bisa dipercaya)

Tidak ada label. Tidak ada yang menandai transaksinya "ini anomali" - saya sendiri tidak punya
catatan itu. Jadi labelnya saya buat: anomali disuntikkan ke test set dengan seed tetap, empat
tipe (spike 8-20x, duplikat, burst nominal bulat, drip nominal kecil), dimutasi di tempat supaya
setiap baris tetap punya label 1:1.

Protokol akhirnya:

- **20 replikasi injeksi**, bukan satu. Angka yang dilaporkan adalah pooled PR-AUC dari ~760 label.
- **Threshold per detektor**, dikalibrasi di holdout bersih pada FPR 1%. Bukan satu threshold
  dipakai untuk semua.
- **Split berbasis waktu**, bukan random, karena data transaksi punya urutan.
- **Metrik per tipe anomali**, karena model yang menangkap semua spike tapi tidak satu pun drip
  adalah produk yang berbeda.
- **Cek per kategori**, untuk memastikan tidak ada kategori yang dihukum model. Flag rate di deret
  bersih 0,8% sementara targetnya 1% - aman.

## Kenapa ini terjadi

Dua penyebab, dan keduanya bisa diukur.

**Pertama, data training-nya 175 baris.** Model 1.839 parameter dengan 175 contoh bukan belajar
pola; ia menghafal. Ketika saya latih arsitektur yang sama di korpus sintetis 10.000 baris (7.000
untuk training), autoencoder menang: PR-AUC 0,215 vs z-score 0,173. Arsitekturnya bukan masalah.
Ukuran datanya yang masalah.

**Kedua, anomali yang saya suntikkan berbentuk outlier magnitudo.** Untuk pola seperti itu,
z-score memang detektor yang tepat - itu literally definisi z-score. Saya membangun model
multivariat lalu mengujinya dengan soal satu dimensi.

## Tiga kesalahan evaluasi yang saya perbaiki di tengah jalan

**Test set 38 baris, injeksi 2% = 4 positif.** PR-AUC dari empat positif itu bukan angka, itu
kebisingan. Saya naikkan ke 10% injeksi dan 20 replikasi. Beda hasilnya besar: satu run bisa 0,10,
pooled 0,154.

**Threshold dipakai ulang antar detektor.** Saya sempat memberi IsolationForest threshold
rekonstruksi milik autoencoder. Skor dua detektor itu beda skala. Baseline-nya jadi tidak mungkin
menang - dan saya hampir menyimpulkan "IF buruk" dari kesalahan kalibrasi saya sendiri.

**Fitur baru diuji tanpa kontrol.** Saya tambah fitur z-score relatif kategori, berharap PR-AUC
naik. Turun: 0,170 -> 0,154. Godaannya adalah menghapus tabel itu. Saya publikasikan sebagai
ablation negatif (`reports/ablation.md`), karena ablation yang hanya memuat hasil positif itu
iklan, bukan eksperimen.

## Setelah tahu hasilnya, apa yang saya ubah

Endpoint `/score` sekarang mengembalikan field `detector`, dan isinya `zscore`. Autoencoder tetap
ada di repo, terlatih, tereksekusi protokol yang sama - sebagai eksperimen sampai ia menang.

Yang saya tolak: menyebut proyek ini "AI" sambil mengirim model yang kalah dari baseline. Itu bukan
marketing, itu utang yang akan ditagih di interview pertama.

## Kalau kamu di posisi saya

Checklist yang saya harap saya punya dua tahun lalu:

1. Sebelum menulis model, tulis dulu cara mengukurnya. Kalau tidak ada label, tulis cara membuat
   labelnya.
2. Jalankan baseline paling bodoh lebih dulu: z-score, rata-rata bergerak, median.
3. Hitung jumlah positif. Di bawah ~20, angkanya belum layak dikutip.
4. Kalibrasi threshold di data bersih, sekali, dan simpan nilainya.
5. Laporkan sebaran, bukan satu run.

Repo: `github.com/drafi19/selena-insights`
