# Tiga bug yang membuat dashboard ML terlihat hidup (padahal tidak)

Dashboard capstone saya dulu menampilkan daftar "transaksi tidak wajar". Dosen dan reviewer suka.
Waktu saya buka lagi kodenya dua tahun kemudian, saya menemukan bahwa daftar itu tidak mengandung
informasi apa pun. Tiga bug, semuanya kecil, semuanya bisa diverifikasi.

## 1. Normalisasi yang tidak pernah membagi

```js
const normalizedData = dataTensor
  .sub(minValues)
  .div((maxValues.sub(minValues) === 0 || 1));
```

`maxValues.sub(minValues)` mengembalikan `tf.Tensor`. Membandingkan tensor dengan `0` pakai `===`
selalu `false` di JavaScript. Jadi `(false || 1)` bernilai `1`, dan pembagiannya selalu `.div(1)`.

Efeknya: data hanya digeser (`x - min`), tidak pernah diskalakan ke [0, 1]. Padahal layer output
model pakai `sigmoid` yang hanya bisa keluar di [0, 1]. Input dan output hidup di skala berbeda,
jadi error rekonstruksi didominasi perbedaan skala - bukan pola transaksi.

Satu karakter (`0` vs `0.0`), dua tahun tanpa ketahuan.

## 2. Threshold persentil 50

Handler-nya memanggil `detectAnomalies(..., 50)`. Artinya: tandai 50% baris teratas sebagai
anomali. Bukan "anomali kalau errornya di atas X", tapi "anomali kalau errornya di atas median
batch ini".

Batch 20 transaksi? 10 ditandai. Batch 5? 2 ditandai. Angkanya selalu terlihat masuk akal, dan
selalu tidak berarti.

Yang membuat ini jahat: secara visual hasilnya meyakinkan. Grafik dengan titik-titik merah di
separuh data terlihat seperti model yang bekerja, bukan seperti threshold yang rusak.

## 3. Model dilatih ulang tiap request

Setiap panggilan `/dashboard`: latih 100 epoch, simpan model, upload dua file ke cloud storage,
hapus folder sementara. Model-nya objek global yang dipakai semua user.

Ini bukan cuma lambat - ini balapan. Dua user membuka dashboard bersamaan akan saling menimpa
model satu sama lain. Dan karena setiap request melatih ulang, "model yang dipakai" berubah
sebelum kamu selesai membaca halaman.

## Kenapa ketiganya lolos

Karena tidak ada yang mengukur. Tidak ada test, tidak ada metrik, tidak ada baseline. Dashboard-nya
tidak pernah error, jadi terlihat benar.

Pola yang sama muncul di banyak proyek mahasiswa dan prototipe: yang diuji adalah "apakah
aplikasinya jalan", bukan "apakah angkanya benar". Dua pertanyaan itu sangat berbeda.

## Yang saya ubah di versi baru

- Span dihitung dengan `np.where(span == 0, 1.0, span)`, dan ada test khusus untuk kolom yang
  variansnya nol.
- Threshold dikalibrasi sekali di holdout bersih pada FPR 1%, disimpan di artefak
  (`threshold.json`, `zscore.json`), tidak pernah dihitung per batch.
- Satu model per versi, tidak ada retraining per request.
- CI menjalankan ruff + pytest di setiap push, dan pre-commit hook memblokir commit merah.

## Checklist audit untuk dashboard ML

Kalau kamu mewarisi proyek seperti ini, empat pertanyaan ini cukup untuk menemukan masalah terbesar:

1. **Apakah normalisasi benar-benar membagi?** Cetak min/max data setelah scaling. Kalau rentangnya
   bukan [0, 1] (atau mean 0 / std 1), ada yang salah.
2. **Apakah threshold bergantung pada batch?** Cari `percentile`, `quantile`, atau `median` di jalur
   inferensi. Kalau ada, nilainya harus dari data kalibrasi, bukan dari data yang sedang dinilai.
3. **Apakah model berubah saat runtime?** Cari `fit`, `train`, atau `save` di handler HTTP. Kalau
   ada, inferensi dan training bercampur.
4. **Apakah ada satu angka pun yang diverifikasi?** Kalau tidak ada test atau metrik, jawabannya
   tidak.

Yang paling murah dari semua perbaikan itu adalah test. Yang paling mahal adalah dua tahun tidak
tahu dashboard-nya bohong.

Repo: `github.com/drafi19/selena-insights`
