# Proyek Feature Engineering & Penanganan Imbalance Data: Implementasi ML Workflow

Repositori ini memuat implementasi praktis dari *Machine Learning Workflow* pada tahap krusial, yaitu **Feature Engineering (Rekayasa Fitur)**, **Feature Selection (Seleksi Fitur)**, dan penanganan ketidakseimbangan data menggunakan teknik **SMOTE**. Proyek ini disimulasikan menggunakan dataset klasifikasi buatan untuk menunjukkan *best practice* dalam mempersiapkan data sebelum masuk ke tahap pemodelan intensif.

---

## Detail Dataset Simulasi
Dataset dibuat secara sintetis menggunakan fungsi `make_classification()` sebanyak 1.000 sampel dengan 15 fitur acak. Untuk mencerminkan kondisi dunia nyata, ditambahkan fitur kategorikal artifisial (`Fitur_12` dan `Fitur_13`) serta skenario **ketidakseimbangan kelas target** dengan proporsi bobot kelas sebesar 90% berbanding 10%.

---

## Tahapan Kerja & Rekayasa Data

### Langkah 1: Seleksi Fitur (Feature Selection) - Embedded Methods
Sebelum melakukan rekayasa lebih lanjut, dilakukan penyaringan fitur untuk memangkas dimensi data yang tidak informatif:
* Menggunakan algoritma **Random Forest Classifier** untuk melatih data numerik dasar secara cepat dan mengekstrak nilai tingkat kepentingan fitur (`feature_importances_`).
* Menetapkan ambang batas (*threshold*) sebesar 5% (0.05). Fitur dengan nilai kontribusi di bawah ambang batas ini akan dieliminasi, menyisakan fitur-fitur yang paling berpengaruh bagi model (`X_important`).

### Langkah 2: Encoding Variabel Kategorikal
Fitur simulasi yang bertipe objek/kategori (`Fitur_12` dan `Fitur_13`) ditransformasikan ke dalam format angka agar dapat dipahami oleh algoritma matematika menggunakan **LabelEncoder** dari library Scikit-Learn.

### Langkah 3: Data Cleaning (Pembersihan Outliers)
Melakukan iterasi pada setiap kolom numerik yang telah terpilih dari tahap seleksi fitur untuk membersihkan nilai ekstrem:
* Menggunakan metode **Interquartile Range (IQR)** dengan menghitung kuartil bawah ($Q1$) dan kuartil atas ($Q3$).
* Menghapus baris sampel data yang teridentifikasi sebagai *outlier* (di luar batas bawah $Q1 - 1.5 \times IQR$ atau batas atas $Q3 + 1.5 \times IQR$) secara sinkron, baik pada matriks fitur ($X$) maupun variabel target ($y$).

### Langkah 4: Penanganan Data Tidak Seimbang (Oversampling via SMOTE)
Ketidakseimbangan kelas target (di mana satu kelas mendominasi secara ekstrem) dapat menyebabkan model bias dan cenderung hanya memprediksi kelas mayoritas. 
* Proyek ini menerapkan **SMOTE (Synthetic Minority Over-sampling Technique)** dari library `imblearn`.
* SMOTE bekerja dengan cara membuat sampel baru secara sintetis pada kelas minoritas berdasarkan kedekatan jarak tetangga (*k-neighbors*), bukan sekadar menduplikasi data yang sudah ada. Hasil akhirnya membuat distribusi kelas target menjadi seimbang (50:50).

### Langkah 5: Penskalaan Fitur (Feature Scaling)
* **StandardScaler:** Tahapan akhir rekayasa data ditutup dengan melakukan standardisasi nilai pada fitur-fitur penting yang telah melalui proses *oversampling*. Langkah ini memastikan distribusi data memiliki rata-rata (*mean*) nol dan varians satu, siap untuk digunakan oleh model klasifikasi tingkat lanjut.

---

## 🛠️ Library Python Utama yang Digunakan
* `sklearn.datasets` (`make_classification`) - Membuat dataset klasifikasi buatan secara acak untuk keperluan eksperimen.
* `sklearn.ensemble` (`RandomForestClassifier`) - Mengekstrak nilai kepentingan fitur untuk metode seleksi *embedded*.
* `sklearn.preprocessing` (`LabelEncoder`, `StandardScaler`) - Mengubah variabel kategori menjadi angka dan menyamakan skala distribusi data numerik.
* `imblearn.over_sampling` (`SMOTE`) - Menyeimbangkan proporsi kelas minoritas secara sintetis pada dataset target.
* `collections` (`Counter`) - Menghitung jumlah persebaran distribusi kelas sebelum dan sesudah proses manipulasi SMOTE.
