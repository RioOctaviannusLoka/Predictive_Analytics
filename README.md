# Laporan Proyek Machine Learning - Rio Octaviannus Loka

## Domain Proyek
Pendidikan merupakan fondasi utama dalam pembangunan suatu bangsa. Kualitas pendidikan tidak hanya tergantung pada kurikulum atau fasilitas, namun juga pada pemahaman yang mendalam terhadap faktor-faktor yang memengaruhi performa akademik siswa. Di era digital ini, jumlah data yang berkaitan dengan aktivitas belajar siswa semakin melimpah. Informasi seperti kebiasaan belajar, partisipasi dalam kegiatan ekstrakurikuler, waktu tidur, dan frekuensi latihan dapat menjadi indikator penting untuk memetakan performa siswa secara lebih objektif dan akurat. Dengan memanfaatkan data tersebut secara optimal, institusi pendidikan dapat mengambil keputusan yang lebih strategis dalam meningkatkan mutu pendidikan.

Masalah utama dalam pendidikan saat ini adalah ketidakmampuan dalam memprediksi atau mengantisipasi performa siswa sebelum evaluasi formal dilakukan. Hal ini menyebabkan upaya intervensi yang sering kali terlambat dan tidak efektif. Untuk itu, dibutuhkan sebuah sistem prediktif berbasis data yang mampu memperkirakan performa siswa sejak awal. Dengan menggunakan pendekatan machine learning, dapat dibangun model yang mampu memahami pola dalam data historis siswa dan menghasilkan prediksi yang mendekati kenyataan. Sistem seperti ini dapat digunakan oleh guru, dosen, atau lembaga pendidikan untuk memberikan perhatian lebih awal kepada siswa yang berisiko rendah atau tinggi.

Dalam penelitian yang dilakukan oleh [Alzubaidi et al. (2021)](https://www.mdpi.com/2076-3417/11/1/237), penggunaan algoritma machine learning seperti Random Forest dalam prediksi performa akademik terbukti mampu menghasilkan akurasi yang tinggi. Penelitian ini menegaskan bahwa dengan data yang tepat dan metode yang sesuai, machine learning dapat menjadi alat bantu yang sangat efektif dalam bidang pendidikan.

Dengan menggunakan [dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) yang diperoleh dari kaggle, akan dibangun model prediktif untuk mengestimasi indeks performa (Performance Index) siswa berdasarkan beberapa faktor seperti jam belajar, skor sebelumnya, aktivitas ekstrakurikuler, jam tidur, dan jumlah latihan soal. Dengan memodelkan hubungan antara variabel input dan target menggunakan algoritma machine learning, diharapkan dapat memberikan insight prediktif yang bernilai bagi dunia pendidikan.

## Business Understanding

### Problem Statements
Dari domain proyek yang sudah dijelaskan, kita dapat merumuskan pernyataan masalah sebagai berikut:
- Bagaimana pengaruh jam belajar (Hours Studied) terhadap indeks performa (Performance Index) siswa ?
- Apakah keterlibatan dalam aktivitas ekstrakurikuler (Extracurricular Activities) berhubungan dengan indeks performa (Performance Index) akademik?
- Apakah model terbaik yang dapat digunakan untuk memprediksi indeks performa (Performance Index) siswa?

### Goals
Berdasarkan pernyataan masalah tersebut, tujuan dari proyek ini adalah:
- Menunjukkan dampak jam belajar (Hours Studied) terhadap indeks performa (Performance Index) siswa
- Menilai pengaruh aktivitas ekstrakurikuler (Extracurricular Activities) terhadap indeks performa (Performance Index) akademik
- Menentukan model machine learning terbaik yang dapat memprediksi nilai indeks performa (Performance Index) dengan akurasi tinggi.

### Solution statements
- Menggunakan dua algoritma regresi, yaitu SVM dan Random Forest Regressor untuk memprediksi indeks performa (Performance Index)
- Melakukan proses Exploratory Data Analysis untuk melihat pengaruh dari tiap fitur terhadap variabel target, khususnya untuk menunjukkan pengaruh jam belajar (Hours Studied) dan aktivitas ekstrakurikuler (Extracurricular Activities) terhadap indeks performa
- Evaluasi model dilakukan menggunakan metrik MSE dan RMSE

## Data Understanding
Dataset yang digunakan pada proyek ini diambil dari platform kaggle dengan sumber sebagai berikut: [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression). Dataset ini didesain untuk mengidentifikasi faktor-faktor yang mempengaruhi performa akademik siswa. Dataset terdiri dari 6 kolom dan 10.000 baris record siswa, dengan setiap baris mengandung informasi terkait beberapa fitur dan sebuah variabel target. Variabel target di dataset ini adalah `Performance Index`.

Informasi terkait kolom-kolom tersebut dapat dilihat pada gambar berikut:
<img src="https://github.com/user-attachments/assets/54c6ab72-b202-4d02-bda1-a74249fe9254" align="center" width=500>
<br>Berdasarkan gambar tersebut, terdapat 1 kolom kategorikal bertipe data object dan 5 kolom bertipe data numerik. Selain itu, bisa dilihat juga bahwa setiap kolom yang ada pada dataset tersebut tidak terdapat nilai null/kosong karena jumlah Non-Null Count dari setiap kolom jumlahnya sama, yaitu 10000 baris.

### Variabel-variabel pada Student Performance dataset adalah sebagai berikut:
Variabel                         | Keterangan
---------------------------------|-------------
Hours Studied                    | Jumlah total jam yang dihabiskan untuk belajar oleh setiap siswa.
Previous Scores                  | Nilai yang diperoleh oleh siswa dalam ujian sebelumnya.
Extracurricular Activities       | Apakah siswa berpartisipasi dalam kegiatan ekstrakurikuler (Ya atau Tidak).
Sleep Hours                      | Rata-rata jumlah jam tidur yang dimiliki siswa per hari.
Sample Question Papers Practiced | Jumlah lembar soal latihan yang dikerjakan siswa.
Performance Index                | Sebuah ukuran dari kinerja keseluruhan setiap siswa. Performance Index mewakili kinerja akademik siswa dan telah dibulatkan ke bilangan bulat terdekat. Indeks berkisar dari 10 hingga 100, dengan nilai yang lebih tinggi menunjukkan kinerja yang lebih baik.

### Penanganan Duplikasi Data
Selanjutnya dilakukan pengecekan dan penghapusan data yang duplikat. Diketahui bahwa jumlah data duplikat ada 127 baris, sehingga setelah dilakukan penghapusan duplikasi data dengan kode `df.drop_duplicates(inplace=True)`, jumlah baris pada data sekarang ada 9873 baris.

### Exploratory Data Analysis

#### Univariate Analysis
1. Menampilkan visualisasi data kolom Extracurricular Activities menggunakan pie chart dari library matplotlib.
    <img src="https://github.com/user-attachments/assets/a6cd6a25-bbbf-4e55-9e7a-42f500915e59" align="center" width=300>
<br>Hasil dari grafik pie diatas menunjukkan bahwa perbandingan antara jumlah siswa yang mengikuti dan tidak mengikuti ekstrakulikuler tidak berbeda jauh, yaitu 49,5% untuk yang mengikuti ekstrakulikuler dan 50.5% siswa yang tidak mengikuti ekstrakulikuler.

2. Menampilkan distribusi data dari kolom-kolom numerik, yaitu `Hours Studied`, `Previous Scores`, `Sleep Hours`, `Sample Question Papers Practiced`, dan `Performance Index` dengan menggunakan histogram.
    <img src="https://github.com/user-attachments/assets/a3f1acae-1397-45d6-8406-8bf4bb709efe" align="center" width=800>
<br>Dari grafik diatas, terlihat bahwa distribusi data kolom `Hours Studied`, `Previous Scores`, `Sleep Hours`, `Sample Question Papers Practiced`, dan `Performance Index` cukup normal.

#### Multivariate Analysis
1. Analisis Pengaruh Extracurricular Activities terhadap Performance Index
    <img src="https://github.com/user-attachments/assets/795a0990-5bd4-429e-bd2b-2637b9ba8ff3" align="center" width=500>
<br>Dari grafik diatas, terlihat bahwa siswa yang mengikuti kegiatan ekstrakurikuler (Yes) memiliki rata-rata Performance Index sedikit lebih tinggi dibandingkan dengan siswa yang tidak mengikuti (No). Namun perbedaannya sangatlah kecil sehingga bisa disimpulkan bahwa pengaruh dari mengikuti aktivitas ekstrakulikuler terhadap performance index sangat kecil.

2. Analisis Pengaruh Hours Studied terhadap Performance Index<br>
    <img src="https://github.com/user-attachments/assets/7b64a842-b8e9-404e-94a9-23123104570e" align="center" width=500>
<br>Dari grafik diatas, terlihat bahwa terdapat perbedaan yang cukup besar antara performance index siswa yang belajar hanya 1 jam dengan siswa yang belajar 9 jam. Perbedaan tersebut juga nampak pada setiap jam belajar, dimana performance index siswa juga turut mengalami peningkatan seiring dengan bertambahnya hours studied.

    Sehingga bisa disimpulkan bahwa hours studied memiliki pengaruh yang cukup besar terhadap performance index siswa.

3. Analisis Pengaruh Fitur Numerik selain Hours Studied terhadap Performance Index
    <img src="https://github.com/user-attachments/assets/27a41639-dc10-48a1-8474-cc7b4c6bfde3" align="center" width=800>
    <img src="https://github.com/user-attachments/assets/4f805e51-753f-4b06-b948-e8f4a5f32771" align="center" width=500>
<br>Dari grafik diatas, terlihat bahwa performance index mengalami perubahan yang sangat kecil, jadi pengaruh sleep hours dan sample question papers practiced terhadap Performance Index sangat kecil. Selain itu, terlihat juga bahwa performance index siswa juga turut meningkat seiring dengan meningkatnya previous scores. Ini menunjukkan bahwa pengaruh previous scores terhadap performance index cukup besar.

4. Menggunakan Heatmap untuk melihat korelasi Fitur Numerik<br>
    <img src="https://github.com/user-attachments/assets/9938f4bb-d2b3-450e-a228-876114f61964" align="center" width=500>
<br>Terlihat dari grafik diatas, bahwa:
    1. Hours Studied memiliki korelasi positif yang tidak terlalu kuat terhadap Performance Index
    2. Previous Score memiliki korelasi positif yang sangat kuat terhadap Performance Index
    3. Selain dari kedua fitur Hours Studied dan Previous Score, fitur-fitur lain tidak memiliki korelasi/korelasi yang sangat kecil antara satu sama lain dan terhadap Performance Index

## Data Preparation
Ditahap ini, dilakukan persiapan terhadap data supaya memiliki bentuk yang sesuai untuk digunakan pada model. Ada 2 tahap persiapan data yang dilakukan:
1. Menangani Duplikasi Data
2. Encoding Fitur Kategorikal
3. Pembagian dataset menjadi data train dan data test

### Menangani Duplikasi Data
Pengecekan duplikasi data dilakukan menggunakan kode berikut: `print("Jumlah duplikasi data:", df.duplicated().sum())`. Dari hasil output kode tersebut, diketahui terdapat 127 data duplikat. Kemudian data duplikat tersebut dihapus dengan kode berikut: `df.drop_duplicates(inplace=True)`. Lalu dilakukan pengecekan ukuran dataset setelah penghapusan data duplikat dimana didapati bahwa dataset sekarang berukuran 9873 baris dengan 6 kolom. Penghapusan data duplikat penting dilakukan karena data yang sama muncul lebih dari sekali dapat menyebabkan distorsi dalam analisis dan hasil model machine learning. Duplikat dapat memperkuat bobot informasi tertentu secara tidak wajar, yang berpotensi membuat model bias terhadap pola-pola yang sebenarnya tidak representatif. Dalam konteks regresi, ini dapat mengarah pada overfitting, di mana model terlalu menyesuaikan diri dengan data pelatihan dan gagal melakukan generalisasi pada data baru. Selain itu, data duplikat juga dapat mengganggu akurasi evaluasi, memperpanjang waktu pemrosesan, serta membebani penyimpanan secara tidak perlu. Oleh karena itu, menghapus duplikat adalah langkah penting dalam tahap pembersihan data (data cleaning) untuk memastikan integritas, efisiensi, dan reliabilitas hasil analisis.

### Encoding Fitur Kategorikal
Fitur kategorikal seperti Extracurricular Activities (yang berisi nilai 'Yes' dan 'No') tidak dapat langsung digunakan oleh sebagian besar algoritma machine learning karena model-model ini hanya menerima input numerik. Oleh karena itu, fitur ini perlu di-encode ke dalam format angka agar bisa diproses oleh algoritma. Penerapan encoding ini dilakukan dengan proses mapping terhadap kolom Extracurricular Activities yang merupakan kolom kategorikal. Proses mapping ini dilakukan dengan mengubah value pada kolom tersebut yaitu **Yes** menjadi **1** dan **No** menjadi **0**. 

Kodenya adalah sebagai berikut:
<br>`df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})`

### Pembagian dataset menjadi data train dan data test
Pembagian dataset menjadi data latih (training) dan data uji (testing) bertujuan untuk mengevaluasi performa model secara obyektif. Model machine learning harus dapat mempelajari pola dari data, tetapi juga harus diuji pada data yang belum pernah dilihat sebelumnya untuk mengukur generalisasi. Selain itu, pembagian dataset ini berguna untuk menghindari overfitting dan nenilai kemampuan prediksi model. Dengan demikian, proses ini penting agar model yang dibangun tidak hanya pintar "menghafal" data, tetapi benar-benar mampu mengenali pola dan membuat prediksi yang akurat.

Pertama-tama, kita menyimpan terlebih dahulu fitur-fitur ke variabel **X** dan variabel target ke variabel **y**. Kemudian, pembagian dataset menjadi data training dan data testing dilakukan dengan menggunakan train_test_split dengan ukuran data training sebesar 80% dan data testing sebesar 20%. 

Kodenya adalah sebagai berikut:
```python
# Memisahkan fitur dan target
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Modeling
Pada tahap ini, pembangunan model dilakukan dengan menggunakan 2 algoritma berbeda yaitu SVM dan Random Forest. 

### Support Vector Machine(SVM)
Tahapan proses pemodelan SVM:
1. Pertama-tama dibuat sebuah objek SVR yang merupakan model Support Vector Regression. Karena pada pembuatan SVR tersebut tidak diberikan parameter apapun, maka SVR akan menggunakan parameter default:
- `kernel='rbf'`, digunakan untuk mengatur kernel berupa Radial Basis Function digunakan untuk menangkap hubungan non-linear.
- `C=1.0`, merupakan parameter regulasi. Semakin besar nilainya, semakin sedikit toleransi terhadap kesalahan.
- `epsilon=0.1`, merupakan margin toleransi di mana tidak dikenakan penalti jika prediksi dalam jarak 0.1 dari nilai sebenarnya.
- `gamma='scale'`, merupakan parameter untuk kernel RBF.

2. Setelah itu dilakukan pelatihan model SVM menggunakan data latih.

3. Kemudian, model digunakan untuk memprediksi variabel target dari data latih dan data test, lalu mengukur nilai mse dan rmse pada kedua data tersebut.

Support Vector Machine memiliki kelebihan berupa cocok untuk dataset kecil dan kompleks, dimana SVM dapat bekerja baik pada dataset dengan jumlah sampel terbatas karena fokus pada margin dan support vectors. Selain itu, SVM mampu menangani non-linearitas dengan penggunaan kernel (misalnya RBF), SVM bisa memodelkan hubungan non-linear antar fitur dan target.

Namun, SVM juga memiliki kekurangan, yaitu sensitif terhadap skala fitur. SVM memerlukan data yang sudah dinormalisasi karena algoritmanya sangat tergantung pada jarak antar data.

### Random Forest Regressor
Tahapan proses pemodelan Random Forest Regressor:
1. Pertama-tama, dibuat objek rf dari RandomForestRegressor, yang merupakan model ensemble berbasis decision tree untuk regresi dengan parameter yang digunakan sebagai berikut:
- `n_estimators=100`, merupakan jumlah pohon keputusan (trees) dalam hutan. Semakin banyak pohon, biasanya model lebih stabil tapi lebih lambat.
- `max_depth=64`, menunjukkan maksimum kedalaman pohon. Mencegah pohon terlalu dalam agar tidak overfitting.
- `random_state=55`, digunakan untuk menjamin hasil reproducible (hasil sama setiap eksekusi).
- `n_jobs=-1`, berfungsi untuk menggunakan seluruh core CPU yang tersedia untuk paralelisasi pelatihan, mempercepat proses training.

2. Setelah itu dilakukan pelatihan model Random Forest menggunakan data latih.

3. Kemudian, model digunakan untuk memprediksi variabel target dari data latih dan data test, lalu mengukur nilai mse dan rmse pada kedua data tersebut.

Random Forest memiliki kelebihan berupa tahan terhadap outlier dan noise. Random Forest adalah algoritma ensemble berbasis pohon keputusan yang mampu menangani data yang tidak sempurna.Selain itu, Random Forest dapat menangani fitur numerik dan kategorikal tanpa praproses khusus seperti normalisasi atau encoding canggih. Random Forest juga memiliki bias rendah dan stabil karena merupakan gabungan banyak pohon, sehingga hasil prediksi umumnya stabil dan akurat.

Selain kelebihan tersebut, Random Forest juga memiliki kelemahan yaitu cenderung overfitting pada data latih yang disebabkan karena model sangat fleksibel, sehingga model bisa terlalu menyesuaikan diri pada data training. Selain itu, kompleksitas model Random Forest juga besar, dimana ukuran model bisa sangat besar, hingga memakan memori dan waktu saat pelatihan dan prediksi.

### Model Terbaik
Hasil evaluasi nilai metriks MSE dan RMSE dari model Random Forest dan SVM:<br>
Model |	MSE (Train)	| MSE (Test)	|RMSE (Train)	|RMSE (Test)
------|-----|-----|----|----
SVR	|5.425	|5.639	|2.329	|2.375
Random Forest	|0.930	|5.599	|0.964	|2.366

Meskipun Random Forest menunjukkan indikasi overfitting (karena MSE dan RMSE di train jauh lebih kecil dibanding test), performanya pada data uji tetap lebih baik dibanding SVR. RMSE Random Forest pada data uji adalah 2.366, sedikit lebih rendah dibanding SVR (2.375), yang menunjukkan prediksinya lebih akurat terhadap data baru.

Selain itu, keunggulan Random Forest yang mampu menangani berbagai jenis fitur tanpa praproses rumit, serta kemampuannya mengatasi outlier dan noise, menjadikannya solusi praktis dan kuat untuk diterapkan pada dunia nyata.

## Evaluation
Proyek ini menggunakan Mean Square Error (MSE) dan Root Mean Square Error (RMSE) sebagai metriks evaluasi yang digunakan untuk menilai performa dari setiap model yang dibuat. 

1. Mean Square Error (MSE)<br>
Mean Squared Error (MSE) adalah metrik evaluasi yang digunakan untuk mengukur rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi dalam masalah regresi. MSE dihitung dengan menjumlahkan kuadrat dari setiap selisih antara nilai aktual dan prediksi, lalu membaginya dengan jumlah total data. Karena selisihnya dikuadratkan, MSE memberikan penalti yang lebih besar terhadap prediksi yang jauh meleset dari nilai sebenarnya, sehingga metrik ini sangat sensitif terhadap outlier.

    Formula: <br>
    <img src="https://github.com/user-attachments/assets/da8154e4-fe1c-4bad-b0af-3b18d167b906" align="center" width=300>
<br>Keterangan:
   - N = jumlah dataset
   - yi = nilai sebenarnya
   - y_pred = nilai prediksi
   
   <br>Cara kerja:
MSE menghitung rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Dengan mengkuadratkan selisihnya, MSE memberi penalti lebih besar terhadap kesalahan besar (outlier). Nilai MSE yang lebih kecil menunjukkan model memiliki prediksi yang lebih mendekati nilai aktual.

2. Root Mean Square Error (RMSE)<br>
Root Mean Squared Error (RMSE) adalah akar kuadrat dari MSE dan digunakan untuk mengukur seberapa besar rata-rata kesalahan prediksi model dalam satuan yang sama dengan data target aslinya. RMSE mempermudah interpretasi karena memiliki skala yang sama dengan nilai yang diprediksi, sehingga memudahkan dalam menilai apakah kesalahan model masih dapat diterima secara praktis. Semakin kecil nilai RMSE, semakin baik performa model dalam memprediksi data.

    Formula: <br>
    <img src="https://github.com/user-attachments/assets/15ba9ce6-94a4-4add-932c-cedca9f85bf0" align="center" width=300>
    <br>Keterangan:
   - n = jumlah dataset
   - yi = nilai sebenarnya
   - yp = nilai prediksi

   <br>Cara Kerja:
RMSE merupakan akar dari MSE. Dengan demikian, RMSE memiliki satuan yang sama seperti target (nilai aktual), sehingga lebih mudah diinterpretasikan secara langsung dalam konteks domain. Sama seperti MSE, nilai RMSE yang lebih rendah menunjukkan model yang lebih akurat.

### Hasil Evaluasi MSE Dan RMSE pada Proyek
Model |	MSE (Train)	| MSE (Test)	|RMSE (Train)	|RMSE (Test)
------|-----|-----|----|----
SVR	|5.425	|5.639	|2.329	|2.375
Random Forest	|0.930	|5.599	|0.964	|2.366

Dapat dilihat bahwa baik di data train maupun data test, RF memiliki performa yang lebih bagus dengan nilai RMSE yang lebih kecil dibandingkan SVM. Sehingga bisa kita simpulkan bahwa Random Forest merupakan model terbaik karena nilai metriks evaluasinya yaitu MSE dan RMSE yang paling kecil nilainya.

Visualisasi hasil evaluasi metriks MSE dan RMSE:<br>
<img src="https://github.com/user-attachments/assets/cb017ac0-ecd0-4e6d-94f9-19fe6c749b5f" align="center" width=800>


### Kesimpulan
Berdasarkan proses dan analisis yang telah dilakukan, diperoleh beberapa kesimpulan penting:
1. Pengaruh jam belajar (Hours Studied) terhadap  indeks performa (Performance Index) siswa dianalisis melalui proses Exploratory Data Analysis (EDA) dan juga terlihat signifikan pada hasil prediksi model. Fitur Hours Studied memiliki korelasi positif yang kuat terhadap indeks performa (Performance Index) . Selain itu, nilai sebelumnya (Previous Scores) juga memiliki pengaruh yang signifikan terhadap P indeks performa (Performance Index) 
2. Hubungan aktivitas ekstrakurikuler terhadap performa akademik juga dianalisis dalam EDA. Ditemukan bahwa fitur Extracurricular Activities tidak memiliki korelasi signifikan terhadap Performance Index, yang mengindikasikan pengaruhnya sangat kecil.
3. Model terbaik untuk memprediksi indeks performa siswa ditentukan melalui perbandingan metrik evaluasi (MSE dan RMSE). Hasil menunjukkan bahwa Random Forest Regressor adalah model dengan performa terbaik dibandingkan dengan Support Vector Machine.
4. Proyek berhasil mencapai setiap goals yang telah ditetapkan dalam proyek ini dengan ketiga poin diatas yang berhasil menjawab setiap problem statement sekaligus goal dalam proyek ini.
5. Semua solution statement yang direncanakan berdampak langsung dan mendukung pencapaian goals proyek, dimana penggunaan dua model regresi (SVM dan Random Forest) memberikan wawasan komparatif dan memungkinkan pemilihan model terbaik secara objektif berdasarkan evaluasi kuantitatif. Selain itu, EDA yang dilakukan sangat penting dalam memahami struktur data dan hubungan antar variabel. Visualisasi seperti korelasi matriks, scatter plot, dan plot distribusi berhasil menunjukkan hubungan antar fitur dan target, terutama untuk Hours Studied dan Extracurricular Activities. Kemudian, evaluasi menggunakan MSE dan RMSE memberikan metrik yang kuat untuk membandingkan performa kedua model, sehingga hasilnya tidak hanya subjektif tetapi juga objektif dan terukur.