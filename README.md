# RF-Kaza-Siddeti-Tahmin-Sistemii

# 🚦 Trafik Kazası Şiddeti Analizi ve Tahmini (RTA Dataset)

Bu proje, trafik kazalarına ait **RTA Dataset** veri seti üzerinde keşifsel veri analizi (EDA), veri temizleme, kodlama (encoding) ve **Random Forest** ile kaza şiddeti tahmini yapan bir Jupyter/Colab not defteridir.

**Dosya:** `TRAFFİC_DATA_SET.ipynb`

---

## 🎯 Amaç

Sürücü, araç, yol, hava ve ışık koşullarına ait bilgilerden yola çıkarak bir kazanın şiddetini (`Accident_severity`) tahmin etmek:

| Sınıf | Açıklama |
|---|---|
| `0` – Slight Injury | Hafif yaralanma |
| `1` – Serious Injury | Ciddi yaralanma |
| `2` – Fatal injury | Ölümlü kaza |

---

## 📦 Veri Seti

- **Kaynak dosya:** `RTA Dataset.csv` (not defteriyle aynı klasörde olmalı)
- **Boyut:** 8.998 satır × 32 sütun
- **Öne çıkan sütunlar:** `Time`, `Day_of_week`, `Age_band_of_driver`, `Sex_of_driver`, `Driving_experience`, `Type_of_vehicle`, `Road_surface_conditions`, `Light_conditions`, `Weather_conditions`, `Type_of_collision`, `Number_of_casualties`, `Cause_of_accident`, `Accident_severity`

En çok eksik veri içeren sütunlar: `Defect_of_vehicle` (3288), `Service_year_of_vehicle` (2957), `Work_of_casuality` (2352), `Fitness_of_casuality` (1943).

---

## 🛠️ Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Kullanılan kütüphaneler: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (`LabelEncoder`, `RandomForestClassifier`, `train_test_split`, `accuracy_score`, `classification_report`).

---

## ▶️ Çalıştırma

1. `RTA Dataset.csv` dosyasını not defteriyle aynı dizine (veya Colab oturumuna) yükleyin.
2. `TRAFFİC_DATA_SET.ipynb` dosyasını Jupyter veya Google Colab'da açın.
3. Hücreleri yukarıdan aşağıya sırayla çalıştırın.

---

## 🔍 Not Defterinin Akışı

### 1. Veri Yükleme ve İlk İnceleme
- `pd.read_csv` ile veri okunur, `head()` ve `isnull().sum()` ile genel yapı ve eksikler incelenir.
- `Time` sütunundan saat bilgisi çıkarılarak **`Hour_1_to_24`** adlı yeni bir özellik üretilir.

### 2. Keşifsel Veri Analizi (EDA)
| Grafik | İncelenen İlişki |
|---|---|
| Boxplot | Haftanın günlerine göre kaza saatlerinin dağılımı |
| Countplot | Sürücü yaş grubu ve direksiyon tecrübesi ↔ kaza şiddeti |
| Countplot (3'lü) | Yol yüzeyi, aydınlatma ve hava durumu ↔ kaza şiddeti |
| Barplot | Kaza nedenine göre ortalama yaralı sayısı |
| Yığılmış (stacked) bar | Kaza nedenlerine göre ölümlü/ciddi/hafif oranları (%) |
| Yığılmış bar | Araç sahibi ↔ sürücünün araçla bağı |

### 3. Veri Temizleme
- **Silinen sütunlar:** `Educational_level`, `Time`, `Work_of_casuality`, `Service_year_of_vehicle`, `Defect_of_vehicle` (çok eksik veya ilgisiz).
- **`Casualty_severity`** sütunu, hedef değişkenle ilişkili olduğundan **veri sızıntısını (data leakage)** önlemek için çıkarılır.
- Eksik değerler kategoriye göre `Unknown` veya `Other` ile doldurulur; `Fitness_of_casuality` içindeki `NormalNormal` hatası `Normal` olarak düzeltilir.
- Hedef değişkeni boş olan satır silinir.

### 4. Kodlama (Encoding)
- **Sıralı (ordinal) sütunlar** sözlüklerle (`map`) manuel olarak sayıya çevrilir: gün, yaş grubu, tecrübe, ışık koşulu, cinsiyet, kaza şiddeti.
- **Sırasız (nominal) sütunlar** `LabelEncoder` ile sayısallaştırılır (araç tipi, yol hizası, kavşak tipi, kaza nedeni vb.).

### 5. Modelleme
- Veri **%80 eğitim / %20 test** olarak ayrılır (`random_state=42`).
- `RandomForestClassifier(n_estimators=100, class_weight='balanced')` eğitilir.
- Son olarak modelin **en önemli 15 özelliği** (feature importance) grafiğe dökülür.

---

## 📊 Sonuçlar

| Metrik | Değer |
|---|---|
| Genel doğruluk (Accuracy) | **%84,50** |
| Hafif yaralanma (0) – recall | 1,00 |
| Ciddi yaralanma (1) – recall | 0,00 |
| Ölümlü kaza (2) – recall | 0,00 |

> ⚠️ **Önemli:** Yüksek doğruluk yanıltıcıdır. Veri seti ciddi şekilde dengesizdir (test setinde 1521 hafif / 262 ciddi / 17 ölümlü) ve model tüm kazaları "hafif" olarak tahmin etmektedir. Asıl önemli olan ciddi ve ölümlü kazalar şu an hiç yakalanamıyor.

---

## 🐞 Bilinen Sorunlar

- **Çift kodlama hatası:** `Day_of_week`, `Age_band_of_driver` ve `Sex_of_driver` sütunları önce bir kez sayıya çevriliyor, ardından ikinci `map` adımında yeniden metin anahtarlarıyla eşleştirilmeye çalışılıyor. Eşleşme olmadığı için değerler `NaN` olup `0` ile dolduruluyor; yani bu üç özellik **tamamen sabit (0)** hale geliyor ve modele bilgi taşımıyor.
- `cinsiyet_haritasi` içinde `'Unkown'` yazım hatası var (`'Unknown'` olmalı).
- `Driving_experience` sütununda hem `'Unknown'` hem `'unknown'` değerleri bulunuyor; tecrübe haritasında yalnızca küçük harfli olan eşleşiyor.
- `Area_accident_occured` içinde baş/son boşluklu ve birleşik (`Rural village areasOffice areas`) kategoriler temizlenmemiş.
- Birkaç sütunda (`Road_surface_conditions`, `Weather_conditions` vb.) 1'er eksik değer kalıyor.

---

## 🚀 Geliştirme Önerileri

- Kodlama adımlarını tek seferde ve ham veri üzerinden yapmak; metin değerlerini `str.strip()` / `str.lower()` ile normalize etmek.
- Sınıf dengesizliği için **SMOTE**, alt/üst örnekleme veya eşik ayarlaması denemek.
- Değerlendirmede accuracy yerine **macro F1**, **recall** ve **karışıklık matrisi** kullanmak.
- Nominal sütunlarda `LabelEncoder` yerine **One-Hot Encoding** denemek.
- XGBoost / LightGBM gibi modellerle karşılaştırma ve `GridSearchCV` ile hiperparametre ayarı yapmak.
- `train_test_split` içinde `stratify=y` kullanmak.
