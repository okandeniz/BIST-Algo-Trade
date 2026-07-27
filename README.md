# BIST100 Algorithmic Trading Robot

BIST100 hisseleri üzerinde çalışan; kural tabanlı sinyal üretimi, portföy seviyesinde backtest, BIST100 karşılaştırması, makine öğrenmesi destekli sinyal filtreleme, walk-forward doğrulama, FastAPI backend ve Streamlit arayüzünü bir araya getiren uçtan uca algoritmik işlem projesidir.

> **Önemli:** Bu proje eğitim, araştırma ve paper-trading amacıyla geliştirilmiştir. Yatırım tavsiyesi değildir ve otomatik emir iletmez.

---

## İçindekiler

- [Projenin amacı](#projenin-amacı)
- [Temel özellikler](#temel-özellikler)
- [Stratejinin çalışma mantığı](#stratejinin-çalışma-mantığı)
- [Makine öğrenmesi yaklaşımı](#makine-öğrenmesi-yaklaşımı)
- [Backtest ve doğrulama metodolojisi](#backtest-ve-doğrulama-metodolojisi)
- [Mevcut sonuçların özeti](#mevcut-sonuçların-özeti)
- [Proje yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Araştırma notebooklarının çalıştırılması](#araştırma-notebooklarının-çalıştırılması)
- [FastAPI ve Streamlit uygulaması](#fastapi-ve-streamlit-uygulaması)
- [Paper-trading kullanım akışı](#paper-trading-kullanım-akışı)
- [API uçları](#api-uçları)
- [Veri yönetimi](#veri-yönetimi)
- [Güvenlik ve gizlilik](#güvenlik-ve-gizlilik)
- [Bilinen sınırlamalar](#bilinen-sınırlamalar)
- [Gelecek geliştirmeler](#gelecek-geliştirmeler)
- [Sorumluluk reddi](#sorumluluk-reddi)

---

## Projenin amacı

Projenin amacı yalnızca teknik göstergelerden alış sinyali üretmek değil; bir işlem sisteminin araştırmadan canlı kullanıma kadar ihtiyaç duyduğu temel bileşenleri tek bir yapı altında toplamaktır.

Proje şu sorulara cevap vermeyi hedefler:

1. Kural tabanlı strateji BIST100 endeksini geçebiliyor mu?
2. Strateji farklı dönemlerde ve farklı parametrelerde dayanıklı mı?
3. Makine öğrenmesi, Robot tarafından üretilen sinyallerin kalitesini artırabiliyor mu?
4. Tahminler veri sızıntısı olmadan walk-forward yöntemle doğrulanabiliyor mu?
5. Kullanıcı günlük sinyalleri, açık pozisyonları, nakdi ve kâr-zararı tek arayüzden takip edebiliyor mu?

---

## Temel özellikler

### Araştırma ve backtest

- BIST100 hisse verilerinin Yahoo Finance üzerinden alınması
- Veri kalitesi kontrolleri
- Sıfır hacimli veya geçersiz barların temizlenmesi
- Olası split boşluklarının kontrollü onarımı ve loglanması
- Teknik gösterge ve momentum özelliklerinin oluşturulması
- BIST100 piyasa filtresi
- Çoklu hisse ve çoklu pozisyon portföy backtesti
- Komisyon ve kayma maliyetleri
- ATR tabanlı pozisyon büyüklüğü
- Stop loss, trailing stop ve fiyat tabanlı çıkışlar
- Parametre, çıkış ve portföy risk analizleri
- Development, Validation ve Audit dönemleri
- BIST100 Gross ve BIST100 Net benchmark karşılaştırması

### Makine öğrenmesi

- Robot sinyalleri için meta-labeling
- Purged expanding-window cross-validation
- Logistic Regression, Random Forest ve gradient boosting karşılaştırmaları
- Büyük kazanan olasılığına dayalı ML Challenger
- Sabit ve önceden kilitlenmiş olasılık eşiği
- Aylık expanding walk-forward eğitim
- Embargo ve tamamlanmış işlem filtresi
- Çok görevli ML V2 deneyi
- Başarısız modelin Validation aşamasında objektif olarak reddedilmesi

### Uygulama

- FastAPI backend
- Streamlit dashboard
- Baseline Robot ve ML Challenger için ayrı paper-trading portföyleri
- Kullanıcı tanımlı başlangıç sermayesi
- Alış ve satış kayıtları
- Otomatik binde 2 alış komisyonu
- Stop loss kaydı
- Açık pozisyon ve işlem geçmişi
- Gerçekleşen ve gerçekleşmemiş kâr-zarar
- Güncel fiyat yenileme
- Stop loss ve strateji çıkış tavsiyesi
- Portföy sıfırlama
- SQLite ile kalıcı yerel kayıt

---

## Stratejinin çalışma mantığı

Robot, günlük fiyat verileri üzerinden her hisse için bir trend ve momentum skoru üretir.

### Skor bileşenleri

| Koşul | Puan |
|---|---:|
| BIST100 piyasa rejimi pozitif | +2 |
| Fiyat EMA200 üzerinde | +2 |
| EMA50, EMA200 üzerinde | +2 |
| Fiyat EMA20 üzerinde | +1 |
| RSI 50 üzerinde | +1 |
| MACD histogram pozitif | +1 |
| ADX 20 üzerinde | +1 |
| Hacim, 20 günlük ortalamanın 1,3 katından yüksek | +2 |
| Fiyat önceki 20 günlük zirveyi aşmış | +2 |

Temel sinyal sınıfları:

```text
Score >= 11  → AL
Score >= 8   → İZLE
Diğer        → ALMA
```

Aynı gün oluşan adaylar şu alanlara göre sıralanır:

```text
Score → RET_126 → RET_63 → ADX
```

### Portföy kuralları

Mevcut final strateji ayarları:

```text
Başlangıç sermayesi         : Kullanıcı tarafından belirlenir
Maksimum açık pozisyon      : 8
İşlem başına risk           : %0,75
Komisyon                    : Binde 2
Kayma                       : Binde 2
İlk stop                    : 1,5 ATR
Trailing stop               : 2,5 ATR
Trailing aktivasyonu        : %6 kâr
Fiyat tabanlı çıkış         : Önceki LOW10 seviyesi
```

Sinyaller gün sonu verisiyle oluşur. Alış ve kapanış bazlı çıkışlar bir sonraki uygun açılış fiyatından uygulanır. Intraday stop veya gap durumları ayrıca ele alınır.

---

## Makine öğrenmesi yaklaşımı

### Baseline Robot

Ana sistemdir. ML kullanılmadan Robot kurallarıyla çalışır.

### ML Challenger

Robot tarafından üretilen `AL` adaylarını tamamen değiştirmez; adayların büyük kazanan olma ihtimalini tahmin ederek ek bir filtre uygular.

Kullanılan deneysel yapı:

```text
Hedef       : İşlemin en az 2R kazandırması
Model       : Logistic Regression
Rol         : Meta-label / sinyal kalite filtresi
Durum       : Experimental Challenger
```

ML Challenger ana stratejinin yerine geçirilmemiştir. Baseline Robot ile paralel paper-trading ortamında takip edilir.

### Walk-forward doğrulama

ML karşılaştırması bilinçli olarak 2025 yılında başlatılır. Bunun nedeni model, hedef ve filtre kararlarının daha önceki Development ve Validation dönemlerinde verilmiş olmasıdır.

Her ayın başında:

1. Sadece geçmişte oluşmuş sinyaller kullanılır.
2. Çıkışı tamamlanmamış işlemler eğitimden çıkarılır.
3. Beş günlük embargo uygulanır.
4. Model geçmiş verilerle yeniden eğitilir.
5. Ay içindeki Robot sinyalleri skorlanır.
6. Kilitlenmiş eşik değiştirilmeden uygulanır.

Bu yapı, gelecek işlem sonuçlarının geçmiş tahminlerde kullanılmasını engeller.

### ML V2 sonucu

Büyük kazanan olasılığı, beklenen R Multiple ve stop riskini birleştiren çok görevli sıralama modeli ayrıca test edilmiştir. Model Validation döneminde Baseline Robot’un getirisini geçemediği için reddedilmiş ve 2025+ Audit döneminde yeni bir seçim yapılmamıştır.

Bu sonuç, daha karmaşık bir modelin her zaman daha iyi portföy performansı üretmediğini gösterir.

---

## Backtest ve doğrulama metodolojisi

Araştırma üç ana döneme ayrılır:

```text
Development : 2018–2022
Validation  : 2023–2024
Audit       : 2025+
```

### Development

- Model ve strateji geliştirme
- Parametre adaylarının incelenmesi
- ML model türlerinin karşılaştırılması

### Validation

- Parametre ve ML kararlarının seçilmesi
- Robustness kurallarının uygulanması
- Baseline’a karşı kabul veya ret kararı

### Audit / walk-forward

- Önceden kilitlenen kararların değerlendirilmesi
- Audit sonucuna bakılarak yeni parametre seçilmemesi
- Aylık expanding-window yeniden eğitim

---

## Mevcut sonuçların özeti

Aşağıdaki değerler mevcut yerel çalıştırmalardan elde edilmiştir. Veri kaynağı, hisse evreni, işlem maliyetleri ve dönem seçimi değiştiğinde sonuçlar da değişebilir.

### Baseline Robot — tam dönem

| Metrik | Sonuç |
|---|---:|
| Başlangıç değeri | 500.000 TL |
| Son değer | 13.315.423,78 TL |
| CAGR | %52,63 |
| Maksimum drawdown | -%25,52 |
| Sharpe | 2,17 |
| Calmar | 2,06 |
| Profit Factor | 2,17 |
| İşlem sayısı | 735 |

Aynı dönemde BIST100 Gross benchmark yaklaşık 7,22 milyon TL son değere ve yaklaşık %41,06 CAGR değerine ulaşmıştır.

> Bu sonuçlar kesin gelecek performansı göstermez. Güncel BIST100 bileşenlerinin geçmişe uygulanması survivorship bias oluşturabilir.

### ML V2 Validation kararı

En iyi çok görevli ML V2 adayı risk ayarlı metriklerde bazı iyileşmeler üretmesine rağmen Validation CAGR değerinde Baseline Robot’un gerisinde kalmıştır. Bu nedenle model reddedilmiş ve ana sistem Baseline Robot olarak korunmuştur.

---

## Proje yapısı

```text
BIST-Algo-Trade/
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── dependencies.py
│       ├── database.py
│       ├── repository.py
│       ├── schemas.py
│       ├── routers/
│       └── services/
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── formatters.py
│
├── src/
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── features.py
│   ├── signals.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── presets.py
│   ├── paper_trading.py
│   ├── ml_dataset.py
│   ├── ml_training.py
│   ├── ml_portfolio.py
│   ├── ml_walkforward.py
│   ├── ml_v2.py
│   └── ...
│
├── notebooks/
│   ├── 01_data_collection_robot.ipynb
│   ├── 02_data_quality_robot.ipynb
│   ├── 03_baseline_strategy_robot.ipynb
│   ├── 04_parameter_analysis_robot.ipynb
│   ├── 05_exit_parameter_analysis_robot.ipynb
│   ├── 06_robustness_analysis_robot.ipynb
│   ├── 07_portfolio_risk_analysis_robot.ipynb
│   ├── 08_daily_signal_paper_trading_robot.ipynb
│   ├── 15_walk_forward_ml_comparison_robot.ipynb
│   └── 16_ml_v2_multitask_ranking_robot.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── live/
│   └── manual_adjustments/
│
├── models/
├── results/
├── requirements.txt
├── requirements_app.txt
├── .env.example
└── README.md
```

---

## Kurulum

### 1. Repoyu klonla

```powershell
git clone https://github.com/okandeniz/BIST-Algo-Trade.git
cd BIST-Algo-Trade
```

### 2. Sanal ortam oluştur

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

PowerShell script çalıştırmayı engelliyorsa yalnızca mevcut terminal için:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Kütüphaneleri yükle

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_app.txt
```

### 4. Ortam değişkenlerini oluştur

```powershell
Copy-Item .env.example .env
```

`.env` dosyasını kişisel ayarlarına göre düzenle. `.env` dosyası GitHub’a gönderilmemelidir.

---

## Araştırma notebooklarının çalıştırılması

Notebooklar genel olarak numara sırasıyla çalıştırılmalıdır.

Temel araştırma akışı:

```text
01 → Veri toplama
02 → Veri kalite kontrolü
03 → Baseline strateji
04 → Sinyal parametre analizi
05 → Çıkış parametre analizi
06 → Robustness analizi
07 → Portföy risk analizi
08 → Günlük sinyal ve paper-trading
...
15 → Walk-forward ML karşılaştırması
16 → ML V2 multi-task deneyi
```

Araştırma ve canlı veri dosyaları ayrıdır:

```text
data/processed/  → Backtest ve ML için tam geçmiş veri
data/live/       → Günlük FastAPI sinyal yenileme verisi
```

FastAPI’nin günlük veri yenilemesi `data/processed` dosyalarının üzerine yazmamalıdır.

---

## FastAPI ve Streamlit uygulaması

### FastAPI

```powershell
python -m uvicorn backend.app.main:app `
    --reload `
    --host 127.0.0.1 `
    --port 8000
```

Adresler:

```text
API health : http://127.0.0.1:8000/health
Swagger    : http://127.0.0.1:8000/docs
```

### Streamlit

İkinci terminalde:

```powershell
$env:API_URL = "http://127.0.0.1:8000"

python -m streamlit run frontend/app.py `
    --server.address 127.0.0.1 `
    --server.port 8501
```

Dashboard:

```text
http://127.0.0.1:8501
```

---

## Paper-trading kullanım akışı

### 1. Portföyü başlat

Baseline Robot ve ML Challenger portföyleri birbirinden bağımsızdır.

Kullanıcı her portföy için kullanacağı sermayeyi kendisi girer:

```text
Örnek başlangıç sermayesi: 300.000 TL
```

Sermaye girilmeden alış kaydı oluşturulamaz.

### 2. Günlük sinyalleri yenile

Günlük Sinyaller sekmesinde:

```text
Sinyalleri yenile
```

düğmesi kullanılır.

Sistem:

- Güncel piyasa verisini indirir
- Teknik özellikleri hesaplar
- Baseline Robot sinyallerini üretir
- ML Challenger filtresini uygular
- Alış ve satış planlarını kaydeder

### 3. Alış kaydı oluştur

- Ticker günlük sinyal listesinden seçilir
- Kullanıcı gerçekleşen lotu girer
- Kullanıcı gerçekleşen alış fiyatını girer
- Stop seviyesi günlük plandan gelir
- Komisyon otomatik hesaplanır

```text
Komisyon = Lot × Alış fiyatı × 0,002
```

Nakit:

```text
Yeni nakit =
Eski nakit - alış tutarı - komisyon
```

### 4. Son fiyatları güncelle

Açık pozisyonlar bölümünde:

```text
Son fiyatları güncelle
```

düğmesi yalnızca açık pozisyonların son fiyatını yeniler.

Şunlar değişmez:

- Lot
- Ortalama maliyet
- Stop loss
- Alış tarihi
- Nakit
- İşlem geçmişi

Son fiyat güncellendiğinde portföy değeri ve gerçekleşmemiş kâr-zarar tekrar hesaplanır.

### 5. Satış kaydı oluştur

Kullanıcı:

- Satılacak hisseyi
- Lotu
- Gerçekleşen satış fiyatını
- Satış tarihini

girer.

Kısmi ve tam satış desteklenir.

### 6. Portföyü sıfırla

Sıfırlama:

- Açık pozisyonları siler
- Tüm alış ve satış kayıtlarını siler
- Kâr-zararı sıfırlar
- Portföyü başlangıç sermayesi bekleyen duruma getirir

---

## API uçları

### Sistem

```text
GET /health
```

### Backtest

```text
GET /api/backtest/summary
GET /api/backtest/equity
GET /api/backtest/yearly
GET /api/backtest/walk-forward/summary
GET /api/backtest/walk-forward/equity
GET /api/backtest/walk-forward/yearly
GET /api/backtest/walk-forward/training-log
```

### Günlük sinyaller

```text
GET  /api/signals/latest
POST /api/signals/refresh
```

### Portföy

```text
GET  /api/portfolios
GET  /api/portfolio/{portfolio_name}/summary
GET  /api/portfolio/{portfolio_name}/positions
GET  /api/portfolio/{portfolio_name}/transactions

POST /api/portfolio/{portfolio_name}/initial-capital
POST /api/portfolio/{portfolio_name}/reset
POST /api/portfolio/{portfolio_name}/refresh-prices

POST /api/portfolio/buy
POST /api/portfolio/sell
```

---

## Veri yönetimi

### GitHub’a gönderilmemesi gereken veriler

Aşağıdaki dosyalar yerel veya yeniden üretilebilir niteliktedir:

```text
data/raw/
data/processed/
data/live/
results/
models/*.joblib
models/*.pkl
*.db
*.sqlite
.env
.streamlit/secrets.toml
```

Özellikle SQLite veritabanı kullanıcı sermayesi, alış-satış kayıtları ve portföy geçmişi içerebilir. Public GitHub reposuna gönderilmemelidir.

### Repoda tutulabilecek dosyalar

- Kaynak kodları
- Notebooklar
- `requirements.txt`
- `requirements_app.txt`
- `.env.example`
- BIST100 ticker listesi
- Veri açıklamaları
- Küçük örnek veya şablon dosyaları
- Seçilmiş anonimleştirilmiş ekran görüntüleri
- README ve dokümantasyon

---

## Güvenlik ve gizlilik

Public GitHub reposuna push etmeden önce aşağıdakileri kontrol et:

```powershell
git status
git diff --cached
```

Aşağıdakilerin stage edilmediğinden emin ol:

- API anahtarları
- `.env`
- Kullanıcı işlem kayıtları
- SQLite veritabanı
- Paper-trading günlük planları
- Kişisel klasör yolları
- E-posta adresi, token veya kimlik bilgileri
- Özel sertifika ve anahtar dosyaları

Bir dosya daha önce Git tarafından takip edildiyse `.gitignore` eklemek tek başına yeterli olmaz:

```powershell
git rm --cached <dosya>
```

Klasör için:

```powershell
git rm -r --cached <klasör>
```

---

## Bilinen sınırlamalar

1. **Survivorship bias:** Güncel BIST100 bileşenlerinin geçmiş dönemlere uygulanması sonuçları iyimser gösterebilir.
2. **Veri kaynağı:** Yahoo Finance verileri eksik, gecikmeli veya revize edilmiş olabilir.
3. **Likidite:** Backtest, her sinyalin istenen lotla gerçekleşeceğini varsayabilir.
4. **Tavan ve taban fiyat:** Gerçek piyasada emirlerin gerçekleşmemesi mümkündür.
5. **Slippage:** Sabit kayma oranı her piyasa koşulunu temsil etmez.
6. **Kurumsal olaylar:** Bölünme, temettü ve endeks bileşen değişimleri ek dikkat gerektirir.
7. **ML seçim riski:** Çok sayıda deney dolaylı overfitting oluşturabilir.
8. **Walk-forward dönemi:** Daha önce incelenmiş dönem kusursuz bir holdout değildir.
9. **Fiyat yenileme:** Yahoo Finance gerçek zamanlı profesyonel piyasa terminali değildir.
10. **Otomatik emir yoktur:** Sistem yalnızca analiz ve kayıt amacıyla kullanılır.

---

## Gelecek geliştirmeler

- Tarihsel BIST100 bileşenleri
- Sektör bazlı konsantrasyon limiti
- Likiditeye göre maksimum pozisyon büyüklüğü
- Tavan/taban ve gerçekleşmeyen emir simülasyonu
- İşlem bazlı günlük risk raporu
- Portföy ve işlem kayıtlarının CSV/Excel dışa aktarımı
- Yetkilendirme ve kullanıcı yönetimi
- Docker
- Test kapsamının genişletilmesi
- GitHub Actions ile lint ve test otomasyonu
- Paper-trading performansının zaman içinde otomatik raporlanması

---

## Sorumluluk reddi

Bu proje eğitim ve araştırma amacıyla hazırlanmıştır. Buradaki sinyaller, analizler, modeller ve arayüz çıktıları yatırım tavsiyesi değildir.

Finansal piyasalarda işlem yapmak sermaye kaybı riski taşır. Backtest sonuçları gelecekte aynı performansın elde edileceğini garanti etmez. Gerçek işlem kararı verilmeden önce veri kalitesi, likidite, işlem maliyetleri, vergi, piyasa koşulları ve kişisel risk toleransı ayrıca değerlendirilmelidir.

---

## Geliştirici

**Okan Deniz**

GitHub: `okandeniz`
