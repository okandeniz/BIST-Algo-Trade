# Notebook rehberi

Notebooklar günlük uygulamadan ayrılmış üç grupta tutulur. Günlük sinyal ve
paper-trading işlemleri notebooklardan değil Streamlit arayüzünden yürütülür.

## `active/` — tekrar üretilebilir raporlar

| Notebook | Amaç | Ne zaman çalıştırılır? |
|---|---|---|
| `01_data_collection_robot.ipynb` | Tarihsel hisse ve endeks verisini indirir | Araştırma verisi yenileneceğinde |
| `02_data_quality_robot.ipynb` | Veriyi temizler ve kalite raporu üretir | Veri toplama sonrasında |
| `09_final_strategy_vs_bist100.ipynb` | Promote edilmiş Baseline raporunu üretir | Strateji veya araştırma verisi değiştiğinde |
| `15_walk_forward_ml_comparison_robot.ipynb` | Kilitli ML challenger karşılaştırmasını üretir | Model artifact'i veya veri değiştiğinde |

Önerilen sıra:

```text
01 → 02 → 09 → 15
```

Bu sıra günlük kullanım için zorunlu değildir. Uygulama daha önce promote edilmiş
artifact'leri okur.

## `labs/` — aktif deneyler

Buradaki notebooklar ürün davranışını doğrudan değiştirmez. Bir deney ancak
Development ve Validation ölçütlerini geçip açıkça promote edildikten sonra
`active/` akışına veya uygulama artifact'lerine alınır.

- `17_baseline_enhancement_experiments.ipynb`
- `18_rs126_soft_priority_experiments.ipynb`
- `19_rs126_enhanced_ml_hybrid_experiments.ipynb`
- `20_rs126_ml_ranking_position_sizing_experiments.ipynb`

## `archive/` — araştırma geçmişi

Parametre seçimi, robustness, eski günlük kullanım akışları ve reddedilmiş ML
deneyleri burada korunur. Yeni kullanıcıların bu dosyaları sırayla çalıştırması
gerekmez.

Özellikle `08` ve `14` numaralı paper-trading notebookları tarihsel referanstır;
günlük işlem kaydı için arayüz kullanılmalıdır.

## Çalıştırma

JupyterLab'i proje kökünden başlatın:

```powershell
python -m jupyter lab
```

Notebooklar proje kökünü bulundukları klasörden bağımsız olarak üst dizinlerdeki
`src/` klasörünü arayarak belirler.
