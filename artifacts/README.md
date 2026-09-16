# Release artifacts

`release_manifest.json`, çalışan uygulamanın hangi araştırma çıktısını kullandığını
tanımlar. Büyük veri, model ve sonuç dosyaları Git'e eklenmez; manifest yalnızca
proje köküne göre göreli yollarını sürümler.

Yeni bir araştırma sonucu otomatik olarak ürüne alınmaz. Promote işlemi için:

1. adayın Validation kararını tamamlayın,
2. üretilecek dosyaları gözden geçirin,
3. manifestteki ilgili mantıksal anahtarları yeni dosyalara yönlendirin,
4. `release_id` ve `promoted_at` alanlarını güncelleyin,
5. testleri ve API artifact durumunu çalıştırın.

Uygulama servisleri dosya adlarını veya notebook numaralarını bilmez; yalnızca
`baseline.periods`, `walk_forward.equity` veya `challenger.model` gibi anahtarları
ister.
