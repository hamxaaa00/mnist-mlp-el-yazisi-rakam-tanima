"""
Kendi el yazımla yazdığım öğrenci numarasını eğitilmiş MLP modeline verme.

Kullanım:
    1) Beyaz kağıda koyu kalemle numarayı yaz, fotoğrafını çek.
    2) İki yoldan biri:
       a) Her rakamı ayrı fotoğraf olarak "el_yazisi/ham/" içine koy.
          Dosya adının ilk karakteri gerçek rakam olsun: 2_a.jpg, 3_a.jpg, 1_a.jpg ...
       b) Numaranın tamamını tek fotoğraf olarak koy (ör. el_yazisi/ham/numara.jpg),
          --tek-resim ile çalıştır. Rakamlar arasında boşluk bırakmak gerekiyor.
    3) python el_yazisi_test.py                 (ayrı rakam dosyaları)
       python el_yazisi_test.py --tek-resim el_yazisi/ham/numara.jpg

İşlenmiş 28x28 görüntüler "el_yazisi/islenmis/" klasörüne kaydedilir,
tahminler ve karşılaştırma görseli "sonuclar/" klasörüne yazılır.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from mnist_mlp import MLP, MNIST_MEAN, MNIST_STD

OGRENCI_NO = "23120205059"
HAM = Path("el_yazisi/ham")
ISLENMIS = Path("el_yazisi/islenmis")
SONUC = Path("sonuclar")


def mnist_formatina_cevir(gri):
    """Gri (L modunda) bir rakam görüntüsünü MNIST'e benzer 28x28 hale getirir.

    MNIST'te arka plan siyah, rakam beyaz; rakam 20x20 kutuya sığdırılmış ve
    kütle merkezi 28x28 alanın ortasına getirilmiş.
    """
    dizi = np.asarray(gri, dtype=np.float32)
    # Kağıt beyaz, mürekkep koyu -> MNIST gibi olması için ters çevir
    if dizi.mean() > 127:
        dizi = 255 - dizi
    # Gölge ve kağıt dokusunu temizlemek için basit eşikleme
    esik = max(dizi.max() * 0.35, 40)
    dizi[dizi < esik] = 0

    satirlar = np.where(dizi.max(axis=1) > 0)[0]
    sutunlar = np.where(dizi.max(axis=0) > 0)[0]
    if len(satirlar) == 0:
        raise ValueError("Görüntüde rakam bulunamadı")
    dizi = dizi[satirlar[0]:satirlar[-1] + 1, sutunlar[0]:sutunlar[-1] + 1]

    # En-boy oranını koruyarak uzun kenarı 20 piksele indir
    y, x = dizi.shape
    olcek = 20.0 / max(y, x)
    yeni = (max(1, round(x * olcek)), max(1, round(y * olcek)))
    kucuk = np.asarray(Image.fromarray(dizi.astype(np.uint8)).resize(yeni, Image.LANCZOS),
                       dtype=np.float32)

    # 28x28 siyah tuvale ortala, sonra kütle merkezine göre kaydır
    tuval = np.zeros((28, 28), dtype=np.float32)
    ust, sol = (28 - kucuk.shape[0]) // 2, (28 - kucuk.shape[1]) // 2
    tuval[ust:ust + kucuk.shape[0], sol:sol + kucuk.shape[1]] = kucuk
    toplam = tuval.sum()
    if toplam > 0:
        yy, xx = np.indices(tuval.shape)
        cy, cx = (yy * tuval).sum() / toplam, (xx * tuval).sum() / toplam
        tuval = np.roll(tuval, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return np.clip(tuval, 0, 255).astype(np.uint8)


def tek_resmi_bol(yol):
    """Tüm numaranın yazılı olduğu resmi sütun boşluklarına göre rakamlara böler."""
    gri = ImageOps.exif_transpose(Image.open(yol)).convert("L")
    dizi = np.asarray(gri, dtype=np.float32)
    if dizi.mean() > 127:
        dizi = 255 - dizi
    murekkep = dizi > max(dizi.max() * 0.35, 40)
    dolu_sutun = murekkep.sum(axis=0) > 0

    parcalar, bas = [], None
    for i, dolu in enumerate(list(dolu_sutun) + [False]):
        if dolu and bas is None:
            bas = i
        elif not dolu and bas is not None:
            if i - bas > 3:  # çok ince lekeleri at
                parcalar.append(gri.crop((bas, 0, i, gri.height)))
            bas = None
    return parcalar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tek-resim", help="Numaranın tamamının olduğu tek görüntü")
    parser.add_argument("--model", default="sonuclar/en_iyi_model.pt")
    args = parser.parse_args()

    ISLENMIS.mkdir(parents=True, exist_ok=True)
    SONUC.mkdir(exist_ok=True)

    if args.tek_resim:
        rakam_resimleri = tek_resmi_bol(args.tek_resim)
        gercekler = list(OGRENCI_NO)
        if len(rakam_resimleri) != len(gercekler):
            print(f"Uyarı: {len(rakam_resimleri)} parça bulundu, numara {len(gercekler)} hane. "
                  "Rakamlar arasında daha fazla boşluk bırakmayı dene.")
            gercekler = (gercekler + ["?"] * len(rakam_resimleri))[:len(rakam_resimleri)]
    else:
        dosyalar = sorted(p for p in HAM.iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        rakam_resimleri = [ImageOps.exif_transpose(Image.open(p)).convert("L") for p in dosyalar]
        gercekler = [p.name[0] for p in dosyalar]

    kayit = torch.load(args.model, map_location="cpu")
    model = MLP(kayit["ayar"]["gizli_katmanlar"], kayit["ayar"].get("dropout", 0.0))
    model.load_state_dict(kayit["agirliklar"])
    model.eval()

    tahminler, guvenler, goruntuler = [], [], []
    for i, resim in enumerate(rakam_resimleri):
        g28 = mnist_formatina_cevir(resim)
        Image.fromarray(g28).save(ISLENMIS / f"{i:02d}_{gercekler[i]}.png")
        x = torch.tensor(g28, dtype=torch.float32) / 255.0
        x = ((x - MNIST_MEAN) / MNIST_STD).unsqueeze(0)  # eğitimdeki normalizasyonun aynısı
        with torch.no_grad():
            olasilik = torch.softmax(model(x), dim=1)[0]
        tahminler.append(str(olasilik.argmax().item()))
        guvenler.append(olasilik.max().item())
        goruntuler.append(g28)

    dogru = sum(t == g for t, g in zip(tahminler, gercekler))
    print("Gerçek  :", "".join(gercekler))
    print("Tahmin  :", "".join(tahminler))
    print(f"Doğruluk: {dogru}/{len(gercekler)} = {dogru / len(gercekler):.2%}")

    fig, eksenler = plt.subplots(1, len(goruntuler), figsize=(1.3 * len(goruntuler), 2))
    eksenler = np.atleast_1d(eksenler)
    for ax, g, t, gr, gv in zip(eksenler, goruntuler, tahminler, gercekler, guvenler):
        ax.imshow(g, cmap="gray")
        ax.set_title(f"{t} ({gv:.0%})", color="green" if t == gr else "red", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Gerçek: {''.join(gercekler)}  Tahmin: {''.join(tahminler)}")
    fig.tight_layout()
    fig.savefig(SONUC / "el_yazisi_tahminler.png", dpi=150)
    print("Görsel kaydedildi:", SONUC / "el_yazisi_tahminler.png")


if __name__ == "__main__":
    main()
