"""
BIL485 Derin Öğrenme - Ödev 1
MNIST veri seti ile MLP tabanlı el yazısı rakam tanıma

Ahmet Hamza Yavuz - 23120205059

Çalıştırma:
    pip install -r requirements.txt
    python mnist_mlp.py                 # tüm deneyler
    python mnist_mlp.py --deney 1       # sadece temel model

Çıktılar "sonuclar/" klasörüne yazılır:
    - deneyX_grafik.png    : loss ve accuracy grafikleri
    - sonuclar.json        : tüm deneylerin metrikleri
    - en_iyi_model.pt      : validation accuracy'si en yüksek model
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ekran olmadan grafik kaydetmek için
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# MNIST eğitim setinin ortalama ve standart sapması (normalizasyon için)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

SONUC_KLASORU = Path("sonuclar")
SEED = 42


class MLP(nn.Module):
    """Tam bağlı katmanlardan oluşan çok katmanlı algılayıcı (CNN yok)."""

    def __init__(self, gizli_katmanlar=(256, 128), dropout=0.0):
        super().__init__()
        katmanlar = [nn.Flatten()]  # 28x28 -> 784
        giris = 28 * 28
        for boyut in gizli_katmanlar:
            katmanlar.append(nn.Linear(giris, boyut))
            katmanlar.append(nn.ReLU())
            if dropout > 0:
                katmanlar.append(nn.Dropout(dropout))
            giris = boyut
        katmanlar.append(nn.Linear(giris, 10))  # 10 sınıf, softmax CrossEntropyLoss içinde
        self.ag = nn.Sequential(*katmanlar)

    def forward(self, x):
        return self.ag(x)


def veri_yukle(batch_size):
    donusum = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    egitim_tam = datasets.MNIST("veri", train=True, download=True, transform=donusum)
    test = datasets.MNIST("veri", train=False, download=True, transform=donusum)

    # 60.000 eğitim örneği -> 50.000 train + 10.000 validation
    # Test seti (10.000) sadece en sonda bir kere kullanılıyor
    uretec = torch.Generator().manual_seed(SEED)
    egitim, dogrulama = random_split(egitim_tam, [50_000, 10_000], generator=uretec)

    return (
        DataLoader(egitim, batch_size=batch_size, shuffle=True),
        DataLoader(dogrulama, batch_size=1000),
        DataLoader(test, batch_size=1000),
    )


def bir_epoch(model, yukleyici, kayip_fn, optimizer, cihaz, egitim=True):
    model.train() if egitim else model.eval()
    toplam_kayip, dogru, adet = 0.0, 0, 0
    with torch.set_grad_enabled(egitim):
        for x, y in yukleyici:
            x, y = x.to(cihaz), y.to(cihaz)
            cikti = model(x)
            kayip = kayip_fn(cikti, y)
            if egitim:
                optimizer.zero_grad()
                kayip.backward()
                optimizer.step()
            toplam_kayip += kayip.item() * x.size(0)
            dogru += (cikti.argmax(1) == y).sum().item()
            adet += x.size(0)
    return toplam_kayip / adet, dogru / adet


def grafik_ciz(gecmis, baslik, dosya):
    epochlar = range(1, len(gecmis["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochlar, gecmis["train_loss"], "o-", label="Train")
    ax1.plot(epochlar, gecmis["val_loss"], "s-", label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochlar, gecmis["train_acc"], "o-", label="Train")
    ax2.plot(epochlar, gecmis["val_acc"], "s-", label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle(baslik)
    fig.tight_layout()
    fig.savefig(dosya, dpi=120)
    plt.close(fig)


def deney_calistir(ad, ayar, cihaz):
    print(f"\n===== {ad} =====")
    print(ayar)
    torch.manual_seed(SEED)
    train_dl, val_dl, test_dl = veri_yukle(ayar["batch_size"])

    model = MLP(ayar["gizli_katmanlar"], ayar.get("dropout", 0.0)).to(cihaz)
    kayip_fn = nn.CrossEntropyLoss()
    if ayar["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=ayar["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=ayar["lr"], momentum=0.9)

    gecmis = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    en_iyi_val, en_iyi_durum = 0.0, None

    baslangic = time.time()
    for epoch in range(1, ayar["epoch"] + 1):
        tr_loss, tr_acc = bir_epoch(model, train_dl, kayip_fn, optimizer, cihaz, egitim=True)
        va_loss, va_acc = bir_epoch(model, val_dl, kayip_fn, None, cihaz, egitim=False)
        gecmis["train_loss"].append(tr_loss)
        gecmis["val_loss"].append(va_loss)
        gecmis["train_acc"].append(tr_acc)
        gecmis["val_acc"].append(va_acc)
        # model seçimi validation'a göre yapılıyor, test setine bakılmıyor
        if va_acc > en_iyi_val:
            en_iyi_val = va_acc
            en_iyi_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
    sure = time.time() - baslangic

    # en iyi validation ağırlıkları ile test (test seti burada ilk ve tek kez kullanılıyor)
    model.load_state_dict(en_iyi_durum)
    te_loss, te_acc = bir_epoch(model, test_dl, kayip_fn, None, cihaz, egitim=False)
    print(f"Test accuracy: {te_acc:.4f} | Eğitim süresi: {sure:.1f} sn")

    grafik_ciz(gecmis, ad, SONUC_KLASORU / f"{ad.split(' ')[0].lower()}_grafik.png")
    sonuc = {
        "ayar": ayar,
        "train_loss": gecmis["train_loss"][-1],
        "val_loss": gecmis["val_loss"][-1],
        "train_acc": gecmis["train_acc"][-1],
        "val_acc": gecmis["val_acc"][-1],
        "en_iyi_val_acc": en_iyi_val,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "egitim_suresi_sn": round(sure, 1),
        "gecmis": gecmis,
    }
    return sonuc, model


# Deney 1 ödevde istenen temel model; diğerleri hiperparametre karşılaştırması için
DENEYLER = {
    "1": ("Deney1 - Temel MLP (2 gizli katman, Adam)",
          dict(gizli_katmanlar=(256, 128), optimizer="adam", lr=1e-3, batch_size=64, epoch=10)),
    "2": ("Deney2 - Daha derin ve geniş (3 gizli katman)",
          dict(gizli_katmanlar=(512, 256, 128), optimizer="adam", lr=1e-3, batch_size=64, epoch=10)),
    "3": ("Deney3 - SGD (momentum 0.9)",
          dict(gizli_katmanlar=(256, 128), optimizer="sgd", lr=0.01, batch_size=64, epoch=10)),
    "4": ("Deney4 - Dropout 0.2",
          dict(gizli_katmanlar=(256, 128), optimizer="adam", lr=1e-3, batch_size=64, epoch=10,
               dropout=0.2)),
    "5": ("Deney5 - Yüksek öğrenme oranı (lr=0.01)",
          dict(gizli_katmanlar=(256, 128), optimizer="adam", lr=1e-2, batch_size=64, epoch=10)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deney", choices=list(DENEYLER) + ["hepsi"], default="hepsi")
    args = parser.parse_args()

    SONUC_KLASORU.mkdir(exist_ok=True)
    cihaz = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("Cihaz:", cihaz)

    secilen = DENEYLER.keys() if args.deney == "hepsi" else [args.deney]
    sonuclar, en_iyi_acc = {}, 0.0
    for anahtar in secilen:
        ad, ayar = DENEYLER[anahtar]
        sonuc, model = deney_calistir(ad, ayar, cihaz)
        sonuclar[ad] = sonuc
        # el yazısı testinde kullanılacak model validation'a göre seçiliyor
        if sonuc["en_iyi_val_acc"] > en_iyi_acc:
            en_iyi_acc = sonuc["en_iyi_val_acc"]
            torch.save({"ayar": ayar, "agirliklar": model.state_dict()},
                       SONUC_KLASORU / "en_iyi_model.pt")

    with open(SONUC_KLASORU / "sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2)

    print("\n===== ÖZET TABLO =====")
    print(f"{'Deney':50s} {'TrainAcc':>9s} {'ValAcc':>8s} {'TestAcc':>8s} {'Süre(sn)':>9s}")
    for ad, s in sonuclar.items():
        print(f"{ad:50s} {s['train_acc']:9.4f} {s['val_acc']:8.4f} "
              f"{s['test_acc']:8.4f} {s['egitim_suresi_sn']:9.1f}")


if __name__ == "__main__":
    main()
