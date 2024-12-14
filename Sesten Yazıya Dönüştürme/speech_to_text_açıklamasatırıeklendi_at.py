import os
import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
import noisereduce as nr
import numpy as np

# Model ayarları
MODEL_PATHS = {
    "tr": "C:/modeller/vosk-model-small-tr-0.3",
    "en": "C:/modeller/vosk-model-small-en-us-0.15"
}
DEFAULT_LANGUAGE = "tr"  # Varsayılan dil

# Varsayılan dil için model kontrolü
if DEFAULT_LANGUAGE not in MODEL_PATHS:
    raise ValueError(f"Belirtilen varsayılan dil ({DEFAULT_LANGUAGE}) için bir model yolu tanımlanmadı.")

MODEL_PATH = MODEL_PATHS[DEFAULT_LANGUAGE]
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model yolu bulunamadı: {MODEL_PATH}")

# Vosk modeli ve tanıyıcıyı başlat
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# Mikrofon ayarları
SAMPLE_RATE = 16000
DEVICE = None  # Varsayılan mikrofon

# Ses kaydı için bir kuyruk
audio_queue = queue.Queue(maxsize=10)  # Kuyruk boyutu sınırlı

# Çıktı dosyası
OUTPUT_FILE = "sonuclar.txt"

# Çıktı dosyasını temizleme
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("")

def callback(indata, frames, time, status):
    """Ses cihazından gelen verileri al ve kuyruğa koy."""
    if status:
        print(f"Ses cihazı hatası: {status}", flush=True)
    try:
        # Gürültü azaltma
        audio_data = np.frombuffer(indata, dtype="int16")
        reduced_noise = nr.reduce_noise(y=audio_data, sr=SAMPLE_RATE)
        audio_queue.put_nowait(reduced_noise.tobytes())
    except queue.Full:
        print("Ses kuyruğu doldu, bazı veriler kaybolabilir.")

def process_result(result_json):
    """JSON formatındaki sonucu ayrıştır, yazdır ve dosyaya kaydet."""
    try:
        result = json.loads(result_json)
        if "text" in result and result["text"]:
            print("Sonuç:", result["text"])
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(result["text"] + "\n")
    except json.JSONDecodeError:
        print("Sonuç işlenirken bir hata oluştu: Geçersiz JSON.")

def change_language(language):
    """Model dilini değiştir."""
    global model, recognizer
    if language not in MODEL_PATHS:
        raise ValueError(f"Dil için model yolu bulunamadı: {language}")
    new_model_path = MODEL_PATHS[language]
    if not os.path.exists(new_model_path):
        raise FileNotFoundError(f"Model yolu bulunamadı: {new_model_path}")
    model = Model(new_model_path)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    print(f"Dil değiştirildi: {language}")

print("Ses dinleniyor... (Ctrl+C ile çıkmak için)")

try:
    # Mikrofonu aç
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype="int16",
                           channels=1, callback=callback, device=DEVICE):
        while True:
            # Kuyruktaki veriyi al ve tanıma gönder
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                process_result(recognizer.Result())
            else:
                partial_result = recognizer.PartialResult()
                print("Kısmi Sonuç:", json.loads(partial_result).get("partial", ""))
except KeyboardInterrupt:
    print("\nDinleme durduruldu.")
except Exception as e:
    print(f"Hata oluştu: {e}")
