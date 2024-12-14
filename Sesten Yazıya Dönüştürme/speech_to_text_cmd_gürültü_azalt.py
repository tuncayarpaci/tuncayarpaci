import os
import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
import noisereduce as nr
import numpy as np

# Türkçe model yolunu ayarlayın
MODEL_PATH = "C:/modeller/vosk-model-small-tr-0.3"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model yolu bulunamadı: {MODEL_PATH}")
model = Model(MODEL_PATH)

# Mikrofon ayarları
SAMPLE_RATE = 16000
DEVICE = None  # Varsayılan mikrofon

# Vosk Tanıyıcı
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Ses kaydı için bir kuyruk
audio_queue = queue.Queue()

# Çıktı dosyası
OUTPUT_FILE = "sonuclar.txt"

# Çıktı dosyasını temizleme (isteğe bağlı)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("")

def callback(indata, frames, time, status):
    """Ses cihazından gelen verileri al ve kuyruğa koy."""
    if status:
        print(f"Ses cihazı hatası: {status}", flush=True)
    # Gürültü azaltma uygulanabilir
    audio_data = np.frombuffer(indata, dtype="int16")
    reduced_noise = nr.reduce_noise(y=audio_data, sr=SAMPLE_RATE)
    audio_queue.put(reduced_noise.tobytes())

def process_result(result_json):
    """JSON formatındaki sonucu ayrıştır, yazdır ve dosyaya kaydet."""
    result = json.loads(result_json)
    if "text" in result and result["text"]:
        print("Sonuç:", result["text"])
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(result["text"] + "\n")

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
                print("Kısmi Sonuç:", json.loads(partial_result)["partial"])
                
except KeyboardInterrupt:
    print("\nDinleme durduruldu.")
except Exception as e:
    print(f"Hata oluştu: {e}")
