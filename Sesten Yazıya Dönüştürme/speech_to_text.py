import os
import queue
import sounddevice as sd
import time
from vosk import Model, KaldiRecognizer

# Türkçe model yolunu ayarlayın
MODEL_PATH = "C:/modeller/vosk-model-small-tr-0.3"
model = Model(MODEL_PATH)

# Mikrofon ayarları
SAMPLE_RATE = 16000
DEVICE = None  # Varsayılan mikrofon

# Vosk Tanıyıcı
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Ses kaydı için bir kuyruk
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    """Ses cihazından gelen verileri al ve kuyruğa koy."""
    if status:
        print(f"Ses cihazı hatası: {status}", flush=True)
    audio_queue.put(bytes(indata))

print("Ses dinleniyor... (Ctrl+C ile çıkmak için)")

try:
    # Mikrofonu aç
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype="int16",
                           channels=1, callback=callback, device=DEVICE):
        while True:
            # Kuyruktaki veriyi al ve tanıma gönder
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                print("Sonuç:", result)
               
            else:
                partial_result = recognizer.PartialResult()
                print("Kısmi Sonuç:", partial_result)
                
except KeyboardInterrupt:
    print("\nDinleme durduruldu.")
except Exception as e:
    print(f"Hata oluştu: {e}")
