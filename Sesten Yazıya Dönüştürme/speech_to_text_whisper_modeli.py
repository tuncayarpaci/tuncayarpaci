import os
import queue
import sounddevice as sd
import torch
import numpy as np
import whisper
import wave
import io
import warnings

SAMPLE_RATE = 16000  # Daha yüksek örnekleme oranı

warnings.filterwarnings("ignore", category=FutureWarning)

# Model ayarları
MODEL_NAME = "tiny"  # Whisper model türü: tiny, base, small, medium, large
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_NAME, device=device)

# Mikrofon ayarları
DEVICE = None  # Varsayılan mikrofon
BLOCK_SIZE = SAMPLE_RATE // 4  # 0.25 saniyelik bloklar

audio_queue = queue.Queue(maxsize=50)  # Ses kuyruğu
OUTPUT_FILE = "sonuclar.txt"  # Çıktı dosyası

# Çıktı dosyasını temizleme
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("")

def save_audio_to_wav(audio_data, filename):
    """Ses verisini WAV dosyasına kaydet."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)

def transcribe_audio(filename):
    """Whisper modeliyle sesi yazıya dök."""
    language = "tr"
    result = model.transcribe(filename, fp16=torch.cuda.is_available(), language=language)
    text = result.get("text", "")
    if text:
        print("Sonuç:", text)
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")

def callback(indata, frames, time, status):
    """Ses cihazından gelen verileri al ve kuyruğa koy."""
    if status:
        print(f"Ses cihazı hatası: {status}", flush=True)
    try:
        # Gelen veri NumPy dizisine dönüştürülüyor
        audio_data = np.frombuffer(indata, dtype='int16')
        audio_queue.put_nowait(audio_data)
    except queue.Full:
        print("Ses kuyruğu doldu, bazı veriler kaybolabilir.")

try:
    print("Mevcut ses cihazları:")
    print(sd.query_devices())

    TEMP_AUDIO_FILE = "temp_audio.wav"  # Geçici dosya adı

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                           channels=1, callback=callback, device=DEVICE):
        print("Dinleme başlatıldı. Çıkmak için Ctrl+C kullanın.")
        while True:
            # Kuyruktaki veriyi al
            data = audio_queue.get()

            # Geçici dosyaya kaydet
            save_audio_to_wav(data, TEMP_AUDIO_FILE)

            # Whisper modeliyle yazıya dök
            transcribe_audio(TEMP_AUDIO_FILE)
except KeyboardInterrupt:
    print("\nDinleme durduruldu.")
except Exception as e:
    print(f"Hata oluştu: {e}")
finally:
    # Geçici dosyayı temizle
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
