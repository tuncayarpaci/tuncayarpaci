import pyaudio
import numpy as np
import whisper

# Whisper modelini yükleyin
model = whisper.load_model("base")

# Mikrofon ayarları
RATE = 16000  # Örnekleme oranı
CHUNK = 1024  # Veri bloğu boyutu

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Ses dinleniyor... (Ctrl+C ile çıkabilirsiniz)")

try:
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        result = model.transcribe(audio_data, language="en")  # Dil seçimini yapabilirsiniz
        print(result["text"])  # Gerçek zamanlı metin çıktısı
except KeyboardInterrupt:
    print("\nDinleme durduruldu.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
