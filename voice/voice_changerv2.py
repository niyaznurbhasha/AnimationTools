import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, sosfilt

class AudioProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        # Convert m4a to wav if necessary for librosa
        if file_path.endswith(".m4a"):
            self.file_path = self._convert_m4a_to_wav(file_path)
        
        self.audio_segment = AudioSegment.from_file(file_path)
        self.y, self.sr = librosa.load(self.file_path, sr=None, mono=True)  # Ensure mono for Librosa processing

    def _convert_m4a_to_wav(self, file_path):
        # Convert m4a to wav for librosa compatibility
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(".m4a", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    def pitch_shift_ya(self, n_steps):
        self.y = librosa.effects.pitch_shift(y=self.y, sr=self.sr, n_steps=n_steps)


    def adjust_speed(self, playback_speed):
        samples = np.array(self.audio_segment.get_array_of_samples())
        self.audio_segment = self.audio_segment._spawn(samples.tobytes(), overrides={'frame_rate': int(self.audio_segment.frame_rate * playback_speed)})

    def add_texture_and_timbre(self, noise_level=0.02, lowcut=300, highcut=3400):
        noise = np.random.randn(len(self.y))
        self.y = self.y + noise_level * noise
        self.apply_eq_filter(lowcut, highcut)

    def apply_formant_shift(self, shift_factor=0.8):
        D = librosa.stft(self.y)
        D_shifted = np.zeros_like(D)
        for i in range(D.shape[0]):
            if i * shift_factor < D.shape[0]:
                D_shifted[int(i * shift_factor), :] = D[i, :]
        self.y = librosa.istft(D_shifted)

    def simple_compression(self, threshold_dB=-20, ratio=2.0):
        self.audio_segment = self.audio_segment.apply_gain(-threshold_dB * (1 - 1 / ratio))

    def add_reverb(self, delay_ms=100, decay=0.5):
        overlay_audio = self.audio_segment
        for _ in range(3):  # Number of echoes
            overlay_audio = overlay_audio.overlay(self.audio_segment - 6, position=delay_ms)
            delay_ms *= 2
        self.audio_segment = overlay_audio

    def apply_eq_filter(self, lowcut, highcut, order=5):
        # Ensure that lowcut and highcut are within valid range (0 < lowcut < highcut < sr / 2)
        nyquist = 0.5 * self.sr
      #  print(f"Sample Rate: {self.sr}, Nyquist: {nyquist}")
     #   print(f"Lowcut: {lowcut}, Highcut: {highcut}")

        lowcut = max(0.001, min(lowcut, nyquist))  # Ensure lowcut is positive and less than Nyquist
        highcut = max(0.001, min(highcut, nyquist * 0.99))  # Set highcut slightly below Nyquist to avoid error

     #   print(f"Normalized Lowcut: {lowcut / nyquist}, Normalized Highcut: {highcut / nyquist}")

        # Apply the bandpass filter
        sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band', output='sos')
        self.y = sosfilt(sos, self.y)

    def apply_lo_fi(self, target_sr=8000):
        self.y = librosa.core.resample(self.y, orig_sr=self.sr, target_sr=target_sr)
        self.sr = target_sr

    def add_tape_hiss(self, noise_level=0.02):
        noise = np.random.randn(len(self.y)) * noise_level
        self.y = self.y + noise

    def apply_distortion(self, distortion_level=0.1):
        self.y = np.clip(self.y, -distortion_level, distortion_level)

    def export_audio(self, export_path):
        sf.write(export_path, self.y, self.sr)
        self.audio_segment.export(export_path.replace('.wav', '_pydub.wav'), format='wav')
    
    def apply_saturation(self, level=1.2):
    # Amplify the signal slightly and apply tanh for saturation-like effect
        self.y = np.tanh(self.y * level)

if __name__ == "__main__":
    processor = AudioProcessor("test_recording.m4a")
    processor.pitch_shift_ya(n_steps=-1)  # Shift pitch down by 2 semitones for a slightly lower voice

    # Apply Alan Watts-like effects
    processor.apply_lo_fi(target_sr=8000)  # Lo-Fi effect
    processor.apply_eq_filter(lowcut=100, highcut=4000)  # EQ to boost mids, cut lows and highs
  #  processor.add_tape_hiss(noise_level=0.02)  # Add subtle tape hiss
    processor.add_reverb(delay_ms=100, decay=0.5)  # Add light reverb
  #  processor.apply_distortion(distortion_level=0.01)  # Subtle distortion
    processor.simple_compression(threshold_dB=-20, ratio=3.0)  # Compression
    processor.apply_saturation(level=1.7)  # Apply subtle saturation for a warmer effect

    processor.export_audio("alan_watts_style_audio.wav")
