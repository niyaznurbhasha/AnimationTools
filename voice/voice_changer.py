import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, lfilter, sosfilt

class AudioProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_segment = AudioSegment.from_file(file_path)
        self.y, self.sr = librosa.load(file_path, sr=None, mono=True)  # Ensure mono for Librosa processing

    def pitch_shift_ya(self, n_steps):
        # Perform pitch shift
        self.y = librosa.effects.pitch_shift(self.y, self.sr, n_steps=n_steps)

    def adjust_speed(self, playback_speed):
        # Adjust speed for PyDub audio_segment
        samples = np.array(self.audio_segment.get_array_of_samples())
        self.audio_segment = self.audio_segment._spawn(samples.tobytes(), overrides={'frame_rate': int(self.audio_segment.frame_rate * playback_speed)})

    def add_texture_and_timbre(self, noise_level=0.02, lowcut=300, highcut=3400, cutoff=3000):
        # This function is simplified for demonstration
        pass

    def apply_formant_shift(self, shift_factor=0.8):
        # Simplified formant shift; actual implementation would be more complex
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
            delay_ms *= 2  # Double the delay for each echo
        self.audio_segment = overlay_audio

    def apply_eq_filter(self, lowcut, highcut, order=5):
        sos = butter(order, [lowcut / (0.5 * self.sr), highcut / (0.5 * self.sr)], btype='band', output='sos')
        self.y = sosfilt(sos, self.y)

    def export_audio(self, export_path):
        # Export using soundfile for Librosa's y
        sf.write(export_path, self.y, self.sr)
        # If you need to export the PyDub audio_segment, you'll need a separate method or line.

if __name__ == "__main__":
    processor = AudioProcessor("/Users/niyaz/Downloads/test_recording.wav")
    processor.pitch_shift_ya(5)  # Example: shift pitch by -2 semitones
    processor.adjust_speed(0.9)  # Example: slow down the playback
    # processor.add_texture_and_timbre()  # Uncomment if implemented
    processor.apply_formant_shift(1.05)  # Apply formant shift
    processor.simple_compression()
    processor.add_reverb()
    # processor.apply_eq_filter(800, 12000)  # Uncomment if needed

    processor.export_audio("processed_audio.wav")  # Export the processed audio
