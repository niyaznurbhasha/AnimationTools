import torch
from TTS.api import TTS
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
import noisereduce as nr
from pydub import AudioSegment
import subprocess
import os

device = 'cpu'
# Load an audio file and return the waveform and sampling rate
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Save the audio file
def save_audio(file_path, y, sr):
    sf.write(file_path, y, sr)

# Demucs separation (to remove background music from the target voice)
def separate_vocals_with_demucs(input_path, output_directory):
    # Use subprocess to call Demucs for separating the audio
    print(f"Separating vocals for {input_path}...")
    subprocess.run(["demucs", "-n", "mdx_extra", input_path, "-o", output_directory], check=True)
    print("Demucs separation completed!")

# Noise reduction
def noise_reduction(input_path, output_path):
    y, sr = load_audio(input_path)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    save_audio(output_path, reduced_noise, sr)

# Equalization (Low-pass filter to smooth out high-frequency cracks)
def equalization(input_path, output_path, cutoff=5000):
    y, sr = load_audio(input_path)
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='low', analog=False)
    y_filtered = lfilter(b, a, y)
    save_audio(output_path, y_filtered, sr)

# Compression (To smooth out dynamic range)
def compression(input_path, output_path, threshold=0.5, ratio=2):
    sound = AudioSegment.from_file(input_path)
    compressed_sound = sound.compress_dynamic_range(threshold=threshold, ratio=ratio)
    compressed_sound.export(output_path, format="wav")

# Reverb (To smooth transitions and soften cracks)
def add_reverb(input_path, output_path, reverberance=50):
    sound = AudioSegment.from_file(input_path)
    reverb_sound = sound.overlay(sound, gain_during_overlay=-reverberance)
    reverb_sound.export(output_path, format="wav")

# De-clicking and de-cracking
def de_click(input_path, output_path, prop_decrease=0.6):
    y, sr = load_audio(input_path)
    # Applying noise reduction with tuned parameters
    y_processed = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
    save_audio(output_path, y_processed, sr)

# Voice conversion after separating vocals
def perform_voice_conversion(vocal_wav, target_wav, output_path):
    # Load the voice conversion model
    model_name = "voice_conversion_models/multilingual/vctk/freevc24"
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False).to(device)

    # Perform the voice conversion
    tts.voice_conversion_to_file(source_wav=vocal_wav, target_wav=target_wav, file_path=output_path)

# Main function to apply Demucs, voice conversion, and post-processing
def main(post_process_method, input_audio_path, target_voice_path, post_processed_output, remove_music=False):
    output_directory = "output_directory"

    if remove_music:
        # Separate vocals only from the target voice path (not the source)
        print(f"Separating background music from target: {target_voice_path}")
        separate_vocals_with_demucs(target_voice_path, output_directory)
        
        # Use the separated target vocals for voice conversion
        target_vocal_path = f"{output_directory}/mdx_extra/{os.path.splitext(os.path.basename(target_voice_path))[0]}/vocals.wav"
        if not os.path.exists(target_vocal_path):
            raise FileNotFoundError(f"Could not find the separated target vocals at {target_vocal_path}")
    else:
        target_vocal_path = target_voice_path  # No separation needed

    # Perform voice conversion after separating target vocals
    print(f"Performing voice conversion on vocals from {input_audio_path} using target vocals {target_vocal_path}...")
    converted_output = "converted_output.wav"
    perform_voice_conversion(input_audio_path, target_vocal_path, converted_output)

    # Post-processing (optional)
    if post_process_method == 'noise_reduction':
        noise_reduction(converted_output, post_processed_output)
    elif post_process_method == 'equalization':
        equalization(converted_output, post_processed_output)
    elif post_process_method == 'compression':
        compression(converted_output, post_processed_output)
    elif post_process_method == 'reverb':
        add_reverb(converted_output, post_processed_output)
    elif post_process_method == 'de_click':
        de_click(converted_output, post_processed_output)
    elif post_process_method == 'none':  # No post-processing
        print("No post-processing applied. Output is the converted audio.")
    else:
        print(f"Unknown method: {post_process_method}. Choose from ['noise_reduction', 'equalization', 'compression', 'reverb', 'de_click', 'none']")

# Example usage:
if __name__ == "__main__":
    post_process_method = "equalization"  # Choose method: 'noise_reduction', 'equalization', 'compression', 'reverb', 'de_click', 'none'
    input_audio_path = "/Users/niyaz/Downloads/converted_output.wav"  # Input audio with music
    target_voice_path = "/Users/niyaz/Downloads/att2.mp3"  # Target speaker's voice (Person B)
    post_processed_output = "final_output3.wav"  # Output after post-processing
    remove_music = True  # Set to True to remove music using Demucs for the target audio
    main(post_process_method, input_audio_path, target_voice_path, post_processed_output, remove_music)
