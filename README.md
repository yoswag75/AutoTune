# 🎵 Auto-Tuning 

A web-based audio processing application built with Python, Streamlit, and Librosa. This tool performs automatic pitch correction (Auto-Tune) by detecting the fundamental frequency of audio segments and shifting them to the nearest musical note in the chromatic scale.

## 📖 Overview

This application provides a user-friendly interface for digital signal processing (DSP). It allows users to upload vocal or instrumental tracks and apply pitch quantization. The system uses the YIN algorithm for pitch detection and performs phase-vocoded pitch shifting to "tune" audio to perfect semitones (A4 = 440Hz standard).

## ✨ Key Features

- **🎙️ Automatic Pitch Quantization**: Snaps detected frequencies to the nearest valid musical note (C, C#, D, etc.).
- **🎛️ Adjustable Correction Strength**:
  - `1.0`: Hard, robotic "T-Pain" effect
  - `0.1 - 0.5`: Subtle, natural pitch correction
- **🗣️ Voice Activity Detection (VAD)**: Uses RMS energy and Zero-Crossing Rate to process only voiced segments, ignoring background noise.
- **🌊 Smooth Crossfading**: Linear crossfading between processed chunks to eliminate audio artifacts (clicks/pops).
- **📂 Broad File Support**: Supports `.wav`, `.mp3`, `.flac`, and `.m4a` files.

## 🛠️ Tech Stack

- **Streamlit**: Interactive web interface
- **Librosa**: Audio analysis and manipulation (pitch shifting, time stretching)
- **NumPy**: High-performance mathematical operations
- **SoundFile**: Audio file reading/writing

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/streamlit-auto-tuner.git
cd streamlit-auto-tuner
```

### 2. Set up a virtual environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file with the following contents:
```
streamlit>=1.28.0
numpy>=1.24.0
librosa>=0.10.0
soundfile>=0.12.0
```

Then install using pip:
```bash
pip install -r requirements.txt
```

*Note: You may need FFmpeg installed on your system to process compressed formats like MP3.*

## 💻 Usage

Run the Streamlit application from your terminal:

```bash
streamlit run main.py
```

The app will open automatically in your web browser at `http://localhost:8501`.

### Application Workflow:

1. **Upload**: Drag and drop your audio file into the uploader
2. **Adjust Settings**: Use the sidebar slider to control Pitch Correction Strength
3. **Process**: Click "Process Audio File"
4. **Download**: Listen to the "Auto-tuned Audio" and click the download button to save the `.wav` file

## ⚙️ How It Works

The `AutoTuneProcessor` class handles the DSP pipeline:

1. **Chunking**: Audio is processed in small windows to manage memory and allow progress tracking
2. **Voice Activity Detection**: The script checks RMS (volume) and Spectral Centroid (brightness) to determine if a chunk contains a voice
3. **Pitch Detection**: `librosa.yin` estimates the fundamental frequency (f0)
4. **Note Mapping**: The frequency is compared to a generated dictionary of musical notes (C2 to B6)
5. **Pitch Shifting**: The audio is shifted by the semitone difference between the detected pitch and the target note
6. **Reconstruction**: Overlapping chunks are crossfaded to ensure the output audio is seamless

## ⚠️ Limitations

- **Monophonic Sources Only**: The algorithm works best on single voices or instruments. Polyphonic audio (chords, full bands) will result in unpredictable artifacts
- **Processing Time**: As this runs on the CPU using Python, long files (3+ minutes) may take a moment to process depending on your hardware

## 📄 License

This project is open-source and available under the GNU License.
