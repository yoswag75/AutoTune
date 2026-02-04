import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

class AutoTuneProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.note_frequencies = self._generate_note_frequencies()
        self.pitch_shift_strength = 1.0
        
    def _generate_note_frequencies(self):
        """Generate frequencies for all notes in multiple octaves"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        frequencies = {}
        
        for octave in range(2, 7):
            for i, note in enumerate(notes):
                semitone_offset = (octave - 4) * 12 + (i - 9)
                frequency = 440 * (2 ** (semitone_offset / 12))
                frequencies[f"{note}{octave}"] = frequency
                
        return frequencies
    
    def _find_nearest_note(self, frequency):
        """Find the nearest musical note for a given frequency"""
        if frequency <= 0:
            return frequency, "Silent"
            
        min_diff = float('inf')
        nearest_freq = frequency
        
        for note_freq in self.note_frequencies.values():
            diff = abs(frequency - note_freq)
            if diff < min_diff:
                min_diff = diff
                nearest_freq = note_freq
                
        return nearest_freq, ""
    
    def _detect_pitch_yin(self, audio_chunk):
        """Pitch detection using YIN algorithm"""
        try:
            f0 = librosa.yin(
                audio_chunk,
                fmin=80,
                fmax=800,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                threshold=0.1
            )
            
            valid_pitches = f0[f0 > 0]
            if len(valid_pitches) < 3:
                return None
                
            # Remove outliers
            q75, q25 = np.percentile(valid_pitches, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (iqr * 1.5)
            upper_bound = q75 + (iqr * 1.5)
            
            filtered_pitches = valid_pitches[(valid_pitches >= lower_bound) & (valid_pitches <= upper_bound)]
            
            return np.mean(filtered_pitches) if len(filtered_pitches) > 0 else None
        except:
            return None
    
    def _detect_voiced_segments(self, audio_chunk):
        """Voice activity detection"""
        rms = librosa.feature.rms(y=audio_chunk, frame_length=2048, hop_length=512)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_chunk, frame_length=2048, hop_length=512)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=self.sample_rate, hop_length=512)[0]
        
        energy_threshold = np.mean(rms) * 0.3
        zcr_threshold = np.mean(zcr) * 1.5
        
        is_voiced = (rms > energy_threshold) & (zcr < zcr_threshold) & (spectral_centroid > 200) & (spectral_centroid < 3000)
        
        return np.mean(is_voiced) > 0.5
    
    def _crossfade_audio(self, original, processed):
        """Crossfade between original and processed audio"""
        fade_length = min(len(original) // 8, 1024)
        
        if fade_length < 10:
            return processed
            
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        result = processed.copy()
        result[:fade_length] = original[:fade_length] * fade_out + processed[:fade_length] * fade_in
        result[-fade_length:] = original[-fade_length:] * fade_in + processed[-fade_length:] * fade_out
        
        return result
    
    def process_audio_chunk(self, audio_chunk):
        """Process a single audio chunk"""
        if len(audio_chunk) < self.hop_length * 4:
            return audio_chunk
            
        if not self._detect_voiced_segments(audio_chunk):
            return audio_chunk
            
        detected_pitch = self._detect_pitch_yin(audio_chunk)
        
        if detected_pitch is None or detected_pitch <= 0:
            return audio_chunk
            
        target_frequency, _ = self._find_nearest_note(detected_pitch)
        
        # Calculate semitone shift
        semitone_shift = 12 * np.log2(target_frequency / detected_pitch)
        
        if abs(semitone_shift) < 0.1:
            return audio_chunk
            
        # Apply pitch correction
        corrected_semitone_shift = semitone_shift * self.pitch_shift_strength * 0.7
        
        # Apply pitch shifting
        corrected_audio = librosa.effects.pitch_shift(
            audio_chunk, 
            sr=self.sample_rate, 
            n_steps=corrected_semitone_shift,
            hop_length=self.hop_length
        )
        
        # Crossfade for smooth transitions
        return self._crossfade_audio(audio_chunk, corrected_audio)
    
    def process_file(self, uploaded_file):
        """Process an uploaded audio file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            audio, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            # Process in chunks with overlap
            chunk_size = self.sample_rate // 4
            overlap = chunk_size // 8
            processed_chunks = []
            
            progress_bar = st.progress(0)
            
            for i in range(0, len(audio), chunk_size - overlap):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size // 2:
                    break
                    
                processed_chunk = self.process_audio_chunk(chunk)
                processed_chunks.append(processed_chunk)
                
                progress = min(i / len(audio), 0.95)
                progress_bar.progress(progress)
            
            # Combine chunks with crossfading
            processed_audio = self._combine_chunks(processed_chunks, overlap)
            
            # Match original length
            min_len = min(len(processed_audio), len(audio))
            processed_audio = processed_audio[:min_len]
            
            # Normalize to match original levels
            original_rms = np.sqrt(np.mean(audio**2))
            processed_rms = np.sqrt(np.mean(processed_audio**2))
            
            if processed_rms > 0:
                processed_audio = processed_audio * (original_rms / processed_rms) * 0.9
            
            # Prevent clipping
            processed_audio = np.clip(processed_audio, -0.95, 0.95)
            
            progress_bar.progress(1.0)
            
            # Save processed audio
            output_path = tmp_path.replace('.wav', '_autotuned.wav')
            sf.write(output_path, processed_audio, self.sample_rate)
            
            return output_path, processed_audio
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _combine_chunks(self, chunks, overlap):
        """Combine audio chunks with crossfading"""
        if not chunks:
            return np.array([])
            
        if len(chunks) == 1:
            return chunks[0]
            
        result = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                if overlap > 0 and len(result) > 0:
                    prev_tail = result[-1][-overlap:]
                    curr_head = chunk[:overlap]
                    
                    fade = np.linspace(0, 1, overlap)
                    crossfaded = prev_tail * (1 - fade) + curr_head * fade
                    
                    result[-1] = result[-1][:-overlap]
                    result.append(np.concatenate([crossfaded, chunk[overlap:]]))
                else:
                    result.append(chunk)
        
        return np.concatenate(result)

def main():
    st.set_page_config(
        page_title="Auto-Tuning System",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Auto-Tuning System")
    st.markdown("Upload an audio file to apply auto-tuning and correct pitch to the nearest musical notes.")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = AutoTuneProcessor()
    
    processor = st.session_state.processor
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    processor.pitch_shift_strength = st.sidebar.slider(
        "Pitch Correction Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.1,
        help="1.0 = Full correction, 0.0 = No correction"
    )
    
    sample_rates = [22050, 44100]
    selected_sr = st.sidebar.selectbox("Sample Rate", sample_rates, index=0)
    
    if selected_sr != processor.sample_rate:
        processor.sample_rate = selected_sr
        st.rerun()
    
    # Main content
    st.header("📁 File Processing")
    
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file to apply auto-tuning"
    )
    
    if uploaded_file is not None:
        st.subheader("🎵 Original Audio")
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🎛️ Process Audio File"):
            with st.spinner("Processing audio file..."):
                try:
                    output_path, processed_audio = processor.process_file(uploaded_file)
                    
                    st.success("Audio processing completed!")
                    
                    st.subheader("🎵 Auto-tuned Audio")
                    st.audio(output_path, format='audio/wav')
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Auto-tuned Audio",
                            data=f.read(),
                            file_name="autotuned_audio.wav",
                            mime="audio/wav"
                        )
                    
                    os.unlink(output_path)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()