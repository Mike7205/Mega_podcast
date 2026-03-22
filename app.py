"""
Pod Tools – Audio Studio
Streamlit app for recording, uploading, and editing audio.
Optimised for FSDZMIC S338 USB microphone.
"""

import io
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import requests
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pod Tools – Audio Studio",
    page_icon="🎙️",
    layout="wide",
)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "recorded_audio":  None,
    "uploaded_audio":  None,
    "processed_audio": None,
    "mic_key":         0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ─────────────────────────────────────────────────────────────────
def seg_to_numpy(seg: AudioSegment) -> tuple[np.ndarray, int]:
    """Convert pydub AudioSegment → mono float32 numpy array."""
    seg = seg.set_channels(1)
    sr  = seg.frame_rate
    raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32)
    raw /= 32768.0
    return raw, sr


def numpy_to_seg(y: np.ndarray, sr: int) -> AudioSegment:
    """Convert mono float32 numpy array → pydub AudioSegment."""
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)


def load_audio_bytes(raw: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Load any audio format → mono float32 numpy, preserving sample rate."""
    ext = Path(filename).suffix.lower() if filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        seg = AudioSegment.from_file(tmp)
    finally:
        os.unlink(tmp)
    return seg_to_numpy(seg)


def to_mp3_bytes(y: np.ndarray, sr: int, bitrate: str = "192k") -> bytes:
    buf = io.BytesIO()
    numpy_to_seg(y, sr).export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def encode_audio(y: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    seg = numpy_to_seg(y, sr)
    buf = io.BytesIO()
    if fmt == "WAV":
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue(), "audio/wav"
    if fmt == "MP4":
        seg.export(buf, format="mp4", codec="aac")
        return buf.getvalue(), "audio/mp4"
    seg.export(buf, format=fmt.lower())
    mime = {"MP3": "audio/mpeg", "FLAC": "audio/flac"}.get(fmt, "audio/octet-stream")
    return buf.getvalue(), mime


def plot_waveform(y: np.ndarray, sr: int, title: str = "",
                  start_s: float = 0.0, end_s: float | None = None) -> plt.Figure:
    dur = len(y) / sr
    if end_s is None:
        end_s = dur
    n    = min(len(y), 8000)
    times = np.linspace(0, dur, n)
    y_ds  = np.interp(times, np.linspace(0, dur, len(y)), y)

    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.fill_between(times, y_ds, color="#1db954", alpha=0.85, linewidth=0)
    ax.axvline(start_s, color="#ff4b4b", linewidth=1.5, label=f"Start {start_s:.2f}s")
    ax.axvline(end_s,   color="#ffa64b", linewidth=1.5, label=f"End {end_s:.2f}s")
    ax.set_xlim(0, dur)
    ax.set_xlabel("Time (s)", color="white", fontsize=8)
    ax.set_ylabel("Amp", color="white", fontsize=8)
    if title:
        ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig


def pitch_shift_pydub(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch via frame-rate trick (also shifts speed slightly)."""
    seg     = numpy_to_seg(y, sr)
    factor  = 2 ** (semitones / 12)
    shifted = seg._spawn(seg.raw_data, overrides={"frame_rate": int(sr * factor)})
    shifted = shifted.set_frame_rate(sr)
    result, _ = seg_to_numpy(shifted)
    return result


def time_stretch_pydub(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Speed up / slow down via frame-rate trick, preserving pitch."""
    seg       = numpy_to_seg(y, sr)
    new_rate  = int(sr / rate)
    stretched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    stretched = stretched.set_frame_rate(sr)
    result, _ = seg_to_numpy(stretched)
    return result


def vocal_enhance(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Boost mid-range vocal frequencies via bandpass blend."""
    if strength == 0:
        return y
    lo, hi = 200, 4000
    b_hp, a_hp = butter(4, lo / (sr / 2), btype="high")
    b_lp, a_lp = butter(4, hi / (sr / 2), btype="low")
    y_mid = filtfilt(b_lp, a_lp, filtfilt(b_hp, a_hp, y))
    return ((1 - strength) * y + strength * y_mid).astype(np.float32)


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🎙️  Pod Tools – Audio Studio")

tab_rec, tab_upload, tab_edit = st.tabs(
    ["⏺  Record", "⬆️  Upload", "✂️  Edit & Export"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – RECORD
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.header("Record New Audio")
    st.caption("Uses your browser's microphone — works with FSDZMIC S338 and any other device.")

    if st.button("🔄  Reset microphone", help="Use if recorder shows an error"):
        st.session_state.mic_key += 1
        st.rerun()

    audio_input = st.audio_input("🎤  Click to record", key=f"mic_{st.session_state.mic_key}")

    if audio_input is not None:
        raw_bytes = audio_input.read()

        with st.spinner("Converting…"):
            seg = AudioSegment.from_file(io.BytesIO(raw_bytes))

            mp3_buf = io.BytesIO()
            seg.export(mp3_buf, format="mp3", bitrate="192k")
            mp3_bytes = mp3_buf.getvalue()

            mp4_buf = io.BytesIO()
            seg.export(mp4_buf, format="mp4", codec="aac")
            mp4_bytes = mp4_buf.getvalue()

            y, sr = seg_to_numpy(seg)

        st.session_state.recorded_audio = (y, sr)
        st.success(f"Recorded  {len(y)/sr:.1f}s  |  {sr} Hz  →  go to **Edit & Export** to process")

        fig = plot_waveform(y, sr, "Recording preview")
        st.pyplot(fig)
        plt.close(fig)
        st.audio(mp3_bytes, format="audio/mp3")

        fname = st.text_input("Filename", value="recording.mp4")
        st.download_button("💾  Save as MP4", mp4_bytes, fname, "audio/mp4", key="dl_rec")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("Upload Audio File")

    src = st.radio("Source", ["Local file", "URL / YouTube"], horizontal=True)

    if src == "Local file":
        uploaded = st.file_uploader(
            "Choose a file",
            type=["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac"],
        )
        if uploaded and st.button("Load file"):
            with st.spinner("Loading…"):
                y, sr = load_audio_bytes(uploaded.read(), uploaded.name)
            st.session_state.uploaded_audio = (y, sr)
            st.success(f"Loaded: {uploaded.name}  |  {len(y)/sr:.1f}s  @  {sr} Hz")

    else:
        url = st.text_input("Paste a direct audio URL or YouTube / SoundCloud link")
        if url and st.button("Download & Load"):
            with st.spinner("Downloading…"):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        out_tmpl = os.path.join(tmp, "audio.%(ext)s")
                        res = subprocess.run(
                            ["yt-dlp", "-x", "--audio-format", "wav", "-o", out_tmpl, url],
                            capture_output=True, text=True, timeout=180,
                        )
                        wav_files = list(Path(tmp).glob("*.wav"))
                        if wav_files:
                            y, sr = load_audio_bytes(wav_files[0].read_bytes(), ".wav")
                            st.session_state.uploaded_audio = (y, sr)
                            st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                        else:
                            raise RuntimeError(res.stderr[:300] or "yt-dlp: no output file")
                except Exception as e_yt:
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        ext = url.split(".")[-1].split("?")[0] or "mp3"
                        y, sr = load_audio_bytes(r.content, f"file.{ext}")
                        st.session_state.uploaded_audio = (y, sr)
                        st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                    except Exception as e_http:
                        st.error(f"yt-dlp: {e_yt}\nHTTP: {e_http}")

    if st.session_state.uploaded_audio is not None:
        y, sr = st.session_state.uploaded_audio
        fig = plot_waveform(y, sr, "Uploaded file")
        st.pyplot(fig)
        plt.close(fig)
        st.audio(to_mp3_bytes(y, sr), format="audio/mp3")
        st.caption(f"Duration: {len(y)/sr:.1f}s  |  {sr} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – EDIT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.header("Edit & Export")

    sources = {}
    if st.session_state.recorded_audio is not None:
        sources["Last recording"] = st.session_state.recorded_audio
    if st.session_state.uploaded_audio is not None:
        sources["Uploaded file"] = st.session_state.uploaded_audio

    if not sources:
        st.info("No audio available. Use the **Record** or **Upload** tab first.")
        st.stop()

    chosen_src = st.radio("Audio source", list(sources.keys()), horizontal=True)
    y_orig, sr = sources[chosen_src]
    dur = len(y_orig) / sr

    st.divider()

    # ── Waveform + player ─────────────────────────────────────────────────────
    st.subheader("Waveform")

    if st.session_state.processed_audio is not None:
        y_proc_arr, _ = st.session_state.processed_audio
        st.caption("**Original**")
        fig = plot_waveform(y_orig, sr)
        st.pyplot(fig); plt.close(fig)
        st.audio(to_mp3_bytes(y_orig, sr), format="audio/mp3")
        st.caption("**Processed**")
        fig = plot_waveform(y_proc_arr, sr)
        st.pyplot(fig); plt.close(fig)
        st.audio(to_mp3_bytes(y_proc_arr, sr), format="audio/mp3")
        y_work = y_proc_arr
    else:
        fig = plot_waveform(y_orig, sr, "Working audio")
        st.pyplot(fig); plt.close(fig)
        st.audio(to_mp3_bytes(y_orig, sr), format="audio/mp3")
        y_work = y_orig

    st.divider()

    # ── Processing ────────────────────────────────────────────────────────────
    st.subheader("Processing")

    with st.expander("🔇  Noise Reduction", expanded=True):
        nc1, nc2, nc3 = st.columns(3)
        noise_prop = nc1.slider("Noise reduction strength", 0.0, 1.0, 0.5, 0.05)
        gain_db    = nc2.slider("Gain (dB)", -20, 40, 0)
        stationary = nc3.checkbox("Stationary noise", value=True,
                                  help="Best for constant hum/hiss (fan, AC)")

    with st.expander("🎙️  Vocal Enhancement", expanded=True):
        vc1, vc2 = st.columns(2)
        vocal_clarity = vc1.slider(
            "Vocal clarity", 0.0, 1.0, 0.0, 0.05,
            help="Boosts 200–4000 Hz (voice range), reduces lows and highs.",
        )
        hp_cutoff = vc2.slider(
            "Low-cut filter (Hz)", 0, 500, 80, 10,
            help="Removes rumble below this frequency. 80–120 Hz safe for voice.",
        )

    with st.expander("🎚️  Voice Modulation", expanded=True):
        vm1, vm2 = st.columns(2)
        pitch_steps = vm1.slider(
            "Pitch shift (semitones)", -12, 12, 0,
            help="Shifts pitch up (+) or down (−). ±2 subtle, ±12 = one octave.",
        )
        time_rate = vm2.slider(
            "Speed ×", 0.5, 2.0, 1.0, 0.05,
            help="< 1.0 = slower, > 1.0 = faster.",
        )

    bc1, bc2 = st.columns(2)
    if bc1.button("▶  Apply & compare"):
        with st.spinner("Processing…"):
            y_proc = y_orig.copy()

            # 1. Low-cut filter
            if hp_cutoff > 0:
                b, a = butter(4, hp_cutoff / (sr / 2), btype="high")
                y_proc = filtfilt(b, a, y_proc).astype(np.float32)

            # 2. Noise reduction
            if noise_prop > 0:
                y_proc = nr.reduce_noise(y=y_proc, sr=sr,
                                         prop_decrease=noise_prop,
                                         stationary=stationary)

            # 3. Vocal clarity
            if vocal_clarity > 0:
                y_proc = vocal_enhance(y_proc, sr, vocal_clarity)

            # 4. Gain
            if gain_db != 0:
                y_proc = np.clip(y_proc * (10 ** (gain_db / 20)), -1.0, 1.0).astype(np.float32)

            # 5. Pitch shift
            if pitch_steps != 0:
                y_proc = pitch_shift_pydub(y_proc, sr, pitch_steps)

            # 6. Time stretch
            if time_rate != 1.0:
                y_proc = time_stretch_pydub(y_proc, sr, time_rate)

            st.session_state.processed_audio = (y_proc.astype(np.float32), sr)
        st.rerun()

    if bc2.button("↩  Reset"):
        st.session_state.processed_audio = None
        st.rerun()

    st.divider()

    # ── Split & save ──────────────────────────────────────────────────────────
    st.subheader("Split & Save")

    wc1, wc2 = st.columns(2)
    split_start = wc1.number_input("Segment start (s)", 0.0, float(dur), 0.0, 0.1)
    split_end   = wc2.number_input("Segment end (s)",   0.0, float(dur), float(dur), 0.1)

    s1 = max(0, int(split_start * sr))
    s2 = min(len(y_work), int(split_end * sr))
    segment = y_work[s1:s2]

    if len(segment) > 0:
        fig = plot_waveform(segment, sr, f"Segment  {split_start:.2f}s → {split_end:.2f}s")
        st.pyplot(fig); plt.close(fig)
        st.audio(to_mp3_bytes(segment, sr), format="audio/mp3")

        sc1, sc2 = st.columns(2)
        seg_name   = sc1.text_input("Segment filename", "segment.mp4")
        seg_format = sc2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="seg_fmt")

        seg_bytes, seg_mime = encode_audio(segment, sr, seg_format)
        seg_fname = str(Path(seg_name).with_suffix("." + seg_format.lower()))
        st.download_button("💾  Save segment", seg_bytes, seg_fname, seg_mime, key="dl_seg")
    else:
        st.warning("Segment is empty – adjust start / end.")

    st.divider()

    # ── Save full audio ───────────────────────────────────────────────────────
    st.subheader("Save Full Audio")

    fc1, fc2 = st.columns(2)
    full_name   = fc1.text_input("Output filename", "output.mp4")
    full_format = fc2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="full_fmt")

    full_bytes, full_mime = encode_audio(y_work, sr, full_format)
    full_fname = str(Path(full_name).with_suffix("." + full_format.lower()))
    st.download_button("💾  Save full audio", full_bytes, full_fname, full_mime, key="dl_full")
