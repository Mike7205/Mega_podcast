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

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import requests
import soundfile as sf
import streamlit as st
from pydub import AudioSegment

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except OSError:
    SOUNDDEVICE_OK = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pod Tools – Audio Studio",
    page_icon="🎙️",
    layout="wide",
)

# ─── Session state init ───────────────────────────────────────────────────────
_DEFAULTS = {
    "recorded_audio": None,   # (np.ndarray, int) – original recording
    "working_audio":  None,   # (np.ndarray, int) – current edited version
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ─────────────────────────────────────────────────────────────────
def list_input_devices():
    """Return [(index, name)] for every device that has input channels."""
    if not SOUNDDEVICE_OK:
        return []
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def load_audio_bytes(raw: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Load audio from raw bytes; converts via pydub/ffmpeg when needed."""
    ext = Path(filename).suffix.lower() if filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=None, mono=True)
    finally:
        os.unlink(tmp)
    return y.astype(np.float32), int(sr)


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    start_s: float = 0.0,
    end_s: float | None = None,
) -> plt.Figure:
    dur = len(y) / sr
    if end_s is None:
        end_s = dur
    times = np.linspace(0, dur, num=len(y))

    fig, ax = plt.subplots(figsize=(13, 3), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(times, y, color="#1db954", linewidth=0.35, alpha=0.9)
    ax.axvline(start_s, color="#ff4b4b", linewidth=1.5, label=f"Start  {start_s:.2f}s")
    ax.axvline(end_s,   color="#ffa64b", linewidth=1.5, label=f"End  {end_s:.2f}s")
    ax.set_xlim(0, dur)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.set_ylabel("Amplitude", color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig



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
    st.caption("Uses your browser's microphone — works with any device including FSDZMIC S338.")

    audio_input = st.audio_input("🎤  Click to record")

    if audio_input is not None:
        with st.spinner("Loading recording…"):
            y, sr = load_audio_bytes(audio_input.read(), "recording.wav")
        st.session_state.recorded_audio = (y, sr)
        st.session_state.working_audio  = (y, sr)
        st.success(f"Recorded  {len(y)/sr:.1f} s  @  {sr} Hz")
        st.audio(to_wav_bytes(y, sr), format="audio/wav")

        fname = st.text_input("Save filename", value="recording.wav")
        if st.button("💾  Save to disk", key="save_rec"):
            out = Path(fname)
            sf.write(str(out), y, sr)
            st.success(f"Saved → {out.resolve()}")

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
            st.session_state.working_audio = (y, sr)
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
                            y, sr = librosa.load(str(wav_files[0]), sr=None, mono=True)
                            st.session_state.working_audio = (y.astype(np.float32), int(sr))
                            st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                        else:
                            raise RuntimeError(res.stderr[:300] or "yt-dlp: no output file")
                except Exception as e_yt:
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        ext = url.split(".")[-1].split("?")[0] or "mp3"
                        y, sr = load_audio_bytes(r.content, f"file.{ext}")
                        st.session_state.working_audio = (y, sr)
                        st.success(f"Downloaded  |  {len(y)/sr:.1f}s  @  {sr} Hz")
                    except Exception as e_http:
                        st.error(f"yt-dlp error: {e_yt}\nDirect download error: {e_http}")

    if st.session_state.working_audio is not None:
        y, sr = st.session_state.working_audio
        st.audio(to_wav_bytes(y, sr), format="audio/wav")
        st.caption(f"Duration: {len(y)/sr:.1f}s  |  Sample rate: {sr} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – EDIT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.header("Edit & Export")

    if st.session_state.working_audio is None:
        st.info("No audio loaded yet. Use the **Record** or **Upload** tab first.")
    else:
        y, sr = st.session_state.working_audio
        dur   = len(y) / sr

        # ── Waveform & split markers ──────────────────────────────────────────
        st.subheader("Waveform")
        wc1, wc2 = st.columns(2)
        split_start = wc1.number_input("Segment start (s)", 0.0, float(dur), 0.0, 0.1)
        split_end   = wc2.number_input("Segment end (s)",   0.0, float(dur), float(dur), 0.1)

        fig = plot_waveform(y, sr, "Working Audio", split_start, split_end)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # ── Noise reduction & gain ────────────────────────────────────────────
        st.subheader("Noise Reduction & Amplification")

        nc1, nc2, nc3 = st.columns(3)
        noise_prop = nc1.slider("Noise reduction strength", 0.0, 1.0, 0.5, 0.05)
        gain_db    = nc2.slider("Gain (dB)", -20, 40, 0)
        stationary = nc3.checkbox(
            "Stationary noise model",
            value=True,
            help="Best for constant background hum / hiss",
        )

        if st.button("▶  Apply processing"):
            with st.spinner("Processing… this may take a moment."):
                y_proc = nr.reduce_noise(
                    y=y, sr=sr,
                    prop_decrease=noise_prop,
                    stationary=stationary,
                )
                if gain_db != 0:
                    y_proc = np.clip(y_proc * (10 ** (gain_db / 20)), -1.0, 1.0)
                st.session_state.working_audio = (y_proc.astype(np.float32), sr)
            st.success("Processing applied! Waveform updated.")
            st.rerun()

        if st.button("↩  Reset to original recording"):
            if st.session_state.recorded_audio:
                st.session_state.working_audio = st.session_state.recorded_audio
                st.rerun()
            else:
                st.warning("No original recording in memory – re-upload the file.")

        st.divider()

        # ── Segment preview & save ────────────────────────────────────────────
        st.subheader("Split & Save Segment")

        s1 = max(0, int(split_start * sr))
        s2 = min(len(y), int(split_end * sr))
        segment = y[s1:s2]

        if len(segment) > 0:
            st.audio(to_wav_bytes(segment, sr), format="audio/wav")
            st.caption(
                f"Segment: {split_start:.2f}s → {split_end:.2f}s  "
                f"({split_end - split_start:.2f}s)"
            )

        sc1, sc2 = st.columns(2)
        seg_name   = sc1.text_input("Segment filename", "segment.wav")
        seg_format = sc2.selectbox("Format", ["WAV", "MP3", "FLAC"], key="seg_fmt")

        if st.button("💾  Save segment to disk"):
            if len(segment) == 0:
                st.error("Segment is empty – adjust start / end times.")
            else:
                out_path = Path(seg_name).with_suffix("." + seg_format.lower())
                if seg_format == "WAV":
                    sf.write(str(out_path), segment, sr)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, segment, sr)
                        audio_seg = AudioSegment.from_wav(tmp.name)
                        os.unlink(tmp.name)
                    audio_seg.export(str(out_path), format=seg_format.lower())
                st.success(f"Saved → {out_path.resolve()}")

        st.divider()

        # ── Save full working audio ───────────────────────────────────────────
        st.subheader("Save Full Audio")

        fc1, fc2 = st.columns(2)
        full_name   = fc1.text_input("Output filename", "output.wav")
        full_format = fc2.selectbox("Format", ["WAV", "MP3", "FLAC"], key="full_fmt")

        if st.button("💾  Save full audio to disk"):
            out_path = Path(full_name).with_suffix("." + full_format.lower())
            if full_format == "WAV":
                sf.write(str(out_path), y, sr)
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, y, sr)
                    audio_seg = AudioSegment.from_wav(tmp.name)
                    os.unlink(tmp.name)
                audio_seg.export(str(out_path), format=full_format.lower())
            st.success(f"Saved → {out_path.resolve()}")
