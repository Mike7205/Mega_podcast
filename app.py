"""
Podcast Studio
Streamlit app for recording, uploading, and editing audio.
Optimised for FSDZMIC S338 USB microphone.
"""

import base64
import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import librosa.effects
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

# ─── Video recorder — declared as proper component (iframe survives re-runs) ─
_video_recorder = st.components.v1.declare_component(
    "video_recorder",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Podcast Studio",
    page_icon="🎙️",
    layout="wide",
)

# ─── Wood-grain background styling ───────────────────────────────────────────
st.markdown("""
<style>
/* Spruce wood board background */
.stApp {
    background-color: #6b4423;
    background-image:
        repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent 18px,
            rgba(0,0,0,0.04) 18px,
            rgba(0,0,0,0.04) 20px
        ),
        repeating-linear-gradient(
            180deg,
            transparent 0px,
            transparent 6px,
            rgba(255,255,255,0.015) 6px,
            rgba(255,255,255,0.015) 7px
        ),
        linear-gradient(
            175deg,
            #8b5c2a 0%,
            #7a4f24 15%,
            #6b4320 30%,
            #7d5228 45%,
            #6a3f1c 60%,
            #7b4e26 75%,
            #5e3a18 90%,
            #7a4e25 100%
        );
}

/* Semi-transparent overlay on all main content blocks */
.block-container {
    background: rgba(20, 10, 5, 0.72);
    border-radius: 12px;
    padding: 2rem 2.5rem !important;
    backdrop-filter: blur(2px);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.4);
    border-radius: 8px 8px 0 0;
    padding: 4px 8px 0;
}
.stTabs [data-baseweb="tab"] {
    color: #d4a96a !important;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #f5d7a0 !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px 6px 0 0;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(0,0,0,0.25);
    border-radius: 0 0 8px 8px;
    padding: 1rem;
}

/* Title */
h1 {
    color: #f5d7a0 !important;
    text-shadow: 1px 2px 6px rgba(0,0,0,0.7);
}

/* Metric labels and values */
label, .stSlider label, .stSelectbox label, .stTextInput label {
    color: #e8c990 !important;
}

/* Buttons */
.stButton > button {
    background: rgba(139, 92, 42, 0.85);
    color: #fff5e0;
    border: 1px solid #c4884a;
    border-radius: 6px;
}
.stButton > button:hover {
    background: rgba(180, 120, 60, 0.95);
    border-color: #e0a862;
}

/* Download button */
.stDownloadButton > button {
    background: rgba(60, 100, 60, 0.8);
    color: #d4f0d4;
    border: 1px solid #6aaa6a;
}

/* General text */
p, span, div {
    color: #f0e0c8;
}

/* Slider track */
.stSlider [data-baseweb="slider"] {
    filter: sepia(0.3) hue-rotate(-10deg);
}

/* Expander header text — blue */
.streamlit-expanderHeader,
.streamlit-expanderHeader span,
.streamlit-expanderHeader p,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p,
details > summary,
details > summary span,
details > summary p {
    color: #4a9eff !important;
    font-weight: 600;
}

/* Waveform players: stretch 20% wider using negative margins */
[data-testid="stCustomComponentV1"] {
    width: 120% !important;
    margin-left: -10% !important;
}
[data-testid="stCustomComponentV1"] iframe {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "recorded_audio":   None,   # (np.ndarray, int)
    "uploaded_audio":   None,   # (np.ndarray, int)
    "processed_audio":  None,   # (np.ndarray, int)
    "mic_key":          0,
    "edit_src_override": None,  # force source selection from button
    "_upload_sig":       None,  # (name, size) to avoid re-processing on rerun
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Audio helpers ────────────────────────────────────────────────────────────

def wav_bytes_to_numpy(raw: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes → mono float32 via soundfile (handles 16/24/32-bit)."""
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(sr)


def any_to_wav_bytes(raw: bytes, suffix: str = ".mp3") -> bytes:
    """Convert any audio format to WAV bytes via pydub/ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        seg = AudioSegment.from_file(tmp).set_channels(1)
    finally:
        os.unlink(tmp)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def numpy_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def numpy_to_mp3_bytes(y: np.ndarray, sr: int) -> bytes:
    wav = numpy_to_wav_bytes(y, sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        tmp = f.name
    try:
        seg = AudioSegment.from_wav(tmp)
    finally:
        os.unlink(tmp)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate="192k")
    return buf.getvalue()


def encode_for_download(y: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    wav = numpy_to_wav_bytes(y, sr)
    if fmt == "WAV":
        return wav, "audio/wav"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        tmp = f.name
    try:
        seg = AudioSegment.from_wav(tmp)
    finally:
        os.unlink(tmp)
    buf = io.BytesIO()
    if fmt == "MP4":
        seg.export(buf, format="mp4", codec="aac")
        return buf.getvalue(), "audio/mp4"
    seg.export(buf, format=fmt.lower())
    return buf.getvalue(), {"MP3": "audio/mpeg", "FLAC": "audio/flac"}[fmt]


# ─── Waveform plot ────────────────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int, title: str = "",
                  start_s: float = 0.0, end_s: float | None = None,
                  color: str = "#1db954") -> plt.Figure:
    dur = len(y) / sr
    if end_s is None:
        end_s = dur

    n_bars = 600
    step   = max(1, len(y) // n_bars)
    peaks  = np.array([np.max(np.abs(y[i: i + step]))
                       for i in range(0, len(y) - step, step)])
    t      = np.linspace(0, dur, len(peaks))

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.bar(t,  peaks, width=dur / len(peaks) * 0.85, color=color, alpha=0.9)
    ax.bar(t, -peaks, width=dur / len(peaks) * 0.85, color=color, alpha=0.9)
    ax.axvline(start_s, color="#ff4b4b", linewidth=1.5, label=f"Start {start_s:.2f}s")
    ax.axvline(end_s,   color="#ffa64b", linewidth=1.5, label=f"End   {end_s:.2f}s")
    ax.set_xlim(0, dur)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.set_ylabel("Amplitude", color="white", fontsize=9)
    if title:
        ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a2a")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    fig.tight_layout(pad=0.4)
    return fig


def show_player(y: np.ndarray, sr: int, title: str = "",
                start_s: float = 0.0, end_s: float | None = None,
                color: str = "#1db954") -> None:
    """Canvas waveform with moving red playhead + clickable seek + audio controls."""
    dur = len(y) / sr
    if end_s is None:
        end_s = dur

    # Downsample to 600 bars, normalise to [0, 1]
    n_bars = 600
    step   = max(1, len(y) // n_bars)
    peaks  = [float(np.max(np.abs(y[i: i + step]))) for i in range(0, len(y) - step, step)]
    max_p  = max(peaks) if peaks else 1.0
    if max_p > 0:
        peaks = [p / max_p for p in peaks]

    wav_b64    = base64.b64encode(numpy_to_wav_bytes(y, sr)).decode()
    uid        = abs(hash((title, len(y), id(y))))
    start_frac = start_s / dur if dur > 0 else 0.0
    end_frac   = end_s   / dur if dur > 0 else 1.0

    title_html = (f"<div style='color:{color};font-size:15px;font-weight:600;"
                  f"margin-bottom:6px;'>{title}</div>") if title else ""

    html = f"""
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
</style>
<div style="background:#0e1117;border-radius:8px;padding:10px 12px;box-sizing:border-box;width:100%;">
  {title_html}
  <canvas id="cv{uid}"
    style="width:100%;height:120px;display:block;cursor:pointer;border-radius:4px;">
  </canvas>
  <audio id="au{uid}" src="data:audio/wav;base64,{wav_b64}"
    style="width:100%;margin-top:6px;" controls></audio>
</div>
<script>
(function(){{
  const cv  = document.getElementById("cv{uid}");
  const au  = document.getElementById("au{uid}");
  const peaks = {peaks};
  const color = "{color}";
  const startFrac = {start_frac};
  const endFrac   = {end_frac};

  function draw(progress) {{
    const W = cv.width, H = cv.height;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    const n = peaks.length;
    const barW = W / n;
    const playX = progress * W;

    for (let i = 0; i < n; i++) {{
      const x  = i * barW;
      const h  = peaks[i] * H * 0.88;
      const y0 = (H - h) / 2;
      ctx.fillStyle = (x < playX) ? "rgba(255,255,255,0.40)" : color;
      ctx.fillRect(x + 0.3, y0, Math.max(0.8, barW - 0.6), h);
    }}

    // Start / End trim markers
    function drawMarker(x, col, label) {{
      // solid line
      ctx.setLineDash([]);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = col;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      // top handle triangle
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(x - 7, 0); ctx.lineTo(x + 7, 0); ctx.lineTo(x, 10); ctx.closePath(); ctx.fill();
      // bottom handle triangle
      ctx.beginPath(); ctx.moveTo(x - 7, H); ctx.lineTo(x + 7, H); ctx.lineTo(x, H - 10); ctx.closePath(); ctx.fill();
      // label
      ctx.font = "bold 10px sans-serif";
      ctx.fillStyle = col;
      const lx = (x + 6 + 36 < W) ? x + 6 : x - 42;
      ctx.fillText(label, lx, 14);
    }}
    drawMarker(startFrac * W, "#ff4b4b", "start");
    drawMarker(endFrac   * W, "#ffa64b", "end");

    // Moving red playhead
    if (progress > 0 && progress <= 1) {{
      ctx.strokeStyle = "#ff2222";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(playX, 0); ctx.lineTo(playX, H); ctx.stroke();
    }}
  }}

  function resize() {{
    cv.width  = cv.offsetWidth || 900;
    cv.height = 120;
    draw(au.duration ? au.currentTime / au.duration : 0);
  }}

  resize();
  window.addEventListener("resize", resize);
  au.addEventListener("timeupdate",    () => draw(au.currentTime / au.duration));
  au.addEventListener("ended",         () => draw(1));
  au.addEventListener("loadedmetadata",() => draw(0));

  cv.addEventListener("click", function(e) {{
    if (!au.duration) return;
    const frac = (e.clientX - cv.getBoundingClientRect().left) / cv.offsetWidth;
    au.currentTime = frac * au.duration;
    draw(frac);
  }});
}})();
</script>
"""
    st.components.v1.html(html, height=1450)


# ─── Processing ───────────────────────────────────────────────────────────────

def _shelf(y: np.ndarray, sr: int, freq: float, gain_db: float, btype: str) -> np.ndarray:
    """Low or high shelf filter via bandpass mix."""
    if gain_db == 0:
        return y
    nyq = sr / 2.0
    f = np.clip(freq / nyq, 0.001, 0.999)
    b, a = butter(2, f, btype=btype)
    band = filtfilt(b, a, y).astype(np.float32)
    return (y + band * (10 ** (gain_db / 20) - 1)).astype(np.float32)


def _peak(y: np.ndarray, sr: int, lo: float, hi: float, gain_db: float) -> np.ndarray:
    """Peaking EQ band via bandpass mix."""
    if gain_db == 0:
        return y
    nyq = sr / 2.0
    lo_f = np.clip(lo / nyq, 0.001, 0.999)
    hi_f = np.clip(hi / nyq, 0.001, 0.999)
    if lo_f >= hi_f:
        return y
    b, a = butter(2, [lo_f, hi_f], btype="band")
    band = filtfilt(b, a, y).astype(np.float32)
    return (y + band * (10 ** (gain_db / 20) - 1)).astype(np.float32)


def _compress(y: np.ndarray, sr: int,
              threshold_db: float, ratio: float,
              attack_ms: float, release_ms: float) -> np.ndarray:
    """Chunk-based feed-forward RMS compressor."""
    if ratio <= 1.0:
        return y
    threshold = 10 ** (threshold_db / 20)
    chunk = max(64, int(sr * 0.005))          # ~5 ms chunks
    att_k  = 1.0 - np.exp(-1.0 / max(1, sr * attack_ms  / 1000))
    rel_k  = 1.0 - np.exp(-1.0 / max(1, sr * release_ms / 1000))
    out   = y.copy()
    gain  = 1.0
    for i in range(0, len(y), chunk):
        seg = y[i:i + chunk]
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        if rms > threshold:
            target = (threshold * (rms / threshold) ** (1.0 / ratio)) / rms
        else:
            target = 1.0
        k = att_k if target < gain else rel_k
        gain += k * (target - gain)
        out[i:i + chunk] = (seg * gain).astype(np.float32)
    return out.astype(np.float32)


def _deess(y: np.ndarray, sr: int, amount: float, freq_hz: float) -> np.ndarray:
    """Broadband de-esser: attenuates sibilants above freq_hz."""
    if amount <= 0:
        return y
    nyq = sr / 2.0
    f = np.clip(freq_hz / nyq, 0.001, 0.999)
    b, a = butter(4, f, btype="high")
    sib = filtfilt(b, a, y).astype(np.float32)
    return (y - sib * np.clip(amount, 0, 1) * 0.85).astype(np.float32)


def apply_processing(y: np.ndarray, sr: int,
                     noise_prop: float, stationary: bool,
                     gain_db: float, hp_cutoff: int,
                     vocal_clarity: float,
                     pitch_steps: int, speed: float,
                     eq_low: float = 0, eq_mid: float = 0, eq_high: float = 0,
                     comp_thresh: float = -40, comp_ratio: float = 1.0,
                     comp_attack: float = 10, comp_release: float = 100,
                     deess_amount: float = 0, deess_freq: float = 6000,
                     limit_enabled: bool = False, limit_ceil: float = -1) -> np.ndarray:
    out = y.copy()

    # 1. High-pass (remove rumble)
    if hp_cutoff > 0:
        b, a = butter(4, hp_cutoff / (sr / 2), btype="high")
        out  = filtfilt(b, a, out).astype(np.float32)

    # 2. Noise reduction
    if noise_prop > 0:
        out = nr.reduce_noise(y=out, sr=sr,
                              prop_decrease=noise_prop,
                              stationary=stationary).astype(np.float32)

    # 3. EQ – 3-band (low shelf / mid peak / high shelf)
    out = _shelf(out, sr, 200,  eq_low,  "low")
    out = _peak (out, sr, 500, 4000, eq_mid)
    out = _shelf(out, sr, 6000, eq_high, "high")

    # 4. Vocal clarity (existing)
    if vocal_clarity > 0:
        lo, hi = 200, 4000
        b2, a2 = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        mid = filtfilt(b2, a2, out).astype(np.float32)
        out = ((1 - vocal_clarity) * out + vocal_clarity * mid).astype(np.float32)

    # 5. Compressor
    out = _compress(out, sr, comp_thresh, comp_ratio, comp_attack, comp_release)

    # 6. De-esser
    out = _deess(out, sr, deess_amount, deess_freq)

    # 7. Pitch / speed
    if pitch_steps != 0:
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=float(pitch_steps))
    if speed != 1.0:
        out = librosa.effects.time_stretch(out, rate=speed)

    # 8. Gain
    if gain_db != 0:
        out = (out * (10 ** (gain_db / 20))).astype(np.float32)

    # 9. Limiter (last in chain)
    if limit_enabled:
        ceil = 10 ** (limit_ceil / 20)
        out = (np.tanh(out / ceil) * ceil).astype(np.float32)
    else:
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

    return out.astype(np.float32)


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🎙️  Podcast Studio")

tab_rec, tab_edit = st.tabs(
    ["⏺  Record", "✂️  Edit & Export"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – RECORD
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.header("Record New Audio")
    st.caption("Works with FSDZMIC S338 and any other device.")

    rec_mode = st.radio(
        "Recording mode",
        ["🎤  Audio only", "🎥  Audio + Video"],
        horizontal=True,
        help="Audio only — saves to session for editing in Edit & Export tab.\n"
             "Audio + Video — records camera + microphone, download directly from browser.",
    )

    if rec_mode == "🎤  Audio only":
        if st.button("🔄  Reset microphone", help="Click if the recorder shows an error"):
            st.session_state.mic_key += 1
            st.rerun()

        audio_input = st.audio_input("🎤  Click to record",
                                     key=f"mic_{st.session_state.mic_key}")

        if audio_input is not None:
            raw_wav = audio_input.read()

            with st.spinner("Loading…"):
                y, sr = wav_bytes_to_numpy(raw_wav)

            st.session_state.recorded_audio = (y, sr)
            st.success(f"Recorded  {len(y)/sr:.1f}s  |  {sr} Hz")

            show_player(y, sr, "Recording preview")

            st.divider()
            fname = st.text_input("Save filename", value="recording.mp4")
            dl_bytes, dl_mime = encode_for_download(y, sr, "MP4")
            dl_name = str(Path(fname).with_suffix(".mp4"))
            st.download_button("💾  Save as MP4", dl_bytes, dl_name, dl_mime, key="dl_rec")

    else:
        st.info(
            "Nagrywa kamerę + mikrofon bezpośrednio w przeglądarce. "
            "Po zakończeniu pobierz plik MP4/WebM przyciskiem poniżej.",
            icon="🎥",
        )

        # Okienko nagrywania — 90% szerokości strony, wyśrodkowane
        col1, col2, col3 = st.columns([1, 9, 1])
        with col2:
            st.components.v1.html(
                _VIDEO_REC_HTML,     # ← Twój duży string z HTML + JS
                height=720,          # zwiększona wysokość
                scrolling=True
            )

        # Komunikat po zakończeniu nagrywania (tymczasowy, dopóki nie przejdziesz na declare_component)
        st.caption("✅ Po nagraniu plik powinien pojawić się do pobrania w przeglądarce.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDIT & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.header("Edit & Export")

    # Source selector
    # ── Quick file loader ──────────────────────────────────────────────────────
    with st.expander("📂  Load file for editing", expanded=st.session_state.uploaded_audio is None):
        qf_col, qu_col = st.columns(2)
        with qf_col:
            qfile = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3", "m4a", "flac", "ogg", "mp4"],
                key="edit_file_uploader",
                label_visibility="collapsed",
            )
            if qfile is not None:
                file_sig = (qfile.name, qfile.size)
                if st.session_state._upload_sig != file_sig:
                    with st.spinner("Loading…"):
                        ext = Path(qfile.name).suffix.lower()
                        raw = qfile.read()
                        if ext in (".wav",):
                            y_q, sr_q = wav_bytes_to_numpy(raw)
                        else:
                            wav_q = any_to_wav_bytes(raw, suffix=ext)
                            y_q, sr_q = wav_bytes_to_numpy(wav_q)
                    st.session_state.uploaded_audio = (y_q, sr_q)
                    st.session_state.edit_src_override = "Uploaded file"
                    st.session_state._upload_sig = file_sig
                    st.rerun()
                else:
                    y_d, sr_d = st.session_state.uploaded_audio
                    st.success(f"{qfile.name}  |  {len(y_d)/sr_d:.1f}s  @  {sr_d} Hz")
        with qu_col:
            qu_url = st.text_input("…or paste URL / YouTube link",
                                   key="edit_url_input",
                                   label_visibility="collapsed",
                                   placeholder="https://… or YouTube URL")
            if st.button("⬇  Download", key="edit_dl_btn"):
                if qu_url.strip():
                    with st.spinner("Downloading…"):
                        try:
                            with tempfile.TemporaryDirectory() as td:
                                subprocess.run(
                                    ["yt-dlp", "-x", "--audio-format", "wav",
                                     "-o", str(Path(td) / "out.%(ext)s"), qu_url],
                                    check=True, capture_output=True,
                                )
                                wav_files = list(Path(td).glob("*.wav"))
                                if not wav_files:
                                    raise RuntimeError("No output file")
                                y_q, sr_q = wav_bytes_to_numpy(wav_files[0].read_bytes())
                        except Exception:
                            r = requests.get(qu_url, timeout=30)
                            r.raise_for_status()
                            ext2 = "." + (qu_url.split(".")[-1].split("?")[0] or "mp3")
                            y_q, sr_q = wav_bytes_to_numpy(any_to_wav_bytes(r.content, suffix=ext2))
                    st.session_state.uploaded_audio = (y_q, sr_q)
                    st.session_state.edit_src_override = "Uploaded file"
                    st.rerun()

    # ── Source selector ────────────────────────────────────────────────────────
    sources: dict[str, tuple[np.ndarray, int]] = {}
    if st.session_state.recorded_audio is not None:
        sources["My recording"] = st.session_state.recorded_audio
    if st.session_state.uploaded_audio is not None:
        sources["Uploaded file"]  = st.session_state.uploaded_audio

    if not sources:
        st.info("No audio available. Record something or load a file above.")
        st.stop()

    src_keys = list(sources.keys())
    if st.session_state.edit_src_override in src_keys:
        default_idx = src_keys.index(st.session_state.edit_src_override)
        st.session_state.edit_src_override = None
    else:
        default_idx = 0

    chosen_src = st.radio("Audio source", src_keys, index=default_idx, horizontal=True)

    y_orig, sr = sources[chosen_src]
    dur = len(y_orig) / sr

    st.divider()

    # Waveform player
    st.subheader("Waveform")
    if st.session_state.processed_audio is not None:
        y_proc, _ = st.session_state.processed_audio
        st.markdown("<span style='font-size:18px; color:#4a9eff; font-weight:600'>Original</span>", unsafe_allow_html=True)
        show_player(y_orig, sr, color="#1db954")
        st.markdown("<span style='font-size:18px; color:#ff4b4b; font-weight:600'>Processed</span>", unsafe_allow_html=True)
        show_player(y_proc, sr, color="#4a9eff")
        y_work = y_proc
    else:
        show_player(y_orig, sr, "Working audio")
        y_work = y_orig

    st.divider()

    # Processing
    st.subheader("Processing")

    with st.form("processing_form"):
        with st.expander("🔇  Noise Reduction", expanded=True):
            c1, c2, c3 = st.columns(3)
            noise_prop = c1.slider("Noise reduction", 0.0, 1.0, 0.5, 0.05)
            gain_db    = c2.slider("Gain (dB)", -20, 40, 0)
            stationary = c3.checkbox("Stationary noise", value=True,
                                     help="Best for constant hum/hiss")

        with st.expander("🎙️  Vocal Enhancement", expanded=True):
            c1, c2 = st.columns(2)
            vocal_clarity = c1.slider("Vocal clarity", 0.0, 1.0, 0.0, 0.05,
                                      help="Boosts 200–4000 Hz voice range")
            hp_cutoff     = c2.slider("Low-cut filter (Hz)", 0, 500, 80, 10,
                                      help="Removes rumble below this frequency")

        with st.expander("🎛️  Equalizer (EQ)", expanded=False):
            c1, c2, c3 = st.columns(3)
            eq_low  = c1.slider("Low shelf 200 Hz (dB)", -12, 12, 0,
                                help="Boost/cut bass & warmth")
            eq_mid  = c2.slider("Mid 500–4k Hz (dB)", -12, 12, 0,
                                help="Boost/cut vocal presence")
            eq_high = c3.slider("High shelf 6 kHz (dB)", -12, 12, 0,
                                help="Boost/cut air & brightness")

        with st.expander("🗜️  Compressor", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            comp_thresh  = c1.slider("Threshold (dBFS)", -40, 0, -40,
                                     help="Level where compression starts · -40 = off")
            comp_ratio   = c2.slider("Ratio", 1.0, 10.0, 1.0, 0.5,
                                     help="1:1 = off · 2:1 gentle · 8:1 heavy")
            comp_attack  = c3.slider("Attack (ms)", 1, 100, 10,
                                     help="How fast compression kicks in")
            comp_release = c4.slider("Release (ms)", 10, 500, 100,
                                     help="How fast compression releases")

        with st.expander("🔉  De-esser", expanded=False):
            c1, c2 = st.columns(2)
            deess_amount = c1.slider("De-ess amount", 0.0, 1.0, 0.0, 0.05,
                                     help="Reduces harsh 's', 'sz', 'cz' sibilants · 0 = off")
            deess_freq   = c2.slider("Sibilant threshold (Hz)", 4000, 10000, 6000, 500,
                                     help="Attenuates frequencies above this value")

        with st.expander("📊  Limiter", expanded=False):
            c1, c2 = st.columns(2)
            limit_enabled = c1.checkbox("Enable limiter", value=False,
                                        help="Soft-clip at the end of the chain — prevents clipping")
            limit_ceil    = c2.slider("Ceiling (dBFS)", -6, 0, -1,
                                      help="Maximum output level",
                                      disabled=False)

        with st.expander("🎚️  Voice Modulation", expanded=False):
            c1, c2 = st.columns(2)
            pitch_steps = c1.slider("Pitch shift (semitones)", -12, 12, 0,
                                    help="±2 subtle · ±12 = one octave")
            speed       = c2.slider("Speed ×", 0.5, 2.0, 1.0, 0.05,
                                    help="< 1 slower · > 1 faster")

        fc1, fc2 = st.columns(2)
        submitted = fc1.form_submit_button("▶  Apply & compare",
                                           use_container_width=True,
                                           type="primary")
        reset     = fc2.form_submit_button("↩  Reset",
                                           use_container_width=True)

    if submitted:
        with st.spinner("Processing…"):
            y_proc = apply_processing(
                y_orig, sr,
                noise_prop, stationary, gain_db, hp_cutoff,
                vocal_clarity, pitch_steps, speed,
                eq_low=eq_low, eq_mid=eq_mid, eq_high=eq_high,
                comp_thresh=comp_thresh, comp_ratio=comp_ratio,
                comp_attack=comp_attack, comp_release=comp_release,
                deess_amount=deess_amount, deess_freq=deess_freq,
                limit_enabled=limit_enabled, limit_ceil=limit_ceil,
            )
            st.session_state.processed_audio = (y_proc, sr)
        st.rerun()

    if reset:
        st.session_state.processed_audio = None
        st.rerun()

    st.divider()

    # Split & Save
    st.subheader("Split & Save")

    if st.session_state.processed_audio is None:
        st.info("⚙️  Apply processing first — Split & Save works only on the processed audio.")
    else:
        y_split, sr_split = st.session_state.processed_audio
        dur_split = float(len(y_split) / sr_split)

        seg_range = st.slider(
            "Drag handles to set segment start / end",
            min_value=0.0,
            max_value=dur_split,
            value=(0.0, dur_split),
            step=0.01,
            format="%.2f s",
            key="split_range",
        )
        split_start, split_end = float(seg_range[0]), float(seg_range[1])

        show_player(
            y_split, sr_split,
            f"Segment  {split_start:.2f}s → {split_end:.2f}s",
            split_start, split_end,
            color="#4a9eff",
        )

        s1 = max(0, int(split_start * sr_split))
        s2 = min(len(y_split), int(split_end * sr_split))
        segment = y_split[s1:s2]

        if len(segment) > 0:
            c1, c2 = st.columns(2)
            seg_name = c1.text_input("Segment filename", "segment.mp4", key="seg_name_split")
            seg_fmt  = c2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="seg_fmt")
            seg_bytes, seg_mime = encode_for_download(segment, sr_split, seg_fmt)
            seg_fname = str(Path(seg_name).with_suffix("." + seg_fmt.lower()))
            st.download_button("💾  Save segment", seg_bytes, seg_fname, seg_mime, key="dl_seg")
        else:
            st.warning("Segment is empty — adjust the handles.")

    st.divider()

    # Save full
    st.subheader("Save Full Audio")
    c1, c2 = st.columns(2)
    full_name = c1.text_input("Output filename", "output.mp4")
    full_fmt  = c2.selectbox("Format", ["MP4", "MP3", "WAV", "FLAC"], key="full_fmt")
    full_bytes, full_mime = encode_for_download(y_work, sr, full_fmt)
    full_fname = str(Path(full_name).with_suffix("." + full_fmt.lower()))
    st.download_button("💾  Save full audio", full_bytes, full_fname, full_mime, key="dl_full")
