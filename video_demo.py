import csv
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.title("Cattle‐Visibility Segments")

# Base directory that contains all processed videos (tracking, identification, ...)
processed_root = Path(__file__).parent / "data" / "HF_dataset" / "processed_videos"

# Let the user choose which processed subfolder to browse (e.g. "tracking", "identification")
subfolders = [d for d in processed_root.iterdir() if d.is_dir()]
if not subfolders:
    st.error("No subfolders found under data/HF_dataset/processed_videos.")
    st.stop()

selected_folder: Path = st.selectbox(
    "Select processed data type", subfolders, format_func=lambda p: p.name
)

# Directory that contains runs for the chosen type
tracking_root = selected_folder

# Find every produced tracking video
video_paths = sorted(tracking_root.rglob("processed_video.mp4"))

if not video_paths:
    st.error("No tracking videos found under data/HF_dataset/processed_videos/tracking.")
    st.stop()

# Let the user pick which video to inspect
def _label(p: Path) -> str:
    """Beautify the path for display in the selector."""
    try:
        # Show the two final path parts (segment folder / run folder)
        return "/".join(p.relative_to(tracking_root).parts[-2:])
    except ValueError:
        return str(p)

selected_video: Path = st.selectbox("Select tracking video", video_paths, format_func=_label)

# Display the selected video
st.video(selected_video)

# Associated timestamps CSV lives alongside the video
csv_path = selected_video.with_name("tracking_timestamps.csv")

segments: list[tuple[int, float, float]] = []

if csv_path.exists():
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["id"])
            t0 = float(row["start_ts"])
            t1 = float(row["end_ts"])
            segments.append((cid, t0, t1))
else:
    st.warning(f"No timestamp file found for {selected_video.name}.")

# Build options list for dropdown – label with cow ID
options_html = "\n".join(
    f'<option value="{t0}">Cow {cid}: {t0:.2f}–{t1:.2f}s</option>'
    for cid, t0, t1 in segments
)

# Inline HTML+JS for the segment selector that controls the first <video> element
html = f"""
<select
  id=\"segment_selector\"
  onchange=\"
    const vid = window.parent.document.querySelector('video');
    if (vid && this.value !== '') {{
      vid.currentTime = parseFloat(this.value);
      vid.play();
    }}\"
  style=\"font-size:14px; padding:4px;\"
>
  <option value=\"\">▶ Select a segment…</option>
  {options_html}
</select>
"""

# Render it just below the video
components.html(html, height=50)


