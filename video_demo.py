import csv
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.title("Cattle‐Visibility Segments")

# Directory that contains all tracking runs
tracking_root = Path(__file__).parent / "data" / "tracking_videos"

# Find every produced tracking video
video_paths = sorted(tracking_root.rglob("tracking_video.mp4"))

if not video_paths:
    st.error("No tracking videos found under data/tracking_videos.")
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


