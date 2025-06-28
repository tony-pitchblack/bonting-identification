import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.title("Cattle‐Visibility Segments")

# 1) Point to your video file. If it's local, convert to a file:// URL
relative_file_path = "videos/source/Professional Cattle Handling [9sWtw_EtHKI].webm"
file_path = (
    Path(__file__).parent
    / relative_file_path
)
video_url = file_path.resolve().as_uri()

# Use Streamlit's built‐in video player so the file is served correctly over HTTP
st.video(file_path)

# 2) Your precomputed segments: (cow_id, t_start, t_end)
segments = [
    (0, 0, 10),
    (1, 10, 20),
    (2, 20, 30),
]

# 3) Build the <option> list for the dropdown
options_html = "\n".join(
    f'<option value="{t0}">#{cid}: {t0:.1f}–{t1:.1f}s</option>'
    for cid, t0, t1 in segments
)

# 4) Inline HTML+JS for the segment selector that controls the first <video> element
html = f"""
<select
  id="segment_selector"
  onchange="
    const vid = window.parent.document.querySelector('video');
    if (vid && this.value !== '') {{
      vid.currentTime = parseFloat(this.value);
      vid.play();
    }}
  "
  style="font-size:14px; padding:4px;"
>
  <option value="">▶ Select a segment…</option>
  {options_html}
</select>
"""

# 5) Render it just below the video
components.html(html, height=50)


