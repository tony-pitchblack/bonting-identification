import csv
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit_tree_select import tree_select


st.title("Cattle‐Visibility Segments")

# Base directory that contains all processed videos (tracking, identification, ...)
processed_root = Path(__file__).parent / "data" / "HF_dataset" / "processed_videos"

# Let the user choose which processed subfolder to browse (e.g. "tracking", "identification")
subfolders = [d for d in processed_root.iterdir() if d.is_dir()]
if not subfolders:
    st.error("No subfolders found under data/HF_dataset/processed_videos.")
    st.stop()

# Sort subfolders by name for consistent display
subfolders = sorted(subfolders, key=lambda p: p.name)

# Build tree structure for file selection
def build_tree_structure():
    tree_nodes = []
    
    for subfolder in subfolders:
        # Find all video paths in this subfolder
        video_paths = list(subfolder.rglob("processed_video.mp4"))
        
        if not video_paths:
            continue
            
        # Group by source name
        source_groups = {}
        for video_path in video_paths:
            source_name = extract_source_name_from_path(video_path, subfolder)
            if source_name not in source_groups:
                source_groups[source_name] = []
            source_groups[source_name].append(video_path)
        
        # Build children for this subfolder
        subfolder_children = []
        for source_name in sorted(source_groups.keys()):
            source_videos = source_groups[source_name]
            
            # Build children for this source
            video_children = []
            for video_path in sorted(source_videos, key=lambda p: p.parent.name):
                video_children.append({
                    "label": video_path.parent.name,
                    "value": str(video_path)
                })
            
            subfolder_children.append({
                "label": source_name,
                "value": f"{subfolder.name}_{source_name}",
                "children": video_children
            })
        
        # Add subfolder to tree
        tree_nodes.append({
            "label": subfolder.name,
            "value": subfolder.name,
            "children": subfolder_children
        })
    
    return tree_nodes

def extract_source_name_from_path(path: Path, tracking_root: Path) -> str:
    """Extract source video name from path structure."""
    try:
        # Get relative path from tracking_root
        rel_path = path.relative_to(tracking_root)
        # The source name should be the first directory in the relative path
        if len(rel_path.parts) >= 2:
            return rel_path.parts[0]
        else:
            return rel_path.parts[0] if rel_path.parts else "Unknown"
    except ValueError:
        return "Unknown"

# Build and display tree selection
tree_nodes = build_tree_structure()

if not tree_nodes:
    st.error("No processed videos found.")
    st.stop()

st.subheader("Select Video")
selected = tree_select(
    tree_nodes,
    check_model="leaf",
    only_leaf_checkboxes=True,
    show_expand_all=True
)

# Get selected video path
if not selected['checked']:
    st.info("Please select a video from the tree above.")
    st.stop()

# Get the first selected video (since we're using leaf selection)
selected_video_str = selected['checked'][0]
selected_video_path = Path(selected_video_str)

if not selected_video_path.exists():
    st.error(f"Selected video file does not exist: {selected_video_path}")
    st.stop()

# Display the selected video
st.video(selected_video_path)

# Associated timestamps CSV lives alongside the video
csv_path = selected_video_path.with_name("tracking_timestamps.csv")

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
    st.warning(f"No timestamp file found for {selected_video_path.name}.")

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


