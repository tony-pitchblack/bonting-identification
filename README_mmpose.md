# Infer 1 file

python mmpose_demo.py \
  data/3d_cattle_demo/renders/cow_boxed/sliding_clip_120s_v1/case=default/render=image/front_upper/2025-07-26_022630/000060.jpg \
  --pose2d animal \
  --vis-out-dir \
    data/3d_cattle_demo/mmpose_results/cow_boxed/sliding_clip_120s_v1/case=default/render=image/front_upper/2025-07-26_022630/000060.jpg

python mmpose_demo.py \
  "data/bonting-identification/source_videos/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear clipped.mp4" \
  --pose2d animal \
  --vis-out-dir \
    "data/bonting-identification/mmpose_results/youtube_segments/Milking R Dairy - Nedap CowControl with the SmartTag Ear clipped.mp4"

# Infer all files in folder

python mmpose_demo.py \
  data/3d_cattle_demo/renders/cow_boxed/sliding_clip_120s_v1/case=default/render=image/front_upper/2025-07-26_022630/ \
  --pose2d animal \
  --vis-out-dir \
    data/3d_cattle_demo/mmpose_results/cow_boxed/sliding_clip_120s_v1/case=default/render=image/front_upper/2025-07-26_022630/