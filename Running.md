PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1 \
uv run video_demo.py \
  --config-file /data/Niharika/folders/videomt/videomt/configs/VSPW/videomt/vit-large/videomt_online_ViTL.yaml \
  --input /data/Niharika/folders/videomt/videomt/input_frames_airplane/ \
  --output /data/Niharika/folders/videomt/videomt/output_vipseg_plane/ \
  --confidence_threshold 0.5 \
  --windows_size 11 \
  --opts MODEL.WEIGHTS /data/Niharika/folders/videomt/videomt/configs/VSPW/vspw_vit_large_95.0_64.9.pth \
  DATALOADER.NUM_WORKERS 0 \
  2>&1 | tee run_log.txt


