#!/usr/bin/env python3
"""Extract per-camera mp4s from a re-rendered output.mcap (foxglove.CompressedVideo)."""
import argparse, subprocess, os
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as PbDec
CAMS={"/top-camera/image-raw":"top","/left-wrist-camera/image-raw":"left","/right-wrist-camera/image-raw":"right"}
ap=argparse.ArgumentParser()
ap.add_argument("--mcap",required=True); ap.add_argument("--out-dir",required=True); ap.add_argument("--fps",default="30")
a=ap.parse_args()
os.makedirs(a.out_dir,exist_ok=True)
buf={v:[] for v in CAMS.values()}
with open(a.mcap,"rb") as f:
    for _s,ch,m,dec in make_reader(f,decoder_factories=[PbDec()]).iter_decoded_messages(topics=list(CAMS)):
        buf[CAMS[ch.topic]].append((m.log_time,bytes(dec.data)))
for name,msgs in buf.items():
    msgs.sort()
    h264=b"".join(d for _,d in msgs)
    tmp=os.path.join(a.out_dir,f"{name}.h264"); open(tmp,"wb").write(h264)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-f","h264","-r",a.fps,"-i",tmp,
                    "-c:v","libx264","-pix_fmt","yuv420p",os.path.join(a.out_dir,f"{name}_camera.mp4")],check=True)
    os.remove(tmp)
    print(f"  {name}: {len(msgs)} frames")
