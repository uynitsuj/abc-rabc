#!/usr/bin/env python3
"""Run in yam_sim/.venv. Rewrite a delivered scene_assembled.xml's asset paths
to this checkout via yam_sim's own normalizer, so it loads standalone in any venv.
"""
import argparse
from pathlib import Path
from yam_sim.rendering.replay.episode import _normalize_recorded_scene_xml

ap = argparse.ArgumentParser()
ap.add_argument("--in-xml", required=True)
ap.add_argument("--out-xml", required=True)
a = ap.parse_args()
xml = Path(a.in_xml).read_text()
Path(a.out_xml).write_text(_normalize_recorded_scene_xml(xml))
print(f"normalized -> {a.out_xml}")
