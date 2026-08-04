#!/usr/bin/env bash
cd /home/karimelrafi/repromo
exec .venv/bin/python scripts/webui/server.py --host 0.0.0.0 --port 8778 --gpu 1
