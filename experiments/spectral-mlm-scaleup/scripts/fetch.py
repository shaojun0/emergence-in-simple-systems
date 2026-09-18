#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable, logged downloader (Windows-friendly; no shell quirks).

  python fetch.py URL OUT [PROXY] [MAX_SECONDS]
"""
import os
import socket
import sys
import time
import urllib.request

CHUNK = 1 << 20
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def human(n):
    return "%.1fMB" % (n / 1e6)


def main():
    url, out = sys.argv[1], sys.argv[2]
    proxy = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("-", "") else None
    max_s = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    t0 = time.time()
    for attempt in range(200):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            r = opener.open(req, timeout=60)
        except Exception as e:                                    # noqa: BLE001
            print("open failed (%s), retry in 5s" % e, flush=True)
            time.sleep(5)
            continue
        total = r.headers.get("Content-Length")
        total = have + int(total) if total else None
        print("start at %s%s" % (human(have), " / %s" % human(total) if total else ""),
              flush=True)
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        last = time.time()
        try:
            with open(out, mode) as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 15:
                        last = time.time()
                        speed = have / max(1e-9, time.time() - t0)
                        print("  %s%s  %.0f KB/s" % (human(have),
                                                     " / %s" % human(total) if total else "",
                                                     speed / 1024), flush=True)
                        if max_s and time.time() - t0 > max_s:
                            print("time budget reached; stopping (resumable)", flush=True)
                            return 0
        except Exception as e:                                    # noqa: BLE001
            print("  interrupted (%s); resuming" % e, flush=True)
            time.sleep(3)
            continue
        sz = os.path.getsize(out)
        if total is None or sz >= total:
            print("DONE %s %s in %.0fs" % (out, human(sz), time.time() - t0), flush=True)
            return 0
        print("  short read (%s / %s); resuming" % (human(sz), human(total)), flush=True)
        time.sleep(2)
    print("GAVE UP", flush=True)
    return 1


if __name__ == "__main__":
    socket.setdefaulttimeout(60)
    raise SystemExit(main())
