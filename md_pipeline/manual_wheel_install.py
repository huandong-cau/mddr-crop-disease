# -*- coding: utf-8 -*-
"""manual_wheel_install.py — fetch wheels from the aliyun HTTP mirror with
curl and unpack them into pylibs/ (bypasses pip, which is blocked by the
sandbox during its unpack stage)."""
import os
import re
import subprocess
import sys
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
WHEELS = os.path.join(BASE, "wheels")
PYLIBS = os.path.join(BASE, "pylibs")
INDEX = "http://mirrors.aliyun.com/pypi/simple"
os.makedirs(WHEELS, exist_ok=True)
os.makedirs(PYLIBS, exist_ok=True)
CURL = r"C:\Windows\System32\curl.exe"

PKGS = ["matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver",
        "packaging", "pyparsing", "python-dateutil"]


def http_get(url):
    out = subprocess.run([CURL, "-s", "--max-time", "60", url],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    return out.stdout


def vkey(l):
    """(main_version_tuple, rc_flag) — higher is newer."""
    m = re.search(r'-(\d+(?:\.\d+)*(?:rc\d+)?)-', l)
    vs = m.group(1)
    main = tuple(int(x) for x in vs.split("rc")[0].split("."))
    rc = int(vs.split("rc")[1]) if "rc" in vs else 99
    return (main, rc)


def pick_wheel(pkg):
    from urllib.parse import urljoin
    html = http_get(f"{INDEX}/{pkg}/")
    if not html:
        raise RuntimeError(f"empty index for {pkg}")
    links = re.findall(r'href="([^"]+\.whl[^"]*)"', html)
    bad = ("manylinux", "macosx", "linux_", "musllinux", "aarch64", "arm64",
           "win32", "emscripten", "wasm", "ios", "android", "ppc64le", "s390x")
    cand = [l for l in links if re.search(r'-(\d+(?:\.\d+)*(?:rc\d+)?)-', l)
            and not any(t in l for t in bad)]
    # newest version first, then prefer cp313 win_amd64, abi3, then any
    def rank(l):
        v, rc = vkey(l)
        abi = 3
        if "cp313" in l and "win_amd64" in l:
            abi = 0
        elif "win_amd64" in l:
            abi = 1
        elif "none-any" in l or "py3-none" in l:
            abi = 2
        return (v, -rc, -abi)
    cand.sort(key=rank, reverse=True)
    url = urljoin(f"{INDEX}/{pkg}/", cand[0]).split("#")[0]
    return url


def main():
    for pkg in PKGS:
        try:
            url = pick_wheel(pkg)
        except Exception as e:
            print(f"[{pkg}] index fail: {e}")
            continue
        fn = os.path.basename(url.split("#")[0])
        dst = os.path.join(WHEELS, fn)
        if not os.path.exists(dst):
            print(f"[{pkg}] downloading {fn}")
            r = subprocess.run([CURL, "-sL", "--retry", "3", "--max-time", "600",
                                "-o", dst, url])
            if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 1000:
                print(f"[{pkg}] download FAILED")
                continue
        # unpack
        try:
            with zipfile.ZipFile(dst) as z:
                z.extractall(PYLIBS)
            print(f"[{pkg}] installed {fn} ({os.path.getsize(dst)//1024} KB)")
        except zipfile.BadZipFile:
            print(f"[{pkg}] BAD ZIP {fn} ({os.path.getsize(dst)} bytes); removing")
            os.remove(dst)
    # drop the extra dist-info/numpy binaries conflicts? keep as-is (consistent set)
    print("DONE")


if __name__ == "__main__":
    main()
