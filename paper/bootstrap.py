#!/usr/bin/env python3
"""Fetch the pinned portable compiler locally, with archive SHA-256 verification."""
from pathlib import Path
import hashlib
import io
import os
import platform
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
URL = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
EXPECTED = '8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'

def main():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise SystemExit('Use a platform-appropriate Tectonic installation; this bootstrap is Linux x86_64 only.')
    target = HERE/'tools/tectonic'
    if target.exists():
        print('Local compiler already exists:',target)
        return
    data = urllib.request.urlopen(URL,timeout=60).read()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise SystemExit('Compiler archive hash mismatch; nothing installed.')
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        member=archive.getmember('tectonic')
        if not member.isfile(): raise SystemExit('Unexpected archive member')
        binary=archive.extractfile(member).read()
    target.parent.mkdir(exist_ok=True)
    temporary=target.with_suffix('.tmp')
    temporary.write_bytes(binary)
    temporary.chmod(0o755)
    os.replace(temporary,target)
    print('Installed pinned portable Tectonic inside paper/tools only.')

if __name__=='__main__': main()
