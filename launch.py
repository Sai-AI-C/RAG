"""
OmniDoc AI -- Mobile & Cloudflare Tunnel Launcher
==================================================
Starts OmniDoc AI with a free public HTTPS URL via Cloudflare Tunnel.
Enables instant access from any mobile phone, tablet, or PC worldwide.

Usage:
    python launch.py
"""

import os
import re
import sys
import time
import socket
import shutil
import threading
import subprocess

PORT = 8501
CLOUDFLARED_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared-windows-amd64.exe"),
]


def find_cloudflared():
    """Find cloudflared -- in project folder (any name) or system PATH."""
    for candidate in CLOUDFLARED_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("cloudflared")
    if found:
        return found
    return None


def is_port_in_use(port):
    """Check if server port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def get_local_ip():
    """Detect local LAN IP address for Wi-Fi / local mobile access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_streamlit():
    """Launch Streamlit with mobile-optimized server settings if not already running."""
    if is_port_in_use(PORT):
        print(f"[OK] Streamlit server is already running on port {PORT}.", flush=True)
        return None

    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.gatherUsageStats", "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )


def wait_for_streamlit(port: int, timeout: int = 30) -> bool:
    """Wait for Streamlit server to bind and accept HTTP connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def start_cloudflare_tunnel(cloudflared_path):
    """Start cloudflared quick tunnel pointing strictly to 127.0.0.1."""
    proc = subprocess.Popen(
        [cloudflared_path, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    public_url = None
    url_found = threading.Event()

    def read_output(stream):
        nonlocal public_url
        for line in iter(stream.readline, b""):
            decoded = line.decode("utf-8", errors="ignore").strip()
            match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", decoded)
            if match and not url_found.is_set():
                public_url = match.group(0)
                url_found.set()

    t1 = threading.Thread(target=read_output, args=(proc.stdout,), daemon=True)
    t2 = threading.Thread(target=read_output, args=(proc.stderr,), daemon=True)
    t1.start()
    t2.start()

    url_found.wait(timeout=30)
    return proc, public_url


def print_banner(public_url, local_ip):
    sep = "=" * 66
    print("\n" + sep, flush=True)
    print("   OmniDoc AI is LIVE & Accessible Worldwide on Mobile & Desktop!", flush=True)
    print(sep, flush=True)
    if public_url:
        print(f"\n   PUBLIC MOBILE URL  -->  {public_url}", flush=True)
        print("   (Works from ANY mobile phone, tablet, or PC anywhere!)", flush=True)
    else:
        print("\n   [WARNING] Could not obtain Cloudflare Public URL automatically.", flush=True)
        print("   Tunnel may still be starting or check Cloudflare log output.", flush=True)
    
    print(f"\n   SAME WI-FI / LAN   -->  http://{local_ip}:{PORT}", flush=True)
    print(f"   LOCAL PC           -->  http://localhost:{PORT}", flush=True)
    print("\n" + sep, flush=True)
    print("   MOBILE CONNECTION INSTRUCTIONS:", flush=True)
    print("   1. Anywhere on Mobile Data / 4G / 5G / Remote Wi-Fi:", flush=True)
    print(f"      Open on Mobile Phone Browser: {public_url or 'https://your-tunnel.trycloudflare.com'}", flush=True)
    print(f"   2. On Same Home / College Wi-Fi Network:", flush=True)
    print(f"      Open on Mobile Phone Browser: http://{local_ip}:{PORT}", flush=True)
    print("\n   [NOTE] Server is RUNNING. Keep this terminal window open!", flush=True)
    print("   Press Ctrl+C anytime to stop the server.", flush=True)
    print(sep + "\n", flush=True)

    # Save active links to MOBILE_LINK.txt for background running
    try:
        link_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MOBILE_LINK.txt")
        with open(link_path, "w", encoding="utf-8") as f:
            f.write("=====================================================\n")
            f.write("        OMNIDOC AI — ACTIVE MOBILE LINKS\n")
            f.write("=====================================================\n\n")
            if public_url:
                f.write(f"PUBLIC MOBILE URL (4G/5G/Anywhere):\n{public_url}\n\n")
            f.write(f"SAME WI-FI / LOCAL NETWORK LINK:\nhttp://{local_ip}:{PORT}\n\n")
            f.write(f"LOCAL PC LINK:\nhttp://localhost:{PORT}\n\n")
            f.write("=====================================================\n")
    except Exception:
        pass


def main():
    sep = "=" * 66
    print("\n" + sep, flush=True)
    print("   OmniDoc AI -- Mobile & Cloudflare Tunnel Launcher", flush=True)
    print(sep, flush=True)

    cloudflared = find_cloudflared()
    if not cloudflared:
        print("\n[ERROR] cloudflared.exe not found!", flush=True)
        print("\n   Download it here (one file, no install needed):", flush=True)
        print("   https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe", flush=True)
        print("\n   Save it in this folder:", flush=True)
        print(f"   {os.path.dirname(os.path.abspath(__file__))}", flush=True)
        print("\n   Then run:  python launch.py  again.\n", flush=True)
        sys.exit(1)

    print(f"\n[OK] cloudflared found: {os.path.basename(cloudflared)}", flush=True)
    local_ip = get_local_ip()

    print(f"[>>] Checking OmniDoc AI server on port {PORT}...", flush=True)
    streamlit_proc = start_streamlit()
    
    # Ensure Streamlit is fully bound to 127.0.0.1:8501 before launching tunnel
    if not wait_for_streamlit(PORT, timeout=30):
        print(f"[ERROR] Streamlit server failed to start on port {PORT} within 30 seconds.", flush=True)
        sys.exit(1)

    print("[>>] Creating Cloudflare public HTTPS tunnel for mobile access...", flush=True)
    cf_proc, public_url = start_cloudflare_tunnel(cloudflared)

    print_banner(public_url, local_ip)

    # Keep alive until Ctrl+C by waiting on the Cloudflare process
    try:
        cf_proc.wait()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down OmniDoc AI...", flush=True)
        cf_proc.terminate()
        if streamlit_proc:
            streamlit_proc.terminate()
        print("       Server stopped. Goodbye!\n", flush=True)


if __name__ == "__main__":
    main()
