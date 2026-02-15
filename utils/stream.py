#!/usr/bin/env python3
"""
Simple RTSP server that streams a webcam device.

Usage:
    python stream.py /dev/video0
    python stream.py 0

Stream URL: rtsp://localhost:8554/stream

Dependencies (macOS):
    brew install gstreamer gst-plugins-base gst-plugins-good \
        gst-plugins-ugly gst-rtsp-server pygobject3

Dependencies (Linux):
    apt install gstreamer1.0-tools gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly \
        gir1.2-gst-rtsp-server-1.0 python3-gi
"""

import argparse
import platform
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


def build_launch_pipeline(device: str) -> str:
    """
    Build a GStreamer launch pipeline string for the given webcam device.

    Selects the appropriate video source element based on the device path
    and platform (v4l2src on Linux, avfvideosrc on macOS).

    Args:
        device: Device path (e.g. "/dev/video0") or integer index (e.g. "0")

    Returns:
        GStreamer launch pipeline string suitable for RTSPMediaFactory.
    """
    if device.startswith("/dev/video"):
        # Explicit Linux device path
        src = f"v4l2src device={device}"
    else:
        try:
            idx = int(device)
        except ValueError:
            print(
                f"Error: '{device}' is not a valid device path or index",
                file=sys.stderr,
            )
            sys.exit(1)

        if platform.system() == "Darwin":
            src = f"avfvideosrc device-index={idx}"
        else:
            src = f"v4l2src device=/dev/video{idx}"

    return (
        f"( {src} ! videoconvert ! video/x-raw,format=I420"
        f" ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast"
        f" ! rtph264pay name=pay0 pt=96 )"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stream a webcam device as an RTSP server"
    )
    parser.add_argument(
        "device",
        help="Webcam device path (e.g. /dev/video0) or index (e.g. 0)",
    )
    parser.add_argument(
        "--port", type=int, default=8554, help="RTSP server port (default: 8554)"
    )
    parser.add_argument(
        "--mount", default="/stream", help="RTSP mount point (default: /stream)"
    )
    args = parser.parse_args()

    Gst.init(None)

    server = GstRtspServer.RTSPServer()
    server.set_service(str(args.port))

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(build_launch_pipeline(args.device))
    factory.set_shared(True)

    mounts = server.get_mount_points()
    mounts.add_factory(args.mount, factory)

    server.attach(None)

    url = f"rtsp://192.168.1.192:{args.port}{args.mount}"
    print(f"RTSP server running at {url}")
    print(f"Device: {args.device}")
    print("Press Ctrl+C to stop")

    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
