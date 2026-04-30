#!/usr/bin/env python3
"""
upload_bitstream.py — Upload and load a Cormorant bitstream on KV260.

Converts the Vivado .bit file to a flat .bin, uploads it to /lib/firmware/
on the board, loads it via the FPGA manager sysfs interface, configures the
Zynq MPSoC PS AXI fabric registers from the .hwh metadata, then applies
the device tree overlay (.dtbo) so that UIO devices appear in /dev/.

See src/bitstream/ for the implementation and src/bitstream/platforms/kv260.py
for KV260-specific register tables and xclbin metadata.

Usage:
  python upload_bitstream.py --config remote_config.json \\
      --bit ../hw/cormorant_hw_128/cormorant_hw_128.runs/impl_1/design_cormorant_wrapper.bit \\
      --hwh ../hw/cormorant_hw_128/.gen/sources_1/bd/design_cormorant/hw_handoff/design_cormorant.hwh \\
      --dtbo ../dts/kv260/pl.dtbo

  # --hwh may be omitted when <bit_stem>.hwh exists alongside the .bit file.
  # xclbinutil must be on PATH (e.g. source Vitis settings64.sh).

  # Check board readiness only (no upload)
  python upload_bitstream.py --config remote_config.json --check-only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bitstream import check_board, upload_bitstream  # noqa: E402
from src.bitstream.board import fpga_state, list_uio_devices  # noqa: E402
from src.remote import (  # noqa: E402
    _green, _red, _dim,
    load_config,
    RemoteSession,
)

_REPO_ROOT = Path(__file__).parent.parent

_DEFAULT_BIT = (
    _REPO_ROOT / "hw" / "cormorant_hw_128"
    / "cormorant_hw_128.runs" / "impl_1"
    / "design_cormorant_wrapper.bit"
)


def _resolve_hwh(explicit: str | None, bit_path: Path) -> Path:
    """
    Resolve the HWH path.
    1. Explicit --hwh argument
    2. <bit_stem>.hwh alongside the .bit file
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = bit_path.with_suffix(".hwh")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"HWH file not found alongside {bit_path.name} "
        f"(looked for {candidate}).  Pass --hwh explicitly."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upload a Cormorant bitstream to a KV260 board.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", required=True, metavar="FILE",
        help="JSON config file (same format as remote_config.json).",
    )
    p.add_argument(
        "--bit", metavar="FILE", default=None,
        help=f"Vivado .bit file.  Default: {_DEFAULT_BIT.relative_to(_REPO_ROOT)}",
    )
    p.add_argument(
        "--hwh", metavar="FILE", default=None,
        help="Hardware-handoff .hwh file.  Default: <bit_stem>.hwh alongside the .bit.",
    )
    p.add_argument(
        "--dtbo", metavar="FILE", required=True,
        help="Device tree overlay .dtbo (required).",
    )
    p.add_argument(
        "--overlay-name", metavar="NAME", default=None,
        help=(
            "Name for the configfs overlay directory and /lib/firmware/<name>.bin.  "
            "Defaults to the .dtbo stem."
        ),
    )
    p.add_argument(
        "--xclbinutil", metavar="CMD", default="xclbinutil",
        help="xclbinutil command or path (default: xclbinutil from PATH).",
    )
    p.add_argument(
        "--check-only", action="store_true",
        help="Verify board readiness only; do not upload or load anything.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    bit_path   = Path(args.bit).expanduser().resolve() if args.bit else _DEFAULT_BIT
    dtbo_path  = Path(args.dtbo).expanduser().resolve()
    xclbinutil = args.xclbinutil

    try:
        hwh_path = _resolve_hwh(args.hwh, bit_path)
    except FileNotFoundError as exc:
        print(_red(f"error: {exc}"), file=sys.stderr)
        return 1

    overlay_name = args.overlay_name or dtbo_path.stem

    if not args.check_only:
        for label, path in [(".bit", bit_path), (".dtbo", dtbo_path), (".hwh", hwh_path)]:
            if not path.exists():
                print(_red(f"error: {label} file not found: {path}"), file=sys.stderr)
                if label == ".bit":
                    print(
                        "       Build the bitstream first:\n"
                        "         cd <repo>/build && cmake .. -DAXI_BUS_WIDTH=128\n"
                        "         make build_hw_kv260",
                        file=sys.stderr,
                    )
                return 1

    cfg     = load_config(args.config, {})
    ssh_cfg = cfg["ssh"]

    print(f"Connecting to {ssh_cfg['user']}@{ssh_cfg['host']}:{ssh_cfg['port']} …")
    session = RemoteSession(ssh_cfg)
    session.connect()
    print(f"  {_green('Connected')}")

    try:
        if args.check_only:
            print(f"\nBoard readiness check")
            ok    = check_board(session)
            state = fpga_state(session)
            print(f"\n  FPGA state:  {_dim(state)}")
            uios = list_uio_devices(session)
            if uios:
                print(f"  UIO devices:")
                for name, dev in uios:
                    print(f"    {dev}  {name}")
            else:
                print(f"  UIO devices: none")
            return 0 if ok else 1

        print(f"\n  .bit        {bit_path}")
        print(f"  .hwh        {hwh_path}")
        print(f"  .dtbo       {dtbo_path}")
        print(f"  name        {overlay_name}")
        print(f"  xclbinutil  {xclbinutil}")

        upload_bitstream(session, bit_path, hwh_path, dtbo_path, overlay_name, xclbinutil)

    except (RuntimeError, TimeoutError) as exc:
        print(f"\n{_red('Error:')} {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
