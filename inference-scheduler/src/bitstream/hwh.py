"""Vivado hardware-handoff (.hwh) parsing."""

import xml.etree.ElementTree as ET
from pathlib import Path

_PS_MODTYPES = {"zynq_ultra_ps_e", "processing_system7"}


def parse_hwh_ps_params(hwh_path: Path) -> tuple[str, dict]:
    """
    Return (family, params) from the PS IP module in the HWH.

    family — "zynq_ultra_ps_e" or "processing_system7"
    params — {name: value} for all C_SAXIGP*/C_MAXIGP*_DATA_WIDTH parameters
    """
    root = ET.parse(hwh_path).getroot()
    for mod in root.iter("MODULE"):
        family = mod.get("MODTYPE", "")
        if family not in _PS_MODTYPES:
            continue
        params = {
            p.get("NAME"): p.get("VALUE")
            for p in mod.findall("./PARAMETERS/PARAMETER")
            if p.get("NAME", "").startswith(("C_SAXIGP", "C_MAXIGP"))
            and p.get("NAME", "").endswith("_DATA_WIDTH")
        }
        return family, params
    raise ValueError(
        f"No PS IP module (zynq_ultra_ps_e / processing_system7) found in {hwh_path}"
    )


def parse_hwh_mem_topology(hwh_path: Path) -> dict:
    """
    Build the MEM_TOPOLOGY dict for xclbinutil from the HWH.

    Always seeds with a PSDDR bank at index 0 (address 0, 256 MiB) so that
    xclAllocBO(..., flags=0) in inference_buf.c resolves to a named bank.
    Additional MEMTYPE=MEMORY ranges from the HWH are appended.
    """
    root = ET.parse(hwh_path).getroot()

    mem_data: list[dict] = [{
        "m_type":         "MEM_DDR4",
        "m_used":         1,
        "m_sizeKB":       256 * 1024,
        "m_tag":          "PSDDR",
        "m_base_address": 0,
    }]

    seen: set[int] = {0}
    for mr in root.iter("MEMRANGE"):
        if mr.get("MEMTYPE") != "MEMORY":
            continue
        try:
            base = int(mr.get("BASEVALUE", "0"), 16)
            high = int(mr.get("HIGHVALUE", "0"), 16)
        except ValueError:
            continue
        if base in seen:
            continue
        seen.add(base)
        mem_data.append({
            "m_type":         "MEM_DDR4",
            "m_used":         1,
            "m_sizeKB":       max((high - base + 1) // 1024, 1),
            "m_tag":          f"MIG{len(mem_data)}",
            "m_base_address": base,
        })

    return {"m_count": len(mem_data), "m_mem_data": mem_data}
