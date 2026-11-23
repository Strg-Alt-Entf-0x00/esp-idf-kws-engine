"""ESPM v3 Exporter for TC-ResNet KWS and Command Models."""

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from model import TCResNet, extract_folded_weights

ESPM_MAGIC_LE = 0x4D505345
ESPM_VERSION = 3
MODEL_TYPE_KWS = 2
MODEL_QUANT_FP32 = 4

SECTION_TYPE_CONFIG_JSON = 5
SECTION_TYPE_TENSOR_INDEX = 6
SECTION_TYPE_TENSOR_DATA = 7

ALIGNMENT = 64
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 40
V3_INDEX_HEADER_SIZE = 36
V3_TENSOR_RECORD_SIZE = 160

HEADER_STRUCT = struct.Struct("<IHHHH32s16sIIIQIHHIII28s")
SECTION_STRUCT = struct.Struct("<HHQQI16s")
INDEX_HEADER_STRUCT = struct.Struct("<4sHHHHIIQQ")
TENSOR_RECORD_STRUCT = struct.Struct("<II4BI8IIB3sQIQ Q64s12s")


def align_to(val: int, alignment: int = ALIGNMENT) -> int:
    return (val + alignment - 1) & ~(alignment - 1)


def fixed_string(val: str, max_len: int) -> bytes:
    encoded = val.encode("utf-8")[:max_len - 1]
    return encoded.ljust(max_len, b"\x00")


def export_kws_to_espm(
    model: TCResNet,
    labels: List[str],
    output_path: Path,
    model_name: str = "kws_tcresnet"
) -> Path:
    """Exports trained PyTorch TCResNet model into a 100% canonical ESPM v3 container."""
    folded = extract_folded_weights(model)

    tensor_data = bytearray()
    tensor_records = []
    total_elements = 0
    total_data_bytes = 0

    for name, tensor in folded.items():
        arr = tensor.cpu().numpy().astype(np.float32)
        shape = tuple(arr.shape)
        elements = math.prod(shape)
        payload = arr.tobytes()

        # 16-byte alignment inside tensor data section
        pad_offset = (-len(tensor_data)) % 16
        tensor_data.extend(b"\x00" * pad_offset)
        data_offset = len(tensor_data)
        tensor_data.extend(payload)

        dimensions = shape + (0,) * (8 - len(shape))
        name_bytes = fixed_string(name, 64)
        name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF

        rec_bin = TENSOR_RECORD_STRUCT.pack(
            name_hash, 0, 2, len(shape), 0, 0, elements, *dimensions,
            0, 0xFF, b"\x00\x00\x00", 0, 0, data_offset, len(payload),
            name_bytes, b"\x00" * 12
        )
        tensor_records.append(rec_bin)
        total_elements += elements
        total_data_bytes += len(payload)

    # 1. Config JSON payload
    config_dict = {
        "model_type": "KWS_TCRESNET",
        "sample_rate_hz": 16000,
        "num_mels": 40,
        "time_frames": 64,
        "num_classes": len(labels),
        "labels": labels,
        "base_channels": 16
    }
    config_json_bytes = json.dumps(config_dict, indent=2).encode("utf-8")

    # 2. Tensor Index payload
    idx_hdr_bytes = INDEX_HEADER_STRUCT.pack(
        b"ETIX", 2, V3_INDEX_HEADER_SIZE, V3_TENSOR_RECORD_SIZE, 0,
        len(tensor_records), 1, total_elements, total_data_bytes
    )
    index_payload = idx_hdr_bytes + b"".join(tensor_records)

    # 3. Build Sections
    sections = [
        (SECTION_TYPE_CONFIG_JSON, 0, "config", config_json_bytes),
        (SECTION_TYPE_TENSOR_INDEX, 0, "tensors", bytes(index_payload)),
        (SECTION_TYPE_TENSOR_DATA, 0, "weights", bytes(tensor_data)),
    ]

    table_end = HEADER_SIZE + len(sections) * SECTION_ENTRY_SIZE
    offset = align_to(table_end)

    rows = []
    for kind, flags, sname, payload in sections:
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        rows.append((kind, flags, sname, offset, payload, crc))
        offset = align_to(offset + len(payload))

    file_size = rows[-1][3] + len(rows[-1][4])

    section_table = b"".join(
        SECTION_STRUCT.pack(kind, flags, s_offset, len(payload), crc, fixed_string(sname, 16))
        for kind, flags, sname, s_offset, payload, crc in rows
    )
    table_crc = zlib.crc32(section_table) & 0xFFFFFFFF
    psram_kb = (len(tensor_data) + 1023) // 1024

    def pack_hdr(header_crc: int) -> bytes:
        return HEADER_STRUCT.pack(
            ESPM_MAGIC_LE, ESPM_VERSION, MODEL_TYPE_KWS, MODEL_QUANT_FP32,
            len(rows), fixed_string(model_name, 32), fixed_string("1.0.0", 16),
            16000, psram_kb, header_crc,
            file_size, HEADER_SIZE, HEADER_SIZE, SECTION_ENTRY_SIZE, ALIGNMENT,
            0, table_crc, b"\x00" * 28
        )

    hdr = pack_hdr(0)
    hdr = pack_hdr(zlib.crc32(hdr) & 0xFFFFFFFF)

    output = bytearray(file_size)
    output[:HEADER_SIZE] = hdr
    output[HEADER_SIZE:HEADER_SIZE + len(section_table)] = section_table
    for _, _, _, s_offset, payload, _ in rows:
        output[s_offset:s_offset + len(payload)] = payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)

    print(f"[ESPM Exporter] Successfully generated '{output_path}' ({file_size} bytes, {total_data_bytes/1024:.2f} KB weights)")
    return output_path
