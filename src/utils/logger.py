import json
import random
import os
from datetime import datetime

def log_extraction_report(filename, text, chunks, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "file": filename,
        "timestamp": datetime.utcnow().isoformat(),
        "characters_total": len(text),
        "chunks_total": len(chunks),
        "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks),
        "sample_chunks": random.sample(chunks, min(3, len(chunks)))
    }

    report_path = os.path.join(output_dir, f"{filename}.report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
