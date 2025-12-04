#!/usr/bin/env python3
"""Search for Whisper ONNX models on HuggingFace"""
import requests

print("=" * 70)
print("🔍 Searching for Whisper ONNX models on HuggingFace...")
print("=" * 70)

# Search for Whisper ONNX models
searches = [
    ("whisper onnx", "Whisper ONNX Models"),
    ("whisper-tiny onnx", "Tiny Models"),
    ("whisper-small onnx", "Small Models"),
    ("whisper-medium onnx", "Medium Models"),
    ("whisper-large onnx", "Large Models"),
]

all_models = {}

for query, label in searches:
    try:
        r = requests.get(
            'https://huggingface.co/api/models',
            params={'search': query, 'limit': 30},
            timeout=10
        )
        if r.ok:
            for m in r.json():
                model_id = m['id']
                if 'whisper' in model_id.lower():
                    all_models[model_id] = {
                        'downloads': m.get('downloads', 0),
                        'likes': m.get('likes', 0),
                    }
    except Exception as e:
        print(f"Error searching for {query}: {e}")

# Sort by downloads
sorted_models = sorted(all_models.items(), key=lambda x: x[1]['downloads'], reverse=True)

print(f"\n📦 Found {len(sorted_models)} Whisper-related ONNX models:\n")
print(f"{'Model ID':<55} {'Downloads':>12} {'Likes':>8}")
print("-" * 78)

for model_id, stats in sorted_models[:40]:
    downloads = stats['downloads']
    likes = stats['likes']
    # Format downloads
    if downloads >= 1_000_000:
        dl_str = f"{downloads/1_000_000:.1f}M"
    elif downloads >= 1_000:
        dl_str = f"{downloads/1_000:.1f}K"
    else:
        dl_str = str(downloads)
    print(f"{model_id:<55} {dl_str:>12} {likes:>8}")

# Also check Xenova models specifically (known ONNX provider)
print("\n" + "=" * 70)
print("🔍 Checking Xenova (known ONNX model provider)...")
print("=" * 70)

try:
    r = requests.get(
        'https://huggingface.co/api/models',
        params={'author': 'Xenova', 'limit': 100},
        timeout=10
    )
    if r.ok:
        whisper_models = [m for m in r.json() if 'whisper' in m['id'].lower()]
        print(f"\n📦 Found {len(whisper_models)} Whisper models from Xenova:\n")
        for m in sorted(whisper_models, key=lambda x: x.get('downloads', 0), reverse=True):
            downloads = m.get('downloads', 0)
            if downloads >= 1_000_000:
                dl_str = f"{downloads/1_000_000:.1f}M"
            elif downloads >= 1_000:
                dl_str = f"{downloads/1_000:.1f}K"
            else:
                dl_str = str(downloads)
            print(f"  {m['id']:<50} {dl_str:>10} downloads")
except Exception as e:
    print(f"Error: {e}")

# Check onnx-community
print("\n" + "=" * 70)
print("🔍 Checking onnx-community...")
print("=" * 70)

try:
    r = requests.get(
        'https://huggingface.co/api/models',
        params={'author': 'onnx-community', 'limit': 100},
        timeout=10
    )
    if r.ok:
        whisper_models = [m for m in r.json() if 'whisper' in m['id'].lower()]
        print(f"\n📦 Found {len(whisper_models)} Whisper models from onnx-community:\n")
        for m in sorted(whisper_models, key=lambda x: x.get('downloads', 0), reverse=True):
            downloads = m.get('downloads', 0)
            if downloads >= 1_000_000:
                dl_str = f"{downloads/1_000_000:.1f}M"
            elif downloads >= 1_000:
                dl_str = f"{downloads/1_000:.1f}K"
            else:
                dl_str = str(downloads)
            print(f"  {m['id']:<50} {dl_str:>10} downloads")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
print("💡 Recommended models for NPU acceleration:")
print("=" * 70)
print("""
For Qualcomm Snapdragon NPU with QNN backend:

1. Xenova/whisper-tiny      - Smallest, fastest, lower quality
2. Xenova/whisper-base      - Good balance for simple tasks  
3. Xenova/whisper-small     - Better quality, still fast
4. Xenova/whisper-medium    - High quality, slower
5. Xenova/whisper-large-v3  - Best quality, slowest

You can also export any HuggingFace Whisper model to ONNX using:
  python -m optimum.exporters.onnx --model openai/whisper-medium ./whisper-medium-onnx
""")
