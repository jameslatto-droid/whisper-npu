#!/usr/bin/env python3
"""
Transcription Presets

Optimized configurations for different recording scenarios.
"""

PRESETS = {
    "meeting-bilingual": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 20,
        "description": "Mixed Spanish/English meetings with background noise",
        "use_cases": ["SEMA meetings", "bilingual conferences", "international calls"]
    },
    "meeting-quiet": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "none",
        "noise_reduction": 0,
        "description": "Clean meeting recordings in quiet rooms",
        "use_cases": ["conference rooms", "studio recordings", "podcasts"]
    },
    "meeting-noisy": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 25,
        "description": "Meetings with significant background noise",
        "use_cases": ["coffee shops", "open offices", "outdoor recordings"]
    },
    "phone-call": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 20,
        "description": "Phone or video calls with compression artifacts",
        "use_cases": ["Zoom calls", "phone recordings", "VoIP"]
    },
    "interview": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 15,
        "description": "One-on-one interviews",
        "use_cases": ["interviews", "podcasts", "dictation"]
    },
    "lecture": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 10,
        "description": "Lectures and presentations",
        "use_cases": ["university lectures", "webinars", "presentations"]
    },
    "quick-draft": {
        "model": "base",
        "language": "auto",
        "preprocess": "none",
        "noise_reduction": 0,
        "description": "Fast draft transcription (lower quality)",
        "use_cases": ["quick review", "initial pass", "testing"]
    },
    "fast": {
        "model": "tiny",
        "language": "auto",
        "preprocess": "none",
        "noise_reduction": 0,
        "description": "Fastest transcription (lowest quality)",
        "use_cases": ["real-time", "rapid prototyping"]
    },
    "balanced": {
        "model": "small",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 15,
        "description": "Good balance of speed and quality",
        "use_cases": ["general purpose", "mixed content"]
    },
    "best-quality": {
        "model": "large-v3-turbo",
        "language": "auto",
        "preprocess": "ffmpeg",
        "noise_reduction": 20,
        "description": "Highest quality transcription",
        "use_cases": ["professional transcription", "important recordings", "archival"]
    }
}


def get_preset(name: str) -> dict:
    """Get preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name].copy()


def list_presets() -> None:
    """Print all available presets."""
    print("\n📋 Available Transcription Presets")
    print("=" * 60)
    
    for name, config in PRESETS.items():
        print(f"\n🎯 {name}")
        print(f"   Model: {config['model']}")
        print(f"   Language: {config['language']}")
        print(f"   Preprocessing: {config['preprocess']}", end="")
        if config['preprocess'] != 'none':
            print(f" ({config['noise_reduction']}dB noise reduction)")
        else:
            print()
        print(f"   Description: {config['description']}")
        print(f"   Use cases: {', '.join(config['use_cases'])}")


def get_preset_args(name: str) -> list:
    """Get command-line arguments for a preset."""
    config = get_preset(name)
    args = [
        "--model", config["model"],
        "--language", config["language"],
        "--preprocess", config["preprocess"],
    ]
    if config["preprocess"] != "none":
        args.extend(["--noise-reduction", str(config["noise_reduction"])])
    return args


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        preset_name = sys.argv[1]
        if preset_name in ["--list", "-l", "list"]:
            list_presets()
        else:
            try:
                config = get_preset(preset_name)
                print(f"\n🎯 Preset: {preset_name}")
                print(f"   {config['description']}")
                print(f"\nCommand-line arguments:")
                args = get_preset_args(preset_name)
                print(f"   {' '.join(args)}")
            except ValueError as e:
                print(f"❌ {e}")
                sys.exit(1)
    else:
        list_presets()
        print("\n" + "=" * 60)
        print("Usage: python presets.py <preset_name>")
        print("       python presets.py --list")
