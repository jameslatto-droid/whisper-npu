#!/usr/bin/env python3
"""
Transcription Quality Checker

Detects common issues like:
- Hallucinations (repetitive text)
- Low vocabulary diversity
- Common error patterns
"""

import re
from collections import Counter
from pathlib import Path


def check_transcription_quality(text: str) -> dict:
    """
    Analyze transcription for potential issues.
    
    Args:
        text: Transcription text to analyze
        
    Returns:
        Dictionary with quality metrics and issues
    """
    if not text or not text.strip():
        return {
            "word_count": 0,
            "unique_words": 0,
            "vocabulary_diversity": 0,
            "issues": ["Empty transcription"],
            "quality_score": 0,
            "is_hallucination": True
        }
    
    words = text.lower().split()
    word_counts = Counter(words)
    
    issues = []
    is_hallucination = False
    
    # Check for repetitive words (hallucination indicator)
    for word, count in word_counts.most_common(10):
        if count > 20 and len(word) > 2:
            ratio = count / len(words)
            if ratio > 0.05:  # More than 5% of text is one word
                issues.append(f"Repetitive word: '{word}' appears {count} times ({ratio:.1%})")
                if ratio > 0.15:
                    is_hallucination = True
    
    # Check for repetitive phrases (2+ repetitions of 10+ char phrases)
    phrase_patterns = re.findall(r'(.{10,50})\1{2,}', text, re.IGNORECASE)
    if phrase_patterns:
        issues.append(f"Repetitive phrases: {len(phrase_patterns)} patterns detected")
        is_hallucination = True
    
    # Check for low vocabulary diversity
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.3:
        issues.append(f"Low vocabulary diversity: {unique_ratio:.1%} unique words")
        if unique_ratio < 0.15:
            is_hallucination = True
    
    # Check for common hallucination patterns
    hallucination_patterns = [
        (r'(okay[.\s,]*){10,}', "Repeated 'okay'"),
        (r"(I'm going to .{0,30}){5,}", "Repeated 'I'm going to...'"),
        (r"(I'm not sure .{0,30}){4,}", "Repeated 'I'm not sure...'"),
        (r"(I don't .{0,30}){5,}", "Repeated 'I don't...'"),
        (r"(I think .{0,30}){5,}", "Repeated 'I think...'"),
        (r'(\b\w+\b)(\s+\1){9,}', "Word repeated 10+ times consecutively"),
        (r"(the first thing .{0,50}){3,}", "Repetitive 'the first thing'"),
        (r"(the second thing .{0,50}){3,}", "Repetitive 'the second thing'"),
    ]
    
    for pattern, description in hallucination_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            issues.append(f"Hallucination pattern: {description}")
            is_hallucination = True
    
    # Calculate quality score (0-100)
    base_score = 100
    
    # Deduct for issues
    base_score -= len(issues) * 15
    
    # Deduct for low diversity
    if unique_ratio < 0.5:
        base_score -= int((0.5 - unique_ratio) * 50)
    
    # Bonus for good diversity
    if unique_ratio > 0.6:
        base_score += 5
    
    quality_score = max(0, min(100, base_score))
    
    # Override score if definite hallucination
    if is_hallucination:
        quality_score = min(quality_score, 25)
    
    return {
        "word_count": len(words),
        "unique_words": len(set(words)),
        "vocabulary_diversity": unique_ratio,
        "issues": issues,
        "quality_score": quality_score,
        "is_hallucination": is_hallucination
    }


def compare_transcriptions(before: str, after: str) -> dict:
    """Compare two transcriptions and show improvement."""
    before_quality = check_transcription_quality(before)
    after_quality = check_transcription_quality(after)
    
    improvement = after_quality["quality_score"] - before_quality["quality_score"]
    
    return {
        "before": before_quality,
        "after": after_quality,
        "improvement": improvement,
        "improved": improvement > 0,
        "hallucination_fixed": before_quality["is_hallucination"] and not after_quality["is_hallucination"]
    }


def print_quality_report(result: dict, title: str = "Quality Report"):
    """Print a formatted quality report."""
    print(f"\n{'=' * 50}")
    print(f"📊 {title}")
    print('=' * 50)
    print(f"Words: {result['word_count']}")
    print(f"Unique words: {result['unique_words']}")
    print(f"Vocabulary diversity: {result['vocabulary_diversity']:.1%}")
    
    # Quality score with color indicator
    score = result['quality_score']
    if score >= 80:
        indicator = "✅"
    elif score >= 50:
        indicator = "⚠️"
    else:
        indicator = "❌"
    
    print(f"Quality score: {indicator} {score}/100")
    
    if result['is_hallucination']:
        print(f"🚨 HALLUCINATION DETECTED")
    
    if result['issues']:
        print(f"\nIssues found ({len(result['issues'])}):")
        for issue in result['issues']:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ No issues detected")


def check_file(file_path: str) -> dict:
    """Check quality of a transcription file."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return check_transcription_quality(text)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_check.py <transcription_file> [comparison_file]")
        print("\nExamples:")
        print("  python quality_check.py transcript.txt")
        print("  python quality_check.py before.txt after.txt")
        sys.exit(1)
    
    file1 = sys.argv[1]
    result1 = check_file(file1)
    
    if "error" in result1:
        print(f"❌ {result1['error']}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        # Comparison mode
        file2 = sys.argv[2]
        result2 = check_file(file2)
        
        if "error" in result2:
            print(f"❌ {result2['error']}")
            sys.exit(1)
        
        print_quality_report(result1, f"BEFORE: {Path(file1).name}")
        print_quality_report(result2, f"AFTER: {Path(file2).name}")
        
        # Show improvement
        improvement = result2['quality_score'] - result1['quality_score']
        print(f"\n{'=' * 50}")
        if improvement > 0:
            print(f"📈 Improvement: +{improvement} points")
        elif improvement < 0:
            print(f"📉 Regression: {improvement} points")
        else:
            print(f"➡️ No change in quality score")
        
        if result1['is_hallucination'] and not result2['is_hallucination']:
            print("🎉 Hallucination fixed!")
    else:
        # Single file mode
        print_quality_report(result1, Path(file1).name)
