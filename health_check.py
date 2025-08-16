#!/usr/bin/env python3
"""
Health check script for Charts Generator app
Run this to diagnose video generation issues
"""

import sys
import subprocess
import importlib

def check_package(package_name):
    """Check if a package is available"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        return False

def main():
    print("🔧 Charts Generator - Health Check")
    print("=" * 40)
    
    # Check Python packages
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'matplotlib',
        'PIL',  # Pillow
    ]
    
    optional_packages = [
        'imageio',
        'imageio_ffmpeg',
        'seaborn'
    ]
    
    print("\n📦 Required Packages:")
    all_required_ok = True
    for pkg in required_packages:
        status = "✅" if check_package(pkg) else "❌"
        print(f"  {status} {pkg}")
        if not check_package(pkg):
            all_required_ok = False
    
    print("\n📦 Optional Packages:")
    for pkg in optional_packages:
        status = "✅" if check_package(pkg) else "⚠️"
        print(f"  {status} {pkg}")
    
    # Check FFmpeg
    print("\n🎬 Video Generation:")
    ffmpeg_ok = check_ffmpeg()
    imageio_ffmpeg_ok = check_package('imageio_ffmpeg')
    pillow_ok = check_package('PIL')
    
    print(f"  {'✅' if ffmpeg_ok else '❌'} FFmpeg (system)")
    print(f"  {'✅' if imageio_ffmpeg_ok else '⚠️'} imageio-ffmpeg")
    print(f"  {'✅' if pillow_ok else '❌'} Pillow (GIF fallback)")
    
    # Summary
    print("\n📊 Video Generation Capabilities:")
    if ffmpeg_ok:
        print("  ✅ MP4 videos (FFmpeg)")
    elif imageio_ffmpeg_ok:
        print("  ✅ MP4 videos (ImageIO-FFmpeg)")
    elif pillow_ok:
        print("  ⚠️  GIF only (Pillow fallback)")
    else:
        print("  ❌ No video generation available")
    
    print("\n🎯 Recommendations:")
    if not all_required_ok:
        print("  • Install missing required packages with: pip install -r requirements.txt")
    
    if not ffmpeg_ok and not imageio_ffmpeg_ok:
        print("  • For MP4 support, install FFmpeg or add imageio-ffmpeg to requirements")
        print("  • Current setup will generate GIF files instead")
    
    if not pillow_ok:
        print("  • Install Pillow for GIF fallback: pip install pillow")
    
    print("\n" + "=" * 40)
    
    if all_required_ok and (ffmpeg_ok or imageio_ffmpeg_ok or pillow_ok):
        print("✅ System ready for deployment!")
        return 0
    else:
        print("❌ System has issues that need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
