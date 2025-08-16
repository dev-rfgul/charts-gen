# Deployment Guide for Charts Generator

## Production Deployment Issues & Solutions

### Common Issue: "unknown file extension: .mp4"

This error occurs when FFmpeg is not available in the production environment. The app now includes automatic fallbacks to handle this.

### Quick Fix Options:

#### Option 1: Install FFmpeg in Production
For most cloud platforms (Heroku, Railway, etc.):

1. **Heroku**: Add the `heroku-buildpack-apt` buildpack and create an `Aptfile`:
   ```
   ffmpeg
   ```

2. **Railway/Render**: Add system dependencies in your build configuration:
   ```bash
   apt-get update && apt-get install -y ffmpeg
   ```

3. **Docker**: Add to your Dockerfile:
   ```dockerfile
   RUN apt-get update && apt-get install -y ffmpeg
   ```

#### Option 2: Use Built-in Fallbacks (Recommended)
The app now automatically:
1. Tries FFmpeg first (MP4)
2. Falls back to imageio-ffmpeg (MP4)
3. Finally uses Pillow (GIF)

### Requirements
Make sure your `requirements.txt` includes:
```
streamlit
matplotlib
pandas
numpy
seaborn
pillow
imageio
imageio-ffmpeg
```

### Testing Locally
To test the fallback behavior:
```bash
# Temporarily disable FFmpeg
sudo apt remove ffmpeg  # or rename ffmpeg executable
streamlit run line-chart.py
# Should fallback to GIF generation
```

### Platform-Specific Notes

#### Streamlit Cloud
- FFmpeg is available by default
- Should work with MP4 generation

#### Heroku
- Needs buildpack for FFmpeg
- Otherwise falls back to GIF

#### Railway/Render
- May need system package installation
- GIF fallback works reliably

#### Local Development
- Install FFmpeg: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (Mac)

### Troubleshooting

1. **Check system status in app**: Expand the "System Status" section to see what's available
2. **Check logs**: Look for fallback warnings in your deployment logs
3. **Test with small data**: Use minimal datasets first to verify functionality

### Performance Tips

- GIF files are larger than MP4 but more universally supported
- For production, consider setting up proper FFmpeg installation for better performance
- Reduce animation complexity for faster processing in constrained environments
