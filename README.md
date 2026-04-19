# MusicSfx-Fusion: Music Generation and Sound Effect Fusion 

**MusicSfx-Fusion:** 音乐生成与音效融合 Gradio 平台

打开 Gradio 页面后分别输入音乐和音效 prompt，可直接聆听或保存融合渲染后的音频。

This project is structured around three core modules: music generation, environmental sound generation, and audio synthesis.

Model composition: MusicGen for music generation + Stable Audio Open 1.0 for sound effect generation.

Audio fusion optimization includes:
- Temporal alignment (length matching)
- Fade-in and fade-out (for improved naturalness)
- Mixing parameter learning (automatic gain / EQ / reverb)
- Pseudo-label construction (no manual annotation required)

The goal is to move beyond standalone music generation toward generating complete auditory scenes.

## 🚀 Quick Start

Follow the four steps below to set up and run the project.

#### 1. Download MusicGen-Medium

```bash
python download_musicgen.py
```

#### 2. Download Stable Audio Open 1.0

```bash
python download_stableaudio.py
```

#### 📦 Requirements

```bash
pip install librosa transformers accelerate soundfile gradio diffusers scipy torchsde
```

#### 3. Run the Application

```bash
python app.py
```

## References

This project is inspired by: https://github.com/minigenepig
