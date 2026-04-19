# MusicSfx-Fusion：音乐生成与音效融合

本项目整体围绕三个模块展开：音乐生成、环境音生成、音频合成。

模型组合：MusicGen 音乐生成 + AudioLDM 2 音效生成

音频融合优化：

时间对齐（长度匹配）
淡入淡出（提升自然度）
混音参数学习（自动 gain / EQ / reverb）
伪标签构建（无需人工标注）
目标是从生成音乐走向生成完整声音场景。

## 🚀 Quick Start

Follow the steps below to set up and run the project.

#### 1. Download MusicGen-Medium

```bash
python download_musicgen.py
```

#### 2. Download Stable Audio Open 1.0

```bash
python download_stableaudio.py
```

#### 3. Run the Application

```bash
python app.py
```

## References

This project is inspired by:
- https://github.com/minigenepig
