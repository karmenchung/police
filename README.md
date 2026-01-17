## 模型下载
Qwen 模型：
```
huggingface-cli.exe download Qwen/Qwen3-4B-Instruct-2507 --local-dir .\Qwen/Qwen3-4B-Instruct-2507 --local-dir-use-symlinks False
```

下载 https://github.com/index-tts/index-tts 到 ./TTS
```
huggingface-cli.exe download IndexTeam/IndexTTS-2 --local-dir .\TTS\index-tts\checkpoints\ --local-dir-use-symlinks False
cd .\TTS\index-tts

$env:UV_CACHE_DIR = "F:\C_disk\uv\cache"
$env:UV_PYTHON_INSTALL_DIR = "F:\C_disk\uv\python"
$env:UV_TOOL_DIR = "F:\C_disk\uv\tool"
uv sync --extra webui

$env:HF_HOME = "F:\C_disk\hf_cache"
$env:HF_HUB_CACHE = "F:\C_disk\hf_cache\hub"
$env:TRANSFORMERS_CACHE = "F:\C_disk\hf_cache\transformers"
uv run .\webui.py
```