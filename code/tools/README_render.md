# Objaverse GLB 批量渲染

本目录提供 `render_objaverse.py`，用于在 Blender 后台模式中把本地 `.glb` 文件批量渲染成 Zero123++ 风格的 1 个输入视角和 6 个目标视角训练数据。

## 安装和打开 Blender

1. 从 Blender 官网安装 Blender。
2. 确认命令行可以找到 Blender：
   - Windows 可以把 Blender 安装目录加入 `PATH`，或直接使用 `C:\Program Files\Blender Foundation\Blender <version>\blender.exe`。
   - Linux 通常可以使用包管理器安装，或下载官方压缩包后运行其中的 `blender`。
3. 本脚本必须通过 Blender Python 运行，不需要额外安装 Pillow、numpy 等 Python 包。

## 运行命令

在 `code/` 目录下运行：

```bash
blender -b --python render_objaverse.py -- \
  --input_dir ./PATH/TO/GLBS \
  --output_dir ./renders \
  --resolution 256 \
  --ref_azimuths 0 90 180 270 \
  --fov 30 \
  --camera_radius 4.0 \
  --target_radius 0.8
```

从项目根目录运行时，把脚本路径改为：

```bash
blender -b --python code/render_objaverse.py -- \
  --input_dir ./PATH/TO/GLBS \
  --output_dir ./renders
```

Windows PowerShell 示例：

```powershell
blender -b --python .\render_objaverse.py -- `
  --input_dir .\PATH\TO\GLBS `
  --output_dir .\renders `
  --resolution 256 `
  --ref_azimuths 0 90 180 270
```

Linux 示例：

```bash
blender -b --python ./render_objaverse.py -- \
  --input_dir ./PATH/TO/GLBS \
  --output_dir ./renders \
  --resolution 256 \
  --ref_azimuths 0 90 180 270
```

## 输出结构

输入文件 `xxx.glb`，参考方位角 `90`，输出为：

```text
output_dir/
  xxx/
    ref_090/
      meta.json
      cameras.json
      cond/
        rgb.png
        alpha.png
      targets/
        000_rgb.png
        000_alpha.png
        001_rgb.png
        001_alpha.png
        002_rgb.png
        002_alpha.png
        003_rgb.png
        003_alpha.png
        004_rgb.png
        004_alpha.png
        005_rgb.png
        005_alpha.png
  rendered.jsonl
  failed.jsonl
```

文件含义：

- `cond/rgb.png`：输入视角的白底 RGB 图。
- `cond/alpha.png`：输入视角的二值前景 mask。
- `targets/*_rgb.png`：6 个目标视角的白底 RGB 图。
- `targets/*_alpha.png`：6 个目标视角的二值前景 mask。
- `cameras.json`：输入视角和目标视角的 azimuth、elevation、radius、fov、`c2w`、`w2c`、`K`。
- `meta.json`：源文件、规范化前后的 bbox、center、scale、前景比例和 warning。
- `rendered.jsonl`：成功渲染或跳过的文件记录。
- `failed.jsonl`：导入或渲染失败的文件、错误信息和 traceback。

## 常见问题

为什么要用 `blender -b --python` 而不是 `python render_objaverse.py`？

脚本依赖 `bpy`、`mathutils` 和 Blender 渲染器，这些模块只在 Blender Python 环境中可用。`-b` 表示后台运行，适合批量渲染。

为什么要先 center + scale？

Objaverse 物体的原始坐标和尺度不统一。先把整体 bbox 中心移到原点，并把最长边缩放到 `2 * target_radius`，相机半径和 FOV 才能在不同物体之间保持可比。

`alpha.png` 是什么？

`alpha.png` 表示前景 mask，不是材质透明度图。脚本从透明背景渲染得到 alpha 后阈值化为 0/1 二值 mask，并直接保存，不从白底 RGB 反推。

为什么 RGB 使用白底合成？

Zero123++ 风格训练数据通常使用白色背景图。脚本先渲染透明 RGBA，再按 `rgb_white = alpha * rgb + (1 - alpha) * 1.0` 合成白底 RGB。

如何只测试前几个文件？

使用 `--max_files`：

```bash
blender -b --python render_objaverse.py -- \
  --input_dir ./PATH/TO/YOUR/GLBS \
  --output_dir ./renders_test \
  --resolution 128 \
  --ref_azimuths 0 \
  --max_files 3
```

已经渲染过的样本可以加 `--skip_existing`，当对应 `ref_xxx` 目录完整时会跳过。
