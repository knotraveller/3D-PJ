你现在接手一个 Objaverse 本地 `.glb` 批量渲染任务。请在当前项目中实现一个可运行的 Blender Python 渲染管线，用于把本地 `.glb` 文件渲染成和 Zero123++ 视角设置对齐的多视角训练数据。



目标：

给定一个包含很多 `.glb` 文件的本地目录，批量导入每个 `.glb`，规范化物体尺度和中心，渲染输入视角和 6 个目标视角，输出白底 RGB 图、alpha 前景 mask、相机参数 JSON 和元信息 JSON。



请实现脚本：



```text

render\_objaverse.py

```



要求通过 Blender 命令行运行，而不是普通 Python 运行：



```bash

blender -b --python render\_objaverse.py -- \\

&nbsp; --input\_dir ./PATH/TO/GLBS \\

&nbsp; --output\_dir ./renders \\

&nbsp; --resolution 256 \\

&nbsp; --ref\_azimuths 0 90 180 270 \\

&nbsp; --fov 30 \\

&nbsp; --camera\_radius 4.0 \\

&nbsp; --target\_radius 0.8

```



命令行参数要求：



```text

--input\_dir: 输入 .glb 文件目录，递归查找所有 .glb

--output\_dir: 输出目录

--resolution: 渲染分辨率，默认 256

--ref\_azimuths: 输入参考方位角列表，默认 0 90 180 270

--input\_elevation: 输入视角 elevation，默认 0

--fov: 相机 FOV，默认 30

--camera\_radius: 相机半径，默认 4.0

--target\_radius: 物体规范化后的目标半径，默认 0.8

--engine: 渲染引擎，默认 EEVEE，可选 CYCLES

--skip\_existing: 如果输出目录已存在且完整，则跳过

--max\_files: 可选，只处理前 N 个文件，方便测试

```



核心流程：



1\. 对每个 `.glb` 文件：



&nbsp;  \* 清空 Blender 当前 scene。

&nbsp;  \* 使用 `bpy.ops.import\_scene.gltf(filepath=...)` 导入 `.glb`。

&nbsp;  \* 删除导入文件自带的 camera 和 light，避免不同文件光照不统一。

&nbsp;  \* 保留 mesh、material、texture。

&nbsp;  \* 如果没有任何 mesh，则记录失败并跳过。



2\. 物体规范化：



&nbsp;  \* 遍历 scene 中所有 `obj.type == "MESH"` 的对象。

&nbsp;  \* 根据所有 mesh 的 world-space bounding box 计算整体 `bbox\_min` 和 `bbox\_max`。

&nbsp;  \* 计算 `center = (bbox\_min + bbox\_max) / 2`。

&nbsp;  \* 计算最大边长 `max\_dim = max(bbox\_max - bbox\_min)`。

&nbsp;  \* 对所有 mesh 对象进行平移和缩放，使物体中心位于原点，最长边缩放到 `2 \* target\_radius`。

&nbsp;  \* 更新 scene。

&nbsp;  \* 在 `meta.json` 中保存原始 bbox、center、scale、target\_radius。



3\. 相机设置：



&nbsp;  \* 新建一个 camera。

&nbsp;  \* FOV 设置为参数 `--fov`，默认 30°。

&nbsp;  \* 使用球坐标放置相机，物体中心为原点。

&nbsp;  \* 相机始终 look\_at 原点。

&nbsp;  \* 坐标约定可以采用：



&nbsp;    \* azimuth 绕 z 轴旋转；

&nbsp;    \* elevation 表示相机相对水平面的仰角；

&nbsp;    \* z 轴为上方向；

&nbsp;    \* camera radius 为 `--camera\_radius`。

&nbsp;  \* 保存每个视角的 azimuth、elevation、radius、fov、c2w、w2c、K。



4\. Zero123++ 风格视角：

&nbsp;  对每个参考角 `theta0`：



&nbsp;  \* 输入视角：



&nbsp;    ```text

&nbsp;    azimuth = theta0

&nbsp;    elevation = input\_elevation，默认 0

&nbsp;    ```

&nbsp;  \* 6 个目标视角：



&nbsp;    ```text

&nbsp;    relative\_azimuths = \[30, 90, 150, 210, 270, 330]

&nbsp;    target\_elevations = \[20, -10, 20, -10, 20, -10]

&nbsp;    target\_azimuth = theta0 + relative\_azimuth

&nbsp;    ```

&nbsp;  \* 每个 `theta0` 单独输出一个样本目录。



5\. 输出目录结构：

&nbsp;  对于输入文件 `xxx.glb`，以及参考角 `theta0=90`，输出：



&nbsp;  ```text

&nbsp;  output\_dir/

&nbsp;    xxx/

&nbsp;      ref\_090/

&nbsp;        meta.json

&nbsp;        cameras.json

&nbsp;        cond/

&nbsp;          rgb.png

&nbsp;          alpha.png

&nbsp;        targets/

&nbsp;          000\_rgb.png

&nbsp;          000\_alpha.png

&nbsp;          001\_rgb.png

&nbsp;          001\_alpha.png

&nbsp;          002\_rgb.png

&nbsp;          002\_alpha.png

&nbsp;          003\_rgb.png

&nbsp;          003\_alpha.png

&nbsp;          004\_rgb.png

&nbsp;          004\_alpha.png

&nbsp;          005\_rgb.png

&nbsp;          005\_alpha.png

&nbsp;  ```



6\. 渲染设置：



&nbsp;  \* 分辨率为 `resolution × resolution`。

&nbsp;  \* 使用透明背景渲染 RGBA，目的是得到 alpha 通道。

&nbsp;  \* 将 RGBA 中的 alpha 另存为 `alpha.png`。

&nbsp;  \* 将 RGBA 按白色背景合成为 RGB：



&nbsp;    ```python

&nbsp;    rgb\_white = alpha \* rgb + (1 - alpha) \* 1.0

&nbsp;    ```



&nbsp;    保存为 `rgb.png`。

&nbsp;  \* 不要从白底图反推 mask。

&nbsp;  \* 世界背景、灯光设置保持统一。

&nbsp;  \* 添加固定 area light 或 sun light，避免不同 `.glb` 自带灯光干扰。

&nbsp;  \* 推荐默认使用 EEVEE，保证批量渲染速度。



7\. 相机 JSON：

&nbsp;  `cameras.json` 需要包含：



&nbsp;  ```json

&nbsp;  {

&nbsp;    "fov": 30,

&nbsp;    "camera\_radius": 4.0,

&nbsp;    "input": {

&nbsp;      "azimuth": 0,

&nbsp;      "elevation": 0,

&nbsp;      "c2w": \[\[...]],

&nbsp;      "w2c": \[\[...]],

&nbsp;      "K": \[\[...]]

&nbsp;    },

&nbsp;    "targets": \[

&nbsp;      {

&nbsp;        "index": 0,

&nbsp;        "relative\_azimuth": 30,

&nbsp;        "azimuth": 30,

&nbsp;        "elevation": 20,

&nbsp;        "c2w": \[\[...]],

&nbsp;        "w2c": \[\[...]],

&nbsp;        "K": \[\[...]]

&nbsp;      }

&nbsp;    ]

&nbsp;  }

&nbsp;  ```



8\. 失败处理：



&nbsp;  \* 某个 `.glb` 导入失败时，不要中断整个批处理。

&nbsp;  \* 记录到 `failed.jsonl`。

&nbsp;  \* 失败记录包含：



&nbsp;    ```text

&nbsp;    file path

&nbsp;    error message

&nbsp;    traceback

&nbsp;    ```

&nbsp;  \* 成功处理的文件记录到 `rendered.jsonl`。



9\. 数据过滤：



&nbsp;  \* 渲染后读取 alpha mask，计算前景比例：



&nbsp;    ```python

&nbsp;    foreground\_ratio = alpha.mean()

&nbsp;    ```

&nbsp;  \* 如果所有视角 foreground\_ratio 都过小，比如 `< 0.02`，则标记该样本为异常。

&nbsp;  \* 如果 foreground\_ratio 过大，比如 `> 0.95`，也标记异常。

&nbsp;  \* 异常样本可以先保留，但在 `meta.json` 中写入 warning 字段。

&nbsp;  \* 不要直接删除，方便后续检查。



10\. 代码质量：



\* 代码结构清晰、简洁易懂，不要保留过多而无用的接口。函数化实现。

\* 至少包含以下函数：



&nbsp; ```python

&nbsp; clear\_scene()

&nbsp; import\_glb(path)

&nbsp; compute\_scene\_bbox()

&nbsp; normalize\_scene(target\_radius)

&nbsp; setup\_renderer(resolution, engine)

&nbsp; setup\_lighting()

&nbsp; create\_camera(fov)

&nbsp; set\_camera\_pose(camera, azimuth, elevation, radius)

&nbsp; look\_at(camera, target)

&nbsp; get\_camera\_matrices(camera)

&nbsp; render\_rgba(output\_path)

&nbsp; save\_rgb\_and\_alpha(rgba\_path, rgb\_path, alpha\_path)

&nbsp; process\_one\_glb(glb\_path, output\_dir, args)

&nbsp; main()

&nbsp; ```

\* 尽量只依赖 Blender 自带库；如需要图像处理，优先使用 Blender 内置图像保存能力，必要时可用 `numpy` / `PIL`，但要提醒用户 Blender Python 环境可能没有这些包。

\* 如果使用 `PIL`，请在 README 里说明如何给 Blender Python 安装 Pillow，或者提供不用 PIL 的 fallback 方案。



11\. README：

&nbsp;   请同时生成一个简短 `README\_render.md`，说明：



\* 如何安装 / 打开 Blender。

\* 如何用命令行运行脚本。

\* Windows / Linux 下的命令示例。

\* 输出目录结构。

\* 每个输出文件的含义。

\* 常见问题：



&nbsp; \* 为什么要用 `blender -b --python` 而不是 `python render\_objaverse.py`；

&nbsp; \* 为什么要先 center + scale；

&nbsp; \* alpha.png 表示前景 mask，不是材质透明度，所以是0/1二值的；

&nbsp; \* 为什么 RGB 使用白底合成；

&nbsp; \* 如何只测试前几个文件。



12\. 最后请给出一个最小测试命令：



```bash

blender -b --python render\_objaverse.py -- \\

&nbsp; --input\_dir ./PATH/TO/YOUR/GLBS \\

&nbsp; --output\_dir ./renders\_test \\

&nbsp; --resolution 128 \\

&nbsp; --ref\_azimuths 0 \\

&nbsp; --max\_files 3

```



实现完成后，请先检查脚本是否存在语法问题，并确保目录不存在时会自动创建。



