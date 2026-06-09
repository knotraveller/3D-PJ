你现在接手一个 3DGS 重建项目中的相机参数与 ray embedding 模块实现任务。项目背景如下：

我们使用 Blender 批量渲染 Objaverse 本地 `.glb` 文件，渲染参数固定为：

```text
--resolution 256
--ref_azimuths 0
--fov 30
--camera_radius 4.0
--target_radius 0.8
```

物体已经在 Blender 渲染脚本中被规范化到原点附近，最长边缩放到约 `2 * target_radius = 1.6`。相机位于半径为 `camera_radius=4.0` 的球面上，始终 look_at 原点。

当前 Zero123++ 风格的 7 个视角为：

```text
cond:
  azimuth = 0
  elevation = 0

targets:
  azimuths   = [30, 90, 150, 210, 270, 330]
  elevations = [20, -10, 20, -10, 20, -10]
```

请实现一个完整、清晰、可复用的相机参数与 ray embedding 模块，使得：

1. 渲染阶段可以计算并保存每个视角的相机内参、外参。
2. 训练阶段可以读取这些相机参数。
3. 训练时只需要调用 `get_embedding(...)`，就能获得模型输入所需的 ray embedding。
4. ray embedding 用于和 RGB 拼接，最终输入形状为：

   ```text
   RGB:       [B, V, 3, H, W]
   embedding: [B, V, 6, H, W]
   concat:    [B, V, 9, H, W]
   ```

   其中 `V=7`，`H=W=256`。

请按照以下要求实现。

---

# 一、坐标系约定

采用 Blender 相机坐标系：

```text
camera local +X: right
camera local +Y: up
camera local -Z: forward
world +Z: up
```

球坐标放置相机时，使用如下约定：

```python
x = radius * cos(elevation) * sin(azimuth)
y = -radius * cos(elevation) * cos(azimuth)
z = radius * sin(elevation)
```

其中 `azimuth` 和 `elevation` 都是角度制输入，计算时转成弧度。

相机始终看向世界坐标原点 `(0, 0, 0)`。

---

# 二、需要实现的文件

请新增或完善以下文件：

```text
camera_utils.py
ray_utils.py
test_camera_rays.py
```

如果项目已有类似文件，可以在已有结构中合并，但请保持函数接口清晰。

---

# 三、camera_utils.py 要实现的功能

## 1. 计算内参矩阵 K

实现：

```python
def get_intrinsics_from_fov(resolution: int, fov_deg: float) -> np.ndarray:
    ...
```

对于正方形图像：

```python
W = H = resolution
fx = fy = 0.5 * W / tan(0.5 * fov)
cx = W / 2.0
cy = H / 2.0
```

返回：

```python
K = [
  [fx, 0, cx],
  [0, fy, cy],
  [0, 0, 1]
]
```

对于 `resolution=256, fov=30`，焦距应该约为：

```text
fx = fy ≈ 477.70
cx = cy = 128
```

---

## 2. 根据 azimuth/elevation/radius 计算相机位置

实现：

```python
def camera_position_from_spherical(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
) -> np.ndarray:
    ...
```

返回 shape 为 `[3]` 的相机世界坐标位置。

例如：

```python
camera_position_from_spherical(0, 0, 4.0)
```

应返回接近：

```text
[0, -4, 0]
```

---

## 3. 计算 look_at 外参

实现纯 numpy 版本：

```python
def look_at_c2w(
    camera_position: np.ndarray,
    target: np.ndarray = np.array([0.0, 0.0, 0.0]),
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    ...
```

要求返回 Blender convention 下的 `c2w`，shape 为 `[4, 4]`。

注意 Blender 相机看向 local `-Z` 方向，因此：

```text
camera local -Z 对应 forward direction
camera local +X 对应 right direction
camera local +Y 对应 up direction
```

可以构造：

```python
forward = normalize(target - camera_position)
right = normalize(cross(forward, up))
true_up = cross(right, forward)

R_c2w columns:
  col 0 = right
  col 1 = true_up
  col 2 = -forward
```

因为 local `-Z` 是 forward，所以 local `+Z` 对应 `-forward`。

最终：

```python
c2w[:3, :3] = R_c2w
c2w[:3, 3] = camera_position
```

并返回 `c2w`。

---

## 4. 生成 Zero123++ 风格 7 个相机

实现：

```python
def get_zero123pp_camera_specs(
    ref_azimuth: float = 0.0,
    input_elevation: float = 0.0,
    radius: float = 4.0,
    fov_deg: float = 30.0,
    resolution: int = 256,
) -> dict:
    ...
```

返回一个字典，包含：

```python
{
  "resolution": 256,
  "fov": 30.0,
  "radius": 4.0,
  "views": [
    {
      "name": "cond",
      "index": 0,
      "azimuth": 0.0,
      "elevation": 0.0,
      "relative_azimuth": 0.0,
      "K": ...,
      "c2w": ...,
      "w2c": ...
    },
    {
      "name": "target_000",
      "index": 1,
      "azimuth": 30.0,
      "elevation": 20.0,
      "relative_azimuth": 30.0,
      "K": ...,
      "c2w": ...,
      "w2c": ...
    },
    ...
  ]
}
```

其中 6 个 target 视角为：

```python
relative_azimuths = [30, 90, 150, 210, 270, 330]
target_elevations = [20, -10, 20, -10, 20, -10]
```

因为当前 `ref_azimuth=0`，所以绝对 azimuth 就等于这些相对 azimuth。函数仍然需要支持任意 `ref_azimuth`，即：

```python
target_azimuth = ref_azimuth + relative_azimuth
```

---

## 5. 保存和读取 cameras.json

实现：

```python
def save_cameras_json(camera_specs: dict, json_path: str) -> None:
    ...
```

和：

```python
def load_cameras_json(json_path: str) -> dict:
    ...
```

要求：

* JSON 中矩阵用普通 list 保存，不要保存 numpy array。
* 读取后可以选择转回 numpy array。
* 保存内容包括：

  * resolution
  * fov
  * radius
  * 每个 view 的 name/index/azimuth/elevation/relative_azimuth
  * K
  * c2w
  * w2c

---

# 四、ray_utils.py 要实现的功能

## 1. 根据 K 和 c2w 生成 rays_o, rays_d

实现 numpy 版本：

```python
def get_rays_np(
    K: np.ndarray,
    c2w: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

返回：

```text
rays_o: [H, W, 3]
rays_d: [H, W, 3]
```

使用像素中心：

```python
u = np.arange(W) + 0.5
v = np.arange(H) + 0.5
```

Blender camera convention 下：

```python
x_cam = (u - cx) / fx
y_cam = -(v - cy) / fy
z_cam = -1
```

所以：

```python
dirs_cam = normalize([x_cam, y_cam, -1])
dirs_world = dirs_cam @ R_c2w.T
origin_world = c2w[:3, 3]
```

返回所有像素的射线原点和方向。

---

## 2. 根据 rays_o, rays_d 生成 Plücker ray embedding

实现：

```python
def get_plucker_np(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    order: str = "dm",
) -> np.ndarray:
    ...
```

其中：

```python
moment = np.cross(rays_o, rays_d)
```

如果 `order="dm"`，返回：

```text
[d, m]
```

即：

```python
embedding = concat([rays_d, moment], axis=-1)
```

shape 为：

```text
[H, W, 6]
```

如果 `order="md"`，返回：

```text
[m, d]
```

默认使用 `order="dm"`。

---

## 3. 实现 PyTorch 版本

训练时会用 PyTorch，因此实现：

```python
def get_rays_torch(
    K: torch.Tensor,
    c2w: torch.Tensor,
    resolution: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

支持输入：

```text
K:   [3, 3] 或 [B, V, 3, 3]
c2w: [4, 4] 或 [B, V, 4, 4]
```

返回：

```text
如果输入单视角:
  rays_o: [H, W, 3]
  rays_d: [H, W, 3]

如果输入 batch 多视角:
  rays_o: [B, V, H, W, 3]
  rays_d: [B, V, H, W, 3]
```

实现时注意 broadcast，不要写死 batch size。

---

## 4. 实现训练时统一入口 get_embedding

实现：

```python
def get_embedding(
    K: torch.Tensor,
    c2w: torch.Tensor,
    resolution: int = 256,
    embedding_type: str = "plucker",
    order: str = "dm",
    channel_first: bool = True,
) -> torch.Tensor:
    ...
```

要求：

* 输入：

  ```text
  K:   [B, V, 3, 3]
  c2w: [B, V, 4, 4]
  ```
* 输出默认：

  ```text
  embedding: [B, V, 6, H, W]
  ```
* 如果 `channel_first=False`，则输出：

  ```text
  [B, V, H, W, 6]
  ```

当 `embedding_type="plucker"` 时：

```python
rays_o, rays_d = get_rays_torch(...)
moment = torch.cross(rays_o, rays_d, dim=-1)
embedding = concat([rays_d, moment], dim=-1)
```

当 `embedding_type="ray_dir"` 时，只返回 ray direction：

```text
[B, V, 3, H, W]
```

默认使用 Plücker embedding。

---

# 五、Dataset 使用方式

请给出一个示例，说明训练时如何直接上手使用：

```python
from ray_utils import get_embedding

images = batch["images"]      # [B, V, 3, H, W]
K = batch["K"]                # [B, V, 3, 3]
c2w = batch["c2w"]            # [B, V, 4, 4]

ray_emb = get_embedding(
    K=K,
    c2w=c2w,
    resolution=256,
    embedding_type="plucker",
    order="dm",
    channel_first=True,
)                              # [B, V, 6, H, W]

model_input = torch.cat([images, ray_emb], dim=2)
# model_input: [B, V, 9, H, W]
```

---

# 六、测试要求

请实现 `test_camera_rays.py`，至少测试以下内容：

1. `get_intrinsics_from_fov(256, 30)` 的 `fx/fy` 是否约为 `477.7`。
2. `camera_position_from_spherical(0, 0, 4.0)` 是否约为 `[0, -4, 0]`。
3. `get_zero123pp_camera_specs(...)` 是否返回 7 个 views。
4. 每个 view 的 K 是否 shape 为 `[3, 3]`，c2w/w2c 是否 shape 为 `[4, 4]`。
5. `c2w @ w2c` 是否接近单位矩阵。
6. `get_rays_np(...)` 输出 shape 是否为 `[256, 256, 3]`。
7. 中心像素 ray direction 是否大致指向原点：

   ```python
   center_ray_d ≈ normalize(-camera_position)
   ```

   注意由于像素中心和主点存在 0.5 偏移，允许小误差。
8. `get_embedding(...)` 对 batch 输入是否输出 `[B, V, 6, 256, 256]`。
9. 拼接 RGB 后是否得到 `[B, V, 9, 256, 256]`。

---

# 七、代码风格要求

* 代码清晰，函数有 docstring。
* 不要把参数写死在函数内部，默认值可以是：

  ```python
  resolution=256
  fov=30
  radius=4.0
  ref_azimuth=0
  ```
* 保持 numpy 和 torch 版本的坐标约定完全一致。
* JSON 保存与读取要稳定，不要因为 numpy array 无法序列化而报错。
* 如果项目中有 Blender 渲染脚本，请在渲染完成时调用 `save_cameras_json(...)` 保存 `cameras.json`。
* 训练阶段不要保存 ray map 到磁盘，除非用户明确要求。默认只保存 K/c2w/w2c，训练时在线调用 `get_embedding(...)` 生成 ray embedding。
* 请在 README 或注释中说明：

  * `c2w` 是 camera-to-world。
  * `w2c` 是 world-to-camera。
  * Blender camera local `-Z` 是前方。
  * Plücker embedding 默认顺序是 `[direction, moment]`。
  * `moment = origin × direction`。
  * 训练模型输入为 RGB 与 ray embedding 在 channel 维拼接。

请完成实现后，给出最小使用示例，并确保所有测试可以通过。  
