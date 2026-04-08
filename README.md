# 实验三：贝塞尔曲线（De Casteljau + Taichi 光栅化）

本项目使用 **Python + Taichi** 实现了交互式曲线绘制系统，核心功能包括：

- 基于 **De Casteljau 算法** 的贝塞尔曲线生成。
- 基于像素缓冲区（Frame Buffer）的曲线光栅化绘制。
- 鼠标/键盘交互式控制点编辑。
- 选做扩展：
  - 曲线反走样（Anti-Aliasing）。
  - 均匀三次 B 样条曲线及与贝塞尔曲线的模式切换。

---

## 1. 实验目标对应说明

本项目实现了如下目标：

1. **理解贝塞尔曲线几何意义**：通过交互式控制点实时观察曲线变化。
2. **实现 De Casteljau 算法**：在 CPU 端递归实现 `de_casteljau(points, t)`。
3. **掌握光栅化**：将归一化坐标映射到 `800x800` 像素缓冲区并点亮像素。
4. **掌握交互事件处理**：响应鼠标左键添加点、键盘 `c` 清空、`b` 切换曲线模式。

---

## 2. 环境与运行方式

## 2.1 依赖

- Python 3.12
- Taichi
- NumPy


---
## 2.2 运行


在仓库根目录执行：

```bash
python work3/main.py
```

程序启动后会打开 `800x800` 窗口进行实时绘制。

---

## 3. 核心实现说明

## 3.1 参数与显存（Field）预分配

在 `work3/main.py` 中完成了常量与 GPU 缓冲区预分配：

- 屏幕大小：`WIDTH = 800`, `HEIGHT = 800`
- 最大控制点：`MAX_CONTROL_POINTS = 100`
- 贝塞尔采样数：`BEZIER_SEGMENTS = 1000`
- B 样条每段采样：`BSPLINE_SAMPLES_PER_SPAN = 80`
- 曲线缓存上限：`MAX_CURVE_POINTS = 20000`

主要 Field：

- `pixels`：存放最终 RGB 图像。
- `intensity`：反走样累积亮度缓冲。
- `curve_points_field`：CPU 批量上传曲线点到 GPU。
- `gui_points` / `gui_indices`：固定容量对象池，用于绘制控制点与控制折线。

---

## 3.2 De Casteljau 算法（Bezier）

函数：`de_casteljau(points, t)`

- 输入：控制点列表与参数 `t∈[0,1]`
- 过程：递归线性插值，逐层将 `n` 个点缩减到 `1` 个点。
- 输出：对应参数下的曲线点坐标 `[x, y]`。

`generate_bezier_points(control_points)` 在 CPU 端采样 `1001` 个点，用于后续批量渲染。

---

## 3.3 B 样条曲线（选做）

本项目额外实现了 **均匀三次 B 样条**：

- 单段计算函数：`uniform_cubic_bspline_point(p0, p1, p2, p3, u)`
- 全曲线采样函数：`generate_bspline_points(control_points)`

性质：

- 当控制点数 `n >= 4` 时可生成曲线。
- 共 `n-3` 段，每段独立采样并拼接。
- 支持按键 `b` 与贝塞尔模式实时切换，便于比较“全局控制 vs 局部控制”。

---

## 3.4 GPU 绘制内核与反走样（选做）

### 清屏

- `clear_buffers()`：并行清空 `pixels` 和 `intensity`。

### 反走样绘制

- `draw_curve_aa(n)`：对每个曲线采样点，在其周围 `3x3` 邻域内按高斯权重分配亮度；
- 使用 `ti.atomic_add` 对邻域像素亮度累积，最后映射为绿色通道得到平滑曲线。

该方案避免了“仅点亮单像素”导致的明显锯齿。

---

## 3.5 CPU/GPU 批处理策略

本实现遵循题目强调的性能原则：

1. 在 CPU 上一次性计算所有曲线采样点。
2. 通过 `curve_points_field.from_numpy(...)` 一次性上传到 GPU。
3. 在 GPU kernel 中并行完成像素写入。

这样避免了 Python 循环中频繁跨 CPU/GPU 边界写像素导致的严重卡顿。

---

## 4. 交互说明

运行后可使用以下操作：

- **鼠标左键（LMB）**：添加控制点（上限 100）。
- **`c` 键**：清空控制点与曲线。
- **`b` 键**：切换 `Bezier` / `B-Spline` 模式。

显示内容：

- 红色圆点：控制点。
- 灰色折线：控制多边形。
- 绿色曲线：当前模式下生成的曲线（带反走样）。

---

## 5. 与实验任务清单的对应关系

---

## 6. 文件结构

```text
.
├── README.md
└── work3
    ├── main.py
    └── imgui.ini
```

---

## 7. 结果
项目效果图如下：

![ep48PAYY_converted](https://github.com/user-attachments/assets/2fc1860e-99f6-4ebd-98d6-02ad2b44e106)

