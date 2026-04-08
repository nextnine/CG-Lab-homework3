import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100

# Bezier 采样数
BEZIER_SEGMENTS = 1000

# 每段三次 B 样条采样数
BSPLINE_SAMPLES_PER_SPAN = 80

# 预留足够大的曲线点缓存
MAX_CURVE_POINTS = 20000

# =========================
# GPU Fields
# =========================
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 用于抗锯齿累积亮度
intensity = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))

gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)

# =========================
# CPU 曲线生成
# =========================
def de_casteljau(points, t):
    """CPU 递归 De Casteljau，适合 Bezier 模式"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)


def generate_bezier_points(control_points):
    """生成 Bezier 曲线采样点"""
    if len(control_points) < 2:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.zeros((BEZIER_SEGMENTS + 1, 2), dtype=np.float32)
    for i in range(BEZIER_SEGMENTS + 1):
        t = i / BEZIER_SEGMENTS
        pts[i] = de_casteljau(control_points, t)
    return pts


def uniform_cubic_bspline_point(p0, p1, p2, p3, u):
    """
    均匀三次 B 样条单段点计算
    基函数：
        B0 = (-u^3 + 3u^2 - 3u + 1) / 6
        B1 = ( 3u^3 - 6u^2 + 4) / 6
        B2 = (-3u^3 + 3u^2 + 3u + 1) / 6
        B3 = ( u^3 ) / 6
    """
    u2 = u * u
    u3 = u2 * u

    b0 = (-u3 + 3.0 * u2 - 3.0 * u + 1.0) / 6.0
    b1 = ( 3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
    b2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
    b3 = u3 / 6.0

    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return [x, y]


def generate_bspline_points(control_points):
    """
    生成均匀三次 B 样条采样点
    n 个控制点 -> n-3 段
    """
    n = len(control_points)
    if n < 4:
        return np.zeros((0, 2), dtype=np.float32)

    points = []

    span_count = n - 3
    for span in range(span_count):
        p0 = control_points[span]
        p1 = control_points[span + 1]
        p2 = control_points[span + 2]
        p3 = control_points[span + 3]

        # 前面的段不取终点，最后一段取终点，避免边界重复太多
        sample_count = BSPLINE_SAMPLES_PER_SPAN
        end = sample_count if span < span_count - 1 else sample_count + 1

        for j in range(end):
            u = j / sample_count
            points.append(uniform_cubic_bspline_point(p0, p1, p2, p3, u))

    return np.array(points, dtype=np.float32)


# =========================
# GPU 渲染
# =========================
@ti.kernel
def clear_buffers():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
        intensity[i, j] = 0.0


@ti.func
def gaussian_weight(dist2: ti.f32, sigma: ti.f32) -> ti.f32:
    return ti.exp(-dist2 / (2.0 * sigma * sigma))


@ti.kernel
def draw_curve_aa(n: ti.i32):
    """
    反走样绘制：
    对每个曲线采样点，在其周围 3x3 像素邻域内按距离分配权重。
    """
    sigma = 0.75
    strength = 0.5  # 单个采样点贡献，值越大线越亮越粗

    for i in range(n):
        pt = curve_points_field[i]

        # 转换到像素坐标系
        x = pt[0] * (WIDTH - 1)
        y = pt[1] * (HEIGHT - 1)

        base_x = ti.floor(x, ti.i32)
        base_y = ti.floor(y, ti.i32)

        # 3x3 邻域
        for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
            xi = base_x + dx
            yi = base_y + dy

            if 0 <= xi < WIDTH and 0 <= yi < HEIGHT:
                # 像素中心坐标
                cx = ti.cast(xi, ti.f32) + 0.5
                cy = ti.cast(yi, ti.f32) + 0.5

                dist2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
                w = gaussian_weight(dist2, sigma) * strength

                ti.atomic_add(intensity[xi, yi], w)

    # 将累积亮度映射到最终颜色
    for i, j in intensity:
        g = ti.min(intensity[i, j], 1.0)
        pixels[i, j] = ti.Vector([0.0, g, 0.0])


def main():
    window = ti.ui.Window("Bezier / B-Spline + Anti-Aliasing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    mode = "bezier"   # "bezier" or "bspline"
    dirty = True
    curve_count = 0

    print("操作说明：")
    print("  左键：添加控制点")
    print("  c   ：清空")
    print("  b   ：切换 Bezier / B-Spline")
    print("当前模式：Bezier")

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append([pos[0], pos[1]])
                    dirty = True
                    print(f"Added control point: {pos}")

            elif e.key == 'c':
                control_points = []
                dirty = True
                curve_count = 0
                print("Canvas cleared.")

            elif e.key == 'b':
                mode = "bspline" if mode == "bezier" else "bezier"
                dirty = True
                print(f"Switched mode to: {mode}")

        # 仅在控制点或模式变化时重算曲线
        if dirty:
            if mode == "bezier":
                curve_points_np = generate_bezier_points(control_points)
            else:
                curve_points_np = generate_bspline_points(control_points)

            curve_count = min(len(curve_points_np), MAX_CURVE_POINTS)

            if curve_count > 0:
                upload = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)
                upload[:curve_count] = curve_points_np[:curve_count]
                curve_points_field.from_numpy(upload)

            dirty = False

        clear_buffers()

        if curve_count > 0:
            draw_curve_aa(curve_count)

        canvas.set_image(pixels)

        # 画控制点与控制多边形
        current_count = len(control_points)
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)

            # 红色控制点
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))

            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)

                # 灰色控制折线
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        window.show()


if __name__ == '__main__':
    main()