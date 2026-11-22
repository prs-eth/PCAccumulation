"""Benchmark viewer: segmented point clouds + per-stage timing, frame by frame.

Native Open3D window. Left: the point cloud of the current frame, colored by the pipeline's
output (static / moving + object ID, sweep index, or disagreement between two runs). Right:
frame controls, frame statistics and two timing charts (per-stage ms for this frame, e2e over
all frames with the current frame marked).

It only reads files, so any implementation that writes the same formats can be viewed and
compared side by side (Python V0, the C++/CUDA variants V1..V5, laptop or Jetson):

  outputs dir   <dir>/<seq>_<frame>/ with binio .bin files (format in benchmarks/binio.py)
                  points.bin       float32 [N, 3]   input points after crop + ground removal
                  mos_label.bin    int32   [N]      0 = static, 1 = moving
                  inst_label.bin   int32   [N]      0 = no object, 1..K = object ID
                  poses.bin        float32 [T,4,4]  estimated ego poses, sweep t -> anchor
                  time_indice.bin  int32   [N]      sweep index 0..T-1 (optional: enables the
                                                    "sweep" coloring and ego compensation)
  timing CSV    benchmark CSV, schema variant,device,precision,frame,pass,stage,ms,n_points,
                n_pillars,n_fg_points (frame '000/0005' matches outputs folder '000_0005');
                the median over passes is shown.

Examples:
    python benchmarks/viewer.py                                   # V0 outputs + all CSVs in results/
    python benchmarks/viewer.py --outputs benchmarks/results/V0_outputs pcacc_cpp/out/V1_outputs \\
        --csv benchmarks/results/V0_*.csv pcacc_cpp/out/V1_*.csv
    python benchmarks/viewer.py --screenshot shot.png --frame 10  # render one frame to PNG and exit

Keys (3D view focused): Left/Right = previous/next frame, Space = play/pause.
"""
import argparse, csv, glob, os, statistics, sys, time, traceback
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binio import read_bin  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ['preprocess', 'voxelize', 'h2d', 'pillar_enc', 'backbone', 'ego_motion', 'motion_seg', 'cluster']

# categorical palette (dark-surface steps), fixed order: one hue per stage / per run / per object
HUES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#0ca30c', '#9085e9', '#e66767']
SURFACE, INK, INK2, GRID = '#1a1a19', '#f0efec', '#c3c2b7', '#3a3a38'
STATIC_RGB = np.array([0.45, 0.45, 0.43])
MOVING_NO_ID_RGB = np.array([1.0, 1.0, 1.0])     # moving but dropped by clustering (< 15 points)
DIFF_RGB = np.array([0.90, 0.29, 0.28])


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])


HUES_RGB = np.stack([hex_rgb(h) for h in HUES])


# ----------------------------------------------------------------------------- data
class OutputsRun:
    """One outputs directory (one implementation's per-frame results)."""

    def __init__(self, path):
        self.path = path.rstrip('/')
        self.label = os.path.basename(self.path).removesuffix('_outputs') or self.path
        self.frames = sorted(d for d in os.listdir(self.path)
                             if os.path.isfile(os.path.join(self.path, d, 'points.bin')))
        self._cache = {}

    def load(self, frame):
        if frame in self._cache:
            return self._cache[frame]
        d = os.path.join(self.path, frame)
        if not os.path.isfile(os.path.join(d, 'points.bin')):
            return None
        out = {'points': read_bin(os.path.join(d, 'points.bin'))[:, :3]}
        for name in ('mos_label', 'inst_label', 'poses', 'time_indice'):
            p = os.path.join(d, name + '.bin')
            out[name] = read_bin(p) if os.path.isfile(p) else None
        for name in ('mos_label', 'inst_label', 'time_indice'):         # labels may be stored as float
            if out[name] is not None:
                out[name] = np.rint(out[name]).astype(np.int64).reshape(-1)
        if len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[frame] = out
        return out


class TimingRun:
    """One (variant, precision, device) group from the benchmark CSVs."""

    def __init__(self, key):
        self.variant, self.precision, self.device = key
        dev = self.device.replace('NVIDIA ', '').replace('GeForce ', '')
        self.label = f'{self.variant} {self.precision} | {dev}'
        self.short = f'{self.variant} {self.precision}'
        self.ms = defaultdict(lambda: defaultdict(list))    # frame -> stage -> [ms over passes]
        self.sizes = {}                                     # frame -> (n_points, n_pillars, n_fg)

    def stage_ms(self, frame, stage):
        v = self.ms.get(frame, {}).get(stage)
        return statistics.median(v) if v else None


def load_timing(paths):
    runs = {}
    for p in paths:
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                key = (r['variant'], r['precision'], r['device'])
                run = runs.setdefault(key, TimingRun(key))
                fid = r['frame'].replace('/', '_')
                run.ms[fid][r['stage']].append(float(r['ms']))
                run.sizes[fid] = (int(r['n_points']), int(r['n_pillars']), int(r['n_fg_points']))
    return [runs[k] for k in sorted(runs)]


# ----------------------------------------------------------------------------- coloring
def colorize(data, mode, other=None):
    n = len(data['points'])
    mos, inst, t = data['mos_label'], data['inst_label'], data['time_indice']
    col = np.tile(STATIC_RGB, (n, 1))
    if mode == 'segmentation' and mos is not None:
        col[mos == 1] = MOVING_NO_ID_RGB
        if inst is not None:
            m = inst > 0
            col[m] = HUES_RGB[(inst[m] - 1) % len(HUES_RGB)]
    elif mode == 'motion' and mos is not None:
        col[mos == 1] = HUES_RGB[1]
    elif mode == 'sweep' and t is not None:
        tmax = max(int(t.max()), 1)
        ramp = np.stack([hex_rgb('#cde2fb'), hex_rgb('#104281')])       # anchor = light
        a = (t / tmax)[:, None]
        col = ramp[0] * (1 - a) + ramp[1] * a
    elif mode == 'diff' and other is not None:
        if len(other['points']) == n and other['mos_label'] is not None and mos is not None:
            col[mos != other['mos_label']] = DIFF_RGB
    return col


def compensated(data):
    """Points of every sweep moved into the anchor frame with the estimated poses."""
    pts, poses, t = data['points'], data['poses'], data['time_indice']
    if poses is None or t is None:
        return pts
    R, tr = poses[t, :3, :3], poses[t, :3, 3]
    return np.einsum('nij,nj->ni', R, pts) + tr


# ----------------------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_to_rgb(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def stage_chart(runs, frame, width_px):
    """Horizontal stacked bars: one bar per run, one segment per stage, e2e at the end."""
    dpi = 100
    rows = [r for r in runs if frame in r.ms]
    h = 0.9 + 0.42 * max(len(rows), 1) + 0.55
    fig, ax = plt.subplots(figsize=(width_px / dpi, h), dpi=dpi, facecolor=SURFACE)
    _style(ax)
    if not rows:
        ax.text(0.5, 0.5, 'no timing for this frame', color=INK2, ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return fig_to_rgb(fig)
    for y, r in enumerate(rows):
        left = 0.0
        for i, s in enumerate(STAGES):
            v = r.stage_ms(frame, s)
            if not v:
                continue
            ax.barh(y, v, left=left, height=0.62, color=HUES[i], edgecolor=SURFACE, linewidth=1.5)
            left += v
        e2e = r.stage_ms(frame, 'e2e') or left
        ax.text(left, y, f'  {e2e:.1f} ms', va='center', color=INK, fontsize=8)
    ax.set_yticks(range(len(rows)), [r.short for r in rows], color=INK)
    ax.invert_yaxis()
    xmax = max((r.stage_ms(frame, 'e2e') or 0) for r in rows)
    ax.set_xlim(0, xmax * 1.22 if xmax else 1)
    ax.set_xlabel('ms (median over passes)', color=INK2, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in HUES]
    ax.legend(handles, STAGES, ncol=4, fontsize=7, frameon=False, labelcolor=INK2,
              loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(pad=0.4)
    return fig_to_rgb(fig)


def e2e_chart(runs, frames, frame, width_px):
    """e2e ms over all frames, one line per run, current frame marked."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, 2.3), dpi=dpi, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.grid(axis='x', visible=False)
    idx = frames.index(frame) if frame in frames else None
    for k, r in enumerate(runs):
        ys = [r.stage_ms(f, 'e2e') for f in frames]
        xs = [i for i, v in enumerate(ys) if v is not None]
        ax.plot(xs, [ys[i] for i in xs], color=HUES[k % len(HUES)], linewidth=1.2, label=r.short)
        if idx is not None and ys[idx] is not None:
            ax.plot([idx], [ys[idx]], 'o', ms=6, color=HUES[k % len(HUES)], mec=SURFACE, mew=1.5)
    if idx is not None:
        ax.axvline(idx, color=INK2, linewidth=0.8, linestyle=':')
    ax.set_xlim(0, max(len(frames) - 1, 1))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('frame', color=INK2, fontsize=8)
    ax.set_ylabel('e2e ms', color=INK2, fontsize=8)
    if len(runs) > 1:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc='upper left', ncol=min(len(runs), 3))
    fig.tight_layout(pad=0.4)
    return fig_to_rgb(fig)


# ----------------------------------------------------------------------------- window
class Viewer:
    MODES = [('Segmentation (static / object ID)', 'segmentation'), ('Motion (static / moving)', 'motion'),
             ('Sweep index (time)', 'sweep'), ('Disagreement vs other run', 'diff')]

    def __init__(self, outputs, timing, start=0, point_size=2.0):
        self.outputs, self.timing = outputs, timing
        fr = set()
        for o in outputs:
            fr.update(o.frames)
        for t in timing:
            fr.update(t.ms.keys())
        self.frames = sorted(fr)
        self.idx = min(max(start, 0), len(self.frames) - 1)
        self.run_i, self.other_i, self.mode = 0, min(1, len(outputs) - 1), 'segmentation'
        self.compensate, self.anchor_only = True, False
        self.playing, self.play_fps, self._last_tick = False, 5.0, 0.0
        self.point_size = point_size
        self._camera_set = False

        app = gui.Application.instance
        self.win = app.create_window('PCAccumulation benchmark viewer', 1700, 950)
        w, em = self.win, self.win.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.07, 0.07, 0.07, 1.0])
        self.scene.scene.show_axes(False)
        self.scene.set_on_key(self._on_key)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultUnlit'
        self.mat.point_size = point_size * w.scaling
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = 'unlitLine'
        self.line_mat.line_width = 2 * w.scaling

        p = self.panel = gui.ScrollableVert(0.4 * em, gui.Margins(em, em, em, em))

        # frame navigation
        p.add_child(self._section('Frame'))
        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.set_limits(0, len(self.frames) - 1)
        self.frame_slider.set_on_value_changed(lambda v: self.goto(int(v)))
        p.add_child(self.frame_slider)
        nav = gui.Horiz(0.3 * em)
        for text, fn in (('<', lambda: self.goto(self.idx - 1)), ('>', lambda: self.goto(self.idx + 1))):
            b = gui.Button(text)
            b.set_on_clicked(fn)
            nav.add_child(b)
        self.play_btn = gui.Button('Play')
        self.play_btn.set_on_clicked(self.toggle_play)
        nav.add_child(self.play_btn)
        nav.add_child(gui.Label(' fps'))
        fps = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        fps.set_limits(0.5, 30)
        fps.double_value = self.play_fps
        fps.set_on_value_changed(lambda v: setattr(self, 'play_fps', float(v)))
        nav.add_child(fps)
        nav.add_stretch()
        p.add_child(nav)

        # display options
        p.add_child(self._section('Display'))
        if outputs:
            p.add_child(gui.Label('Point cloud from run'))
            self.run_box = gui.Combobox()
            for o in outputs:
                self.run_box.add_item(o.label)
            self.run_box.set_on_selection_changed(lambda _t, i: self._set('run_i', i))
            p.add_child(self.run_box)
        p.add_child(gui.Label('Color by'))
        self.mode_box = gui.Combobox()
        for text, _ in self.MODES:
            self.mode_box.add_item(text)
        self.mode_box.set_on_selection_changed(lambda _t, i: self._set('mode', self.MODES[i][1]))
        p.add_child(self.mode_box)
        if len(outputs) > 1:
            p.add_child(gui.Label('Other run (for disagreement)'))
            self.other_box = gui.Combobox()
            for o in outputs:
                self.other_box.add_item(o.label)
            self.other_box.selected_index = self.other_i
            self.other_box.set_on_selection_changed(lambda _t, i: self._set('other_i', i))
            p.add_child(self.other_box)
        cb = gui.Checkbox('Ego-motion compensated (estimated poses)')
        cb.checked = self.compensate
        cb.set_on_checked(lambda c: self._set('compensate', c))
        p.add_child(cb)
        cb2 = gui.Checkbox('Anchor sweep only')
        cb2.set_on_checked(lambda c: self._set('anchor_only', c))
        p.add_child(cb2)
        ps = gui.Horiz(0.3 * em)
        ps.add_child(gui.Label('Point size'))
        sl = gui.Slider(gui.Slider.DOUBLE)
        sl.set_limits(1, 6)
        sl.double_value = point_size
        sl.set_on_value_changed(self._set_point_size)
        ps.add_child(sl)
        p.add_child(ps)
        views = gui.Horiz(0.3 * em)
        for text, kind in (("Bird's-eye", 'bev'), ('Perspective', 'persp')):
            b = gui.Button(text)
            b.set_on_clicked(lambda k=kind: self.set_view(k))
            views.add_child(b)
        views.add_stretch()
        p.add_child(views)

        # stats + charts
        p.add_child(self._section('This frame'))
        self.info = gui.Label('')
        p.add_child(self.info)
        self.legend = gui.Label('')
        p.add_child(self.legend)
        p.add_child(self._section('Latency per stage'))
        self.stage_img = gui.ImageWidget()
        p.add_child(self.stage_img)
        p.add_child(self._section('End-to-end latency over frames'))
        self.e2e_img = gui.ImageWidget()
        p.add_child(self.e2e_img)
        self.summary = gui.Label('')
        p.add_child(self.summary)
        self.summary.text = self._summary_text()

        w.add_child(self.scene)
        w.add_child(p)
        w.set_on_layout(self._on_layout)
        w.set_on_tick_event(self._on_tick)
        self.panel_px = int(34 * em)
        self.goto(self.idx)

    # --- helpers
    @staticmethod
    def _section(text):
        lbl = gui.Label(text.upper())
        lbl.text_color = gui.Color(0.6, 0.6, 0.58)
        return lbl

    def _set(self, attr, value):
        setattr(self, attr, value)
        self.refresh(charts=attr == 'run_i')

    def _set_point_size(self, v):
        self.mat.point_size = float(v) * self.win.scaling
        self.scene.scene.modify_geometry_material('pcd', self.mat)
        self.win.post_redraw()

    def _summary_text(self):
        lines = []
        for r in self.timing:
            e2e = [v for f in r.ms for v in r.ms[f].get('e2e', [])]
            if e2e:
                lines.append(f'{r.label}\n  e2e median {statistics.median(e2e):.1f} ms, '
                             f'{1000 / (sum(e2e) / len(e2e)):.2f} FPS, {len(r.ms)} frames')
        return '\n'.join(lines) or 'no timing CSV loaded'

    def _on_layout(self, ctx):
        r = self.win.content_rect
        pw = min(self.panel_px, r.width // 2)
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _on_key(self, e):
        if e.type != gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if e.key == gui.KeyName.RIGHT:
            self.goto(self.idx + 1)
        elif e.key == gui.KeyName.LEFT:
            self.goto(self.idx - 1)
        elif e.key == gui.KeyName.SPACE:
            self.toggle_play()
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    def _on_tick(self):
        if not self.playing or time.time() - self._last_tick < 1.0 / self.play_fps:
            return False
        self._last_tick = time.time()
        self.goto((self.idx + 1) % len(self.frames))
        return True

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.text = 'Pause' if self.playing else 'Play'

    def set_view(self, kind):
        c = np.zeros(3, dtype=np.float32)
        if kind == 'bev':
            self.scene.look_at(c, c + np.array([0, 0, 95], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
        else:
            self.scene.look_at(c, c + np.array([-38, -20, 30], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
        self.win.post_redraw()

    # --- main update
    def goto(self, i):
        self.idx = min(max(i, 0), len(self.frames) - 1)
        if int(self.frame_slider.int_value) != self.idx:
            self.frame_slider.int_value = self.idx
        self.refresh(charts=True)

    def refresh(self, charts=False):
        frame = self.frames[self.idx]
        sc = self.scene.scene
        for g in ('pcd', 'traj'):
            if sc.has_geometry(g):
                sc.remove_geometry(g)

        info = [f'{frame}   ({self.idx + 1} / {len(self.frames)})']
        data = self.outputs[self.run_i].load(frame) if self.outputs else None
        if data is not None:
            other = self.outputs[self.other_i].load(frame) if len(self.outputs) > 1 else None
            pts = compensated(data) if self.compensate else data['points']
            col = colorize(data, self.mode, other)
            keep = np.ones(len(pts), bool)
            if self.anchor_only and data['time_indice'] is not None:
                keep = data['time_indice'] == 0
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep].astype(np.float64)))
            pcd.colors = o3d.utility.Vector3dVector(np.power(col[keep], 2.2))   # sRGB -> linear
            sc.add_geometry('pcd', pcd, self.mat)
            if data['poses'] is not None and self.compensate:
                origins = data['poses'][:, :3, 3].astype(np.float64)
                ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(origins),
                                          o3d.utility.Vector2iVector([[k, k + 1] for k in range(len(origins) - 1)]))
                ls.paint_uniform_color([1.0, 1.0, 1.0])
                sc.add_geometry('traj', ls, self.line_mat)
            if not self._camera_set:
                bbox = o3d.geometry.AxisAlignedBoundingBox([-36, -36, -3], [36, 36, 6])
                self.scene.setup_camera(60, bbox, bbox.get_center())
                self.set_view('bev')
                self._camera_set = True
            mos, inst = data['mos_label'], data['inst_label']
            info.append(f'run {self.outputs[self.run_i].label}: {len(pts):,} points')
            if mos is not None:
                info.append(f'moving points {int((mos == 1).sum()):,} ({100 * (mos == 1).mean():.1f} %)')
            if inst is not None:
                info.append(f'objects (instances) {int(len(np.unique(inst[inst > 0])))}')
            if data['poses'] is not None:
                d = np.linalg.norm(data['poses'][-1, :3, 3])
                info.append(f'ego displacement anchor -> sweep {len(data["poses"]) - 1}: {d:.2f} m')
            if self.mode == 'diff' and other is not None:
                if len(other['points']) == len(pts) and other['mos_label'] is not None:
                    info.append(f'static/moving agreement with {self.outputs[self.other_i].label}: '
                                f'{100 * (mos == other["mos_label"]).mean():.3f} %')
                else:
                    info.append('other run has different input points: cannot compare')
            if self.mode == 'sweep' and data['time_indice'] is None:
                info.append('(no time_indice.bin in this run: sweep coloring unavailable)')
        else:
            info.append('no outputs for this frame')
        for r in self.timing:
            if frame in r.sizes:
                n, m, k = r.sizes[frame]
                info.append(f'pipeline sizes: N {n:,}  pillars {m:,}  fg points {k:,}')
                break
        self.info.text = '\n'.join(info)
        self.legend.text = {
            'segmentation': 'gray = static, white = moving without object, color = object ID\n'
                            'white line = estimated ego trajectory',
            'motion': 'gray = static, orange = moving',
            'sweep': 'light = anchor sweep (t = 0) ... dark = oldest sweep',
            'diff': 'red = static/moving label differs from the other run',
        }[self.mode]

        if charts:
            px = max(self.panel_px - 2 * int(self.win.theme.font_size), 200)
            self.stage_img.update_image(o3d.geometry.Image(stage_chart(self.timing, frame, px)))
            self.e2e_img.update_image(o3d.geometry.Image(e2e_chart(self.timing, self.frames, frame, px)))
        self.win.set_needs_layout()
        self.win.post_redraw()

    def screenshot(self, path):
        """Save 3D view + panel charts side by side, then quit."""
        def done(img3d):
            a = np.asarray(img3d)[:, :, :3]
            fw, fh = self.scene.frame.width, self.scene.frame.height     # undo stretch to window size
            a = a[(np.arange(fh) * a.shape[0] // fh)][:, (np.arange(fw) * a.shape[1] // fw)]
            frame = self.frames[self.idx]
            px = max(self.panel_px - 2 * int(self.win.theme.font_size), 200)
            charts = [stage_chart(self.timing, frame, px), e2e_chart(self.timing, self.frames, frame, px)]
            right = np.concatenate(charts, axis=0)
            h = max(a.shape[0], right.shape[0])
            pad = lambda x: np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0)), constant_values=26)
            plt.imsave(path, np.concatenate([pad(a), pad(right)], axis=1))
            print(f'saved {path}')
            gui.Application.instance.quit()
        self.scene.scene.scene.render_to_image(lambda im: gui.Application.instance.post_to_main_thread(
            self.win, lambda: done(im)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    res = os.path.join(REPO, 'benchmarks', 'results')
    ap.add_argument('--outputs', nargs='*', help='outputs dirs (default: results/*_outputs)')
    ap.add_argument('--csv', nargs='*', help='benchmark CSVs (default: results/*.csv)')
    ap.add_argument('--frame', type=int, default=0, help='start at this frame index')
    ap.add_argument('--point-size', type=float, default=2.0)
    ap.add_argument('--color', default='segmentation', choices=[m for _, m in Viewer.MODES])
    ap.add_argument('--screenshot', help='render the start frame to this PNG and exit')
    args = ap.parse_args()

    out_dirs = args.outputs if args.outputs is not None else sorted(glob.glob(os.path.join(res, '*_outputs')))
    csvs = args.csv if args.csv is not None else sorted(glob.glob(os.path.join(res, '*.csv')))
    csvs = [c for c in csvs if not os.path.basename(c).startswith('benchmark_results')]   # summary table
    outputs = [OutputsRun(d) for d in out_dirs if os.path.isdir(d)]
    timing = load_timing(csvs)
    print(f'outputs: {[o.label for o in outputs]}  timing runs: {[t.label for t in timing]}')

    if not outputs and not timing:
        raise SystemExit('nothing to show: no outputs dirs and no CSVs found')
    gui.Application.instance.initialize()
    try:
        v = Viewer(outputs, timing, start=args.frame, point_size=args.point_size)
        v.mode_box.selected_index = [m for _, m in Viewer.MODES].index(args.color)
        v._set('mode', args.color)
    except Exception:                     # the GUI app would otherwise keep the process alive
        traceback.print_exc()
        os._exit(1)
    if args.screenshot:
        state = {'n': 0}

        def tick():
            state['n'] += 1
            if state['n'] == 10:            # let the first frames render
                v.screenshot(args.screenshot)
            return False
        v.win.set_on_tick_event(tick)
    gui.Application.instance.run()


if __name__ == '__main__':
    main()
