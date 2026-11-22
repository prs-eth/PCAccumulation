"""Live benchmark window: input point cloud, output segmentation and statistics, frame by frame.

Used by the benchmark scripts (`bench_v0.py --view`). The window runs in a separate process
(Open3D needs its own main thread, and this keeps GUI work off the timed Python code). The
benchmark sends one message per frame *after* the frame's timed runs; while the window is still
drawing the previous frame, new frames are skipped, so the benchmark never waits for the GUI
(unless paused from the window).

Layout:
  INPUT   the sweeps as loaded from disk, colored by sweep; points removed by crop + ground
          removal are dim; the box is the crop region
  OUTPUT  pipeline result, ego-motion compensated: static / moving + object ID (or motion only,
          or sweep); white line = estimated ego trajectory
  panel   benchmark progress, Pause / Step, input + output statistics, per-stage latency of this
          frame, e2e latency over the frames, running median / FPS

Charts, colors and the per-frame statistics reuse benchmarks/viewer.py (unchanged), so the live
window and the offline viewer look the same.

Only numpy is imported at module level: the benchmark process never loads Open3D.
"""
import multiprocessing as mp
import queue
import time

import numpy as np


# ----------------------------------------------------------------------------- benchmark side
class LiveViewClient:
    """Lives in the benchmark process; starts the window process and feeds it frames."""

    def __init__(self, key, frame_ids, pause_s=0.0):
        """key = (variant, precision, device), as in the CSV."""
        ctx = mp.get_context('spawn')
        self.q = ctx.Queue(maxsize=4)     # full frames (point clouds), at most one in flight
        self.tq = ctx.Queue()             # small timing record of every frame
        self.ready = ctx.Event()          # window finished drawing the previous frame
        self.paused = ctx.Event()         # set from the window: benchmark waits between frames
        self.step = ctx.Event()           # set from the window: run one frame while paused
        self.pause_s = pause_s
        self.proc = ctx.Process(target=_window_main, daemon=True,
                                args=(tuple(key), list(frame_ids), self.q, self.tq, self.ready, self.paused,
                                      self.step))
        self.proc.start()

    @property
    def alive(self):
        return self.proc.is_alive()

    def wait_if_paused(self):
        """Call between frames (untimed). Blocks while the window has the benchmark paused."""
        while self.alive and self.paused.is_set() and not self.step.is_set():
            time.sleep(0.03)
        self.step.clear()

    def send(self, msg):
        """Non-blocking: the frame is dropped if the window is still busy with the previous one."""
        if not self.alive:
            return False
        self.tq.put({'fid': msg['fid'], 'stages': msg['stages'], 'sizes': msg['sizes']})
        if self.ready.is_set():
            self.ready.clear()
            try:
                self.q.put_nowait(msg)
            except queue.Full:
                pass
            sent = True
        else:
            sent = False
        if self.pause_s > 0 or self.paused.is_set():
            time.sleep(self.pause_s)
        return sent

    def finish(self, text):
        if self.alive:
            try:
                self.q.put({'done': True, 'text': text}, timeout=1.0)
            except queue.Full:
                pass
            print('benchmark finished: close the live view window to exit')
            self.proc.join()


def frame_message(*, fid, pass_idx, passes, frame_idx, n_frames, stages, sizes, raw_points, raw_time,
                  keep, crop, points, time_indice, mos_label, inst_label, poses):
    """Plain numpy/float message for one frame (built outside the timed region)."""
    return {
        'fid': fid, 'pass': pass_idx, 'passes': passes, 'i': frame_idx, 'n': n_frames,
        'stages': {k: float(v) for k, v in stages.items()}, 'sizes': tuple(int(s) for s in sizes),
        'raw_points': np.ascontiguousarray(raw_points[:, :3], dtype=np.float32),
        'raw_time': np.asarray(raw_time, dtype=np.int8).reshape(-1),
        'keep': np.asarray(keep, dtype=bool).reshape(-1),
        'crop': tuple(float(c) for c in crop),           # (xy, z_min, z_max)
        'points': np.ascontiguousarray(points[:, :3], dtype=np.float32),
        'time_indice': np.asarray(time_indice, dtype=np.int64).reshape(-1),
        'mos_label': np.asarray(mos_label, dtype=np.int64).reshape(-1),
        'inst_label': np.asarray(inst_label, dtype=np.int64).reshape(-1),
        'poses': np.asarray(poses, dtype=np.float32),
    }


# ----------------------------------------------------------------------------- window side
def _window_main(key, frame_ids, q, tq, ready, paused, step):
    import open3d.visualization.gui as gui
    gui.Application.instance.initialize()
    _LiveWindow(key, frame_ids, q, tq, ready, paused, step)
    ready.set()
    gui.Application.instance.run()


class _LiveWindow:
    RESIZE_SETTLE_S = 0.4        # no redraws until the window size has been stable this long
    MIN_DRAW_INTERVAL_S = 0.5    # at most 2 frame redraws per second: the GUI stays responsive
    OUT_MODES = [('Segmentation (static / object ID)', 'segmentation'),
                 ('Motion (static / moving)', 'motion'), ('Sweep index (time)', 'sweep')]

    def __init__(self, key, frame_ids, q, tq, ready, paused, step):
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        import viewer as V                                  # benchmarks/viewer.py, used read-only
        self.o3d, self.gui, self.V = o3d, gui, V
        self.q, self.tq, self.ready, self.paused, self.step = q, tq, ready, paused, step
        self.frame_ids = frame_ids
        self.run = V.TimingRun(key)
        title = self.run.label
        self.msg, self.out_mode, self.show_removed, self.done_text = None, 'segmentation', True, None
        self._active_view, self._synced_mat = None, None          # camera link state

        app = gui.Application.instance
        self.win = w = app.create_window(f'Live benchmark | {title}', 1800, 950)
        em = w.theme.font_size                      # px; grows with screen DPI (32 on a 4K laptop)
        w.size = gui.Size(int(70 * em), int(44 * em))   # resized again to the panel's content on first layout

        self.views = []
        for _ in range(2):
            sw = gui.SceneWidget()
            sw.scene = rendering.Open3DScene(w.renderer)
            sw.scene.set_background([0.07, 0.07, 0.07, 1.0])
            sw.scene.show_axes(False)
            sw.set_on_mouse(lambda e, k=len(self.views): self._on_mouse(k, e))
            w.add_child(sw)
            self.views.append(sw)
        self.headers = [gui.Label('INPUT  sweeps as loaded\ndim = removed by crop + ground removal, box = crop'),
                        gui.Label('OUTPUT  pipeline result\nego-motion compensated')]
        for h in self.headers:
            h.background_color = gui.Color(0.07, 0.07, 0.07, 0.8)
            w.add_child(h)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultUnlit'
        self.mat.point_size = 2.0 * w.scaling
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = 'unlitLine'
        self.line_mat.line_width = 2 * w.scaling

        p = self.panel = gui.ScrollableVert(0.4 * em, gui.Margins(em, em, em, em))
        sec = V.Viewer._section

        p.add_child(sec('Benchmark'))
        self.status = gui.Label(f'{title}\nwaiting for the first frame (warm-up) ...')
        p.add_child(self.status)
        row = gui.Horiz(0.3 * em)
        self.pause_btn = gui.Button('Pause')
        self.pause_btn.set_on_clicked(self._toggle_pause)
        row.add_child(self.pause_btn)
        b = gui.Button('Step')
        b.set_on_clicked(lambda: self.step.set())
        row.add_child(b)
        row.add_stretch()
        p.add_child(row)

        p.add_child(sec('Display'))
        p.add_child(gui.Label('Output colored by'))
        box = gui.Combobox()
        for text, _ in self.OUT_MODES:
            box.add_item(text)
        box.set_on_selection_changed(lambda _t, i: self._set_mode(self.OUT_MODES[i][1]))
        p.add_child(box)
        cb = gui.Checkbox('Input: show removed points')
        cb.checked = True
        cb.set_on_checked(self._set_removed)
        p.add_child(cb)
        self.link = gui.Checkbox('Link the two cameras')
        self.link.checked = True
        self.link.set_on_checked(lambda c: self._sync_cameras(force=c))
        p.add_child(self.link)
        views = gui.Horiz(0.3 * em)
        for text, kind in (("Bird's-eye", 'bev'), ('Perspective', 'persp')):
            bt = gui.Button(text)
            bt.set_on_clicked(lambda k=kind: self.set_view(k))
            views.add_child(bt)
        views.add_stretch()
        p.add_child(views)

        p.add_child(sec('Input'))
        self.in_info = gui.Label('')
        p.add_child(self.in_info)
        p.add_child(sec('Output'))
        self.out_info = gui.Label('')
        p.add_child(self.out_info)
        self.legend = gui.Label('')
        p.add_child(self.legend)
        p.add_child(sec('Latency'))
        self.summary = gui.Label('waiting for the first frame')
        p.add_child(self.summary)
        w.add_child(p)

        # latency charts: a wide strip under the two 3D views
        self.stage_img, self.e2e_img = gui.ImageWidget(), gui.ImageWidget()
        w.add_child(self.stage_img)
        w.add_child(self.e2e_img)

        self.em = em
        self.panel_px = int(26 * em)
        self.min_w = self.panel_px + int(36 * em)                  # panel + views/charts wide enough for chart text
        self.min_h = int(24 * em)                                  # refined from the panel's content
        self.panel_h = None
        self._size = None                                          # content size of the last layout
        self._resized_at = 0.0                                     # time of the last size change
        self._chart_size = None                                    # size the charts were drawn at
        self._last_draw = 0.0
        self._pending = None                                       # newest frame not drawn yet
        w.set_on_layout(self._on_layout)
        w.set_on_tick_event(self._on_tick)
        w.set_on_close(self._on_close)
        self._legend()

    # --- callbacks
    def _on_layout(self, ctx):
        gui, r = self.gui, self.win.content_rect
        if self._size != (r.width, r.height):
            self._size = (r.width, r.height)
            self._resized_at = time.time()
        # minimum height = everything in the side panel visible without scrolling
        c = gui.Widget.Constraints()
        c.width = self.panel_px
        self.panel_h = self.panel.calc_preferred_size(ctx, c).height
        self.min_h = max(self.panel_h, int(24 * self.em))

        # lay out at least at the minimum size (a smaller window clips instead of squashing)
        W, H = max(r.width, self.min_w), max(r.height, self.min_h)
        pw = self.panel_px
        left = W - pw
        strip = min(max(int(0.36 * H), int(17 * self.em)), int(26 * self.em))
        vh, vw = H - strip, left // 2
        for k, (sw, h) in enumerate(zip(self.views, self.headers)):
            sw.frame = gui.Rect(r.x + k * vw, r.y, vw - (2 if k == 0 else 0), vh)
            pref = h.calc_preferred_size(ctx, gui.Widget.Constraints())
            h.frame = gui.Rect(r.x + k * vw + 6, r.y + 6, min(pref.width, vw - 12), pref.height)
        sw_ = int(left * 0.55)                                     # stage chart: names + values
        self.stage_img.frame = gui.Rect(r.x, r.y + vh, sw_, strip)
        self.e2e_img.frame = gui.Rect(r.x + sw_, r.y + vh, left - sw_, strip)
        self.panel.frame = gui.Rect(r.x + left, r.y, pw, H)

    def _on_close(self):
        self.paused.clear()                 # never leave the benchmark blocked
        return True

    def _toggle_pause(self):
        if self.paused.is_set():
            self.paused.clear()
            self.pause_btn.text = 'Pause'
        else:
            self.paused.set()
            self.pause_btn.text = 'Resume'
        self._status()

    def _set_mode(self, m):
        self.out_mode = m
        self._draw(charts=False)

    def _set_removed(self, c):
        self.show_removed = c
        self._draw(charts=False)

    def _on_tick(self):
        redraw = self._sync_cameras()
        while True:                                     # timing of every frame, drawn or not
            try:
                t = self.tq.get_nowait()
            except queue.Empty:
                break
            for s, ms in t['stages'].items():
                self.run.ms[t['fid']][s].append(ms)
            self.run.sizes[t['fid']] = t['sizes']
        try:
            msg = self.q.get_nowait()
            if msg.get('done'):
                self.done_text = msg['text']
                self._status()
                redraw = True
            else:
                self._pending = msg
        except queue.Empty:
            pass

        now = time.time()
        if now - self._resized_at < self.RESIZE_SETTLE_S:
            return redraw                       # user is resizing: no heavy work, stay responsive
        r = self.win.content_rect
        if r.width < self.min_w or r.height < self.min_h:   # resize finished below the minimum: grow back
            sz = self.win.size
            self.win.size = self.gui.Size(sz.width + max(0, self.min_w - r.width),
                                          sz.height + max(0, self.min_h - r.height))
            self._resized_at = now
            return True
        if self._pending is not None and now - self._last_draw >= self.MIN_DRAW_INTERVAL_S:
            self.msg, self._pending = self._pending, None
            self._draw(charts=True)
            self._last_draw = time.time()
            self.ready.set()                    # benchmark may send the next full frame
            return True
        if self.msg is not None and self._chart_size != self._chart_target():
            self._draw_charts()                 # window size changed: redraw charts at the new size
            return True
        return redraw

    # --- cameras
    def set_view(self, kind):
        c = np.zeros(3, dtype=np.float32)
        if kind == 'bev':
            eye, up = np.array([0, 0, 95], np.float32), np.array([0, 1, 0], np.float32)
        else:
            eye, up = np.array([-38, -20, 30], np.float32), np.array([0, 0, 1], np.float32)
        for sw in self.views:
            sw.look_at(c, eye, up)                      # also resets the orbit center to the origin
        self._synced_mat = None
        self.win.post_redraw()

    def _on_mouse(self, k, e):
        """Remember which view the user is driving; the default orbit / pan / zoom still runs."""
        self._active_view = k
        return self.gui.Widget.EventCallbackResult.IGNORED

    def _sync_cameras(self, force=False):
        """One-way copy: the view under the mouse drives the other one (no feedback loop)."""
        if not self.link.checked:
            return False
        src = self._active_view if self._active_view is not None else 0
        a, b = self.views[src], self.views[1 - src]
        mat = np.asarray(a.scene.camera.get_view_matrix())
        if not force and self._synced_mat is not None and np.array_equal(mat, self._synced_mat):
            return False
        b.scene.camera.copy_from(a.scene.camera)        # exact view + projection, nothing rebuilt
        b.center_of_rotation = a.center_of_rotation     # orbiting the other view uses the same pivot
        b.force_redraw()
        self._synced_mat = mat.copy()
        return True

    # --- drawing
    def _status(self):
        m, lines = self.msg, [f'{self.run.label}']
        if m is not None:
            lines.append(f'pass {m["pass"] + 1}/{m["passes"]}   frame {m["i"] + 1}/{m["n"]}   ({m["fid"]})')
        if self.done_text:
            lines.append('FINISHED\n' + self.done_text)
        elif self.paused.is_set():
            lines.append('PAUSED (Step runs one frame)')
        else:
            lines.append('running (frames are skipped while the window draws)')
        self.status.text = '\n'.join(lines)

    def _legend(self):
        self.legend.text = {
            'segmentation': 'gray = static, white = moving without object, color = object ID\n'
                            'white line = estimated ego trajectory',
            'motion': 'gray = static, orange = moving',
            'sweep': 'light = anchor sweep (t = 0) ... dark = oldest sweep',
        }[self.out_mode]

    def _draw(self, charts):
        m = self.msg
        if m is None:
            return
        o3d, V = self.o3d, self.V
        ramp = np.stack([V.hex_rgb('#cde2fb'), V.hex_rgb('#104281')])

        # input view
        sc = self.views[0].scene
        for g in ('raw', 'crop'):
            if sc.has_geometry(g):
                sc.remove_geometry(g)
        keep, rt = m['keep'], m['raw_time'].astype(np.float64)
        a = (rt / max(rt.max(), 1))[:, None]
        col = ramp[0] * (1 - a) + ramp[1] * a
        col[~keep] = 0.16
        sel = np.ones(len(keep), bool) if self.show_removed else keep
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(m['raw_points'][sel].astype(np.float64)))
        pc.colors = o3d.utility.Vector3dVector(np.power(col[sel], 2.2))
        sc.add_geometry('raw', pc, self.mat)
        xy, z0, z1 = m['crop']
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox([-xy, -xy, z0], [xy, xy, z1]))
        box.paint_uniform_color([0.8, 0.8, 0.5])
        sc.add_geometry('crop', box, self.line_mat)

        # output view
        sc = self.views[1].scene
        for g in ('pcd', 'traj'):
            if sc.has_geometry(g):
                sc.remove_geometry(g)
        data = {k: m[k] for k in ('points', 'mos_label', 'inst_label', 'poses', 'time_indice')}
        pts = V.compensated(data)
        col = V.colorize(data, self.out_mode)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        pc.colors = o3d.utility.Vector3dVector(np.power(col, 2.2))
        sc.add_geometry('pcd', pc, self.mat)
        org = m['poses'][:, :3, 3].astype(np.float64)
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(org),
                                  o3d.utility.Vector2iVector([[k, k + 1] for k in range(len(org) - 1)]))
        ls.paint_uniform_color([1.0, 1.0, 1.0])
        sc.add_geometry('traj', ls, self.line_mat)

        if not getattr(self, '_camera_set', False):
            bbox = o3d.geometry.AxisAlignedBoundingBox([-36, -36, -3], [36, 36, 6])
            for sw in self.views:
                sw.setup_camera(60, bbox, bbox.get_center())
            self.set_view('bev')
            self._camera_set = True

        # statistics
        n_raw, n_keep = len(keep), int(keep.sum())
        n, pillars, fg = m['sizes']
        sweeps = np.bincount(m['raw_time'].astype(np.int64), minlength=1)
        self.in_info.text = (f'raw points {n_raw:,} in {len(sweeps)} sweeps\n'
                             f'  per sweep: {", ".join(f"{s:,}" for s in sweeps)}\n'
                             f'after crop + ground removal: {n_keep:,} ({100 * n_keep / max(n_raw, 1):.1f} %)\n'
                             f'4D pillars {pillars:,}')
        mos, inst = m['mos_label'], m['inst_label']
        d = np.linalg.norm(m['poses'][-1, :3, 3])
        self.out_info.text = (f'foreground points sent to STPN {fg:,}\n'
                              f'moving points {int((mos == 1).sum()):,} ({100 * (mos == 1).mean():.1f} %)\n'
                              f'objects (instances) {len(np.unique(inst[inst > 0]))}\n'
                              f'ego displacement anchor -> sweep {len(m["poses"]) - 1}: {d:.2f} m')
        self._legend()
        self._status()
        if charts:
            self._draw_charts()
            e2e = [v for f in self.run.ms.values() for v in f.get('e2e', [])]
            if e2e:
                self.summary.text = (f'this frame e2e {m["stages"].get("e2e", 0):.1f} ms\n'
                                     f'all {len(e2e)} frames so far:\n'
                                     f'  median {np.median(e2e):.1f} ms, p90 {np.percentile(e2e, 90):.1f} ms\n'
                                     f'  {1000 / np.mean(e2e):.2f} FPS')
        self.win.set_needs_layout()             # label heights changed (does not count as a resize)
        self.win.post_redraw()

    def _chart_target(self):
        f = self.stage_img.frame, self.e2e_img.frame
        return (f[0].width, f[0].height, f[1].width, f[1].height)

    def _draw_charts(self):
        m = self.msg
        if m is None:
            return
        tw, th, ew, eh = self._chart_target()
        if min(tw, th, ew, eh) < 50:
            return
        dpi = 100 * self.win.scaling * max(1.0, self.em / 16)   # chart text follows the UI font size
        o3d = self.o3d
        self.stage_img.update_image(o3d.geometry.Image(stage_bars(self.V, self.run, m['fid'], tw, th, dpi)))
        self.e2e_img.update_image(o3d.geometry.Image(e2e_line(self.V, self.run, self.frame_ids, m['fid'], ew, eh, dpi)))
        self._chart_size = (tw, th, ew, eh)


# ----------------------------------------------------------------------------- charts (large, readable)
def _fig(V, w_px, h_px, dpi):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor=V.SURFACE)
    ax = fig.add_subplot(111)
    V._style(ax)
    return fig, ax


def _to_rgb(V, fig, w_px, h_px):
    img = V.fig_to_rgb(fig)
    return np.ascontiguousarray(img[:h_px, :w_px])


def stage_bars(V, run, fid, w_px, h_px, dpi):
    """One bar per stage (named on the axis), this frame's ms + share of e2e; tick = run median."""
    fig, ax = _fig(V, w_px, h_px, dpi)
    vals = [run.stage_ms(fid, s) or 0.0 for s in V.STAGES]
    e2e = run.stage_ms(fid, 'e2e') or sum(vals)
    y = np.arange(len(V.STAGES))
    ax.barh(y, vals, height=0.66, color=V.HUES[:len(vals)])
    med = [np.median([v for f in run.ms.values() for v in f.get(s, [])] or [0]) for s in V.STAGES]
    ax.scatter(med, y, marker='|', s=260, color=V.INK, linewidths=1.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(max(v, med[i]), i, f' {v:.1f} ({100 * v / e2e:.0f}%)' if e2e else '', va='center',
                color=V.INK, fontsize=9)
    ax.set_yticks(y, V.STAGES, fontsize=10, color=V.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(vals), max(med), 1e-3) * 1.6)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_title(f'Per stage, e2e {e2e:.1f} ms', loc='left', color=V.INK, fontsize=10)
    ax.set_xlabel('ms (% of e2e)   | = run median', color=V.INK2, fontsize=9)
    fig.tight_layout(pad=0.5)
    return _to_rgb(V, fig, w_px, h_px)


def e2e_line(V, run, frame_ids, fid, w_px, h_px, dpi):
    """e2e ms per frame (benchmark order), current frame marked, run median dashed."""
    fig, ax = _fig(V, w_px, h_px, dpi)
    ax.grid(axis='y', color=V.GRID, linewidth=0.6)
    ax.grid(axis='x', visible=False)
    ys = [run.stage_ms(f, 'e2e') for f in frame_ids]
    xs = [i for i, v in enumerate(ys) if v is not None]
    ax.plot(xs, [ys[i] for i in xs], color=V.HUES[0], linewidth=1.5)
    allv = [v for f in run.ms.values() for v in f.get('e2e', [])]
    if allv:
        med = float(np.median(allv))
        ax.axhline(med, color=V.INK2, linewidth=1.0, linestyle='--')
        ax.text(0, med, f' median {med:.1f} ms', color=V.INK2, fontsize=9, va='bottom')
    if fid in frame_ids:
        i = frame_ids.index(fid)
        if ys[i] is not None:
            ax.plot([i], [ys[i]], 'o', ms=8, color=V.HUES[0], mec=V.SURFACE, mew=2)
            ax.axvline(i, color=V.INK2, linewidth=0.8, linestyle=':')
    ax.set_xlim(0, max(len(frame_ids) - 1, 1))
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9)
    ax.set_xlabel('frame', color=V.INK2, fontsize=9)
    ax.set_ylabel('ms', color=V.INK2, fontsize=9)
    ax.set_title('End-to-end per frame', loc='left', color=V.INK, fontsize=10)
    fig.tight_layout(pad=0.5)
    return _to_rgb(V, fig, w_px, h_px)
