"""STRAY — one-take comic chase, built and rendered in Blender.

Run:
  blender -b -P stray/film.py -- --preview
  blender -b -P stray/film.py -- --render
"""

from __future__ import annotations

import math
import os
import random
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

ROOT = "/agent/repos/Truman-"
sys.path.insert(0, os.path.join(ROOT, "stray"))
import timing as T  # noqa: E402

FRAME_DIR = os.path.join(ROOT, "build", "frames")
SHOT_DIR = os.path.join(ROOT, "build", "shots")
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
BILLBOARDS = [
    "billboard_face.jpg",
    "billboard_dragon.jpg",
    "billboard_city.jpg",
    "billboard_koi.jpg",
    "billboard_car.jpg",
    "billboard_eye.jpg",
    "billboard_ramen.jpg",
    "billboard_figure.jpg",
]

# Palette
INK = (0.01, 0.012, 0.02, 1)
MAGENTA = (1.0, 0.16, 0.42, 1)
CYAN = (0.15, 0.95, 0.95, 1)
YELLOW = (1.0, 0.88, 0.18, 1)
PURPLE = (0.55, 0.22, 1.0, 1)
INDIGO = (0.03, 0.04, 0.08, 1)


def parse_args():
    argv = sys.argv
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    mode = "preview"
    if "--render" in extra:
        mode = "render"
    if "--preview" in extra:
        mode = "preview"
    return mode


# ---------------------------------------------------------------------------
# Low-level mesh helpers
# ---------------------------------------------------------------------------

def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def new_mesh(name, verts, faces, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices):
                uv.data[li].uv = uvs[vi]
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def _torus_mesh(R, r, maj, minor, axis="Z"):
    verts = []
    for i in range(maj):
        a = 2 * math.pi * i / maj
        ca, sa = math.cos(a), math.sin(a)
        for j in range(minor):
            b = 2 * math.pi * j / minor
            cb, sb = math.cos(b), math.sin(b)
            ring = R + r * cb
            if axis == "X":
                verts.append((r * sb, ring * ca, ring * sa))
            elif axis == "Y":
                verts.append((ring * ca, r * sb, ring * sa))
            else:
                verts.append((ring * ca, ring * sa, r * sb))
    faces = []
    for i in range(maj):
        for j in range(minor):
            a = i * minor + j
            b = i * minor + (j + 1) % minor
            c = ((i + 1) % maj) * minor + (j + 1) % minor
            d = ((i + 1) % maj) * minor + j
            faces.append((a, b, c, d))
    return new_mesh("torus", verts, faces)


def primitive(kind, **kw):
    bm = bmesh.new()
    if kind == "cube":
        bmesh.ops.create_cube(bm, size=1.0)
    elif kind == "cyl":
        bmesh.ops.create_cone(
            bm, cap_ends=True, segments=kw.get("seg", 20),
            radius1=kw.get("r", 1), radius2=kw.get("r", 1), depth=kw.get("d", 1),
        )
    elif kind == "cone":
        bmesh.ops.create_cone(
            bm, cap_ends=True, segments=kw.get("seg", 16),
            radius1=kw.get("r", 1), radius2=0.0, depth=kw.get("d", 1),
        )
    elif kind == "uv":
        bmesh.ops.create_uvsphere(
            bm, u_segments=kw.get("u", 12), v_segments=kw.get("v", 8), radius=1.0
        )
    elif kind == "torus":
        bm.free()
        return _torus_mesh(
            kw.get("R", 1.0), kw.get("r", 0.08),
            kw.get("maj", 24), kw.get("min", 8), kw.get("axis", "Z"),
        )
    elif kind == "plane":
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    me = bpy.data.meshes.new(kind)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return me


def ob_from_mesh(name, me, mat=None):
    ob = bpy.data.objects.new(name, me)
    link(ob)
    if mat is not None:
        if len(me.materials) == 0:
            me.materials.append(mat)
        else:
            # Shared meshes keep their own slot; override per object when needed.
            ob.material_slots[0].link = "OBJECT"
            ob.material_slots[0].material = mat
    return ob


def shade_flat(me):
    for p in me.polygons:
        p.use_smooth = False


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _out(nt, color_socket, strength=1.0):
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs[1].default_value = strength
    nt.links.new(color_socket, emit.inputs[0])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(emit.outputs[0], out.inputs[0])
    return emit


def fresh(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    mat.use_backface_culling = False
    return mat


def emission(name, color, strength=1.0):
    mat = fresh(name)
    rgb = mat.node_tree.nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = color
    _out(mat.node_tree, rgb.outputs[0], strength)
    return mat


def toon(name, base, shadow, light, rim_a, rim_b, strength=1.0, halftone=False, fringe=False):
    mat = fresh(name)
    nt = mat.node_tree
    n, l = nt.nodes, nt.links
    geo = n.new("ShaderNodeNewGeometry")
    dot = n.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.40, 0.18, 0.90)
    l.new(geo.outputs["Normal"], dot.inputs[0])
    mp = n.new("ShaderNodeMapRange")
    mp.inputs[1].default_value = -0.2
    mp.inputs[2].default_value = 0.85
    mp.clamp = True
    l.new(dot.outputs["Value"], mp.inputs[0])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[0].color = shadow
    ramp.color_ramp.elements[1].position = 0.62
    ramp.color_ramp.elements[1].color = base
    el = ramp.color_ramp.elements.new(0.80)
    el.color = light
    l.new(mp.outputs["Result"], ramp.inputs["Fac"])

    lw = n.new("ShaderNodeLayerWeight")
    sep = n.new("ShaderNodeSeparateXYZ")
    l.new(geo.outputs["Normal"], sep.inputs[0])
    nx = n.new("ShaderNodeMapRange")
    nx.inputs[1].default_value = -1.0
    nx.inputs[2].default_value = 1.0
    l.new(sep.outputs["X"], nx.inputs[0])
    rim = n.new("ShaderNodeMixRGB")
    rim.inputs[1].default_value = rim_a
    rim.inputs[2].default_value = rim_b
    l.new(nx.outputs["Result"], rim.inputs["Color1"] if False else rim.inputs[0])

    inv = n.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    l.new(lw.outputs["Facing"], inv.inputs[1])
    edge = n.new("ShaderNodeMath")
    edge.operation = "POWER"
    edge.inputs[1].default_value = 3.2
    l.new(inv.outputs[0], edge.inputs[0])

    mixed = n.new("ShaderNodeMixRGB")
    l.new(edge.outputs[0], mixed.inputs[0])
    l.new(ramp.outputs["Color"], mixed.inputs[1])
    l.new(rim.outputs["Color"], mixed.inputs[2])
    color = mixed.outputs["Color"]

    if halftone:
        tc = n.new("ShaderNodeTexCoord")
        sp = n.new("ShaderNodeSeparateXYZ")
        l.new(tc.outputs["Window"], sp.inputs[0])
        fx = n.new("ShaderNodeMath")
        fx.operation = "MULTIPLY"
        fx.inputs[1].default_value = 110.0
        fy = n.new("ShaderNodeMath")
        fy.operation = "MULTIPLY"
        fy.inputs[1].default_value = 110.0
        l.new(sp.outputs["X"], fx.inputs[0])
        l.new(sp.outputs["Y"], fy.inputs[0])
        sx = n.new("ShaderNodeMath")
        sx.operation = "SINE"
        sy = n.new("ShaderNodeMath")
        sy.operation = "SINE"
        l.new(fx.outputs[0], sx.inputs[0])
        l.new(fy.outputs[0], sy.inputs[0])
        prod = n.new("ShaderNodeMath")
        prod.operation = "MULTIPLY"
        l.new(sx.outputs[0], prod.inputs[0])
        l.new(sy.outputs[0], prod.inputs[1])
        shade = n.new("ShaderNodeMath")
        shade.operation = "SUBTRACT"
        shade.inputs[0].default_value = 1.0
        l.new(mp.outputs["Result"], shade.inputs[1])
        mask = n.new("ShaderNodeMath")
        mask.operation = "MULTIPLY"
        l.new(prod.outputs[0], mask.inputs[0])
        l.new(shade.outputs[0], mask.inputs[1])
        dots = n.new("ShaderNodeMixRGB")
        dots.inputs[2].default_value = shadow
        l.new(mask.outputs[0], dots.inputs[0])
        l.new(color, dots.inputs[1])
        color = dots.outputs["Color"]

    if fringe:
        # Distant plates carry a cyan/magenta misregister on the rim only.
        tint = n.new("ShaderNodeMixRGB")
        tint.blend_type = "ADD"
        tint.inputs[0].default_value = 0.35
        l.new(color, tint.inputs[1])
        l.new(rim.outputs["Color"], tint.inputs[2])
        color = tint.outputs["Color"]

    _out(nt, color, strength)
    return mat


def glass_mat():
    mat = fresh("Glass")
    nt = mat.node_tree
    n, l = nt.nodes, nt.links
    tr = n.new("ShaderNodeBsdfTransparent")
    tr.inputs[0].default_value = (0.75, 0.9, 1.0, 1)
    em = n.new("ShaderNodeEmission")
    em.inputs[0].default_value = (0.35, 0.85, 0.95, 1)
    em.inputs[1].default_value = 0.18
    lw = n.new("ShaderNodeLayerWeight")
    mix = n.new("ShaderNodeMixShader")
    l.new(lw.outputs["Fresnel"], mix.inputs[0])
    l.new(tr.outputs[0], mix.inputs[1])
    l.new(em.outputs[0], mix.inputs[2])
    out = n.new("ShaderNodeOutputMaterial")
    l.new(mix.outputs[0], out.inputs[0])
    mat.blend_method = "BLEND"
    mat.surface_render_method = "BLENDED"
    mat.shadow_method = "NONE"
    mat.use_transparency_overlap = True
    return mat


def road_mat():
    mat = fresh("Road")
    nt = mat.node_tree
    n, l = nt.nodes, nt.links
    tc = n.new("ShaderNodeTexCoord")
    sp = n.new("ShaderNodeSeparateXYZ")
    l.new(tc.outputs["UV"], sp.inputs[0])
    mu = n.new("ShaderNodeMath")
    mu.operation = "MULTIPLY"
    mu.inputs[1].default_value = 52.0
    l.new(sp.outputs["X"], mu.inputs[0])
    sn = n.new("ShaderNodeMath")
    sn.operation = "SINE"
    l.new(mu.outputs[0], sn.inputs[0])
    ab = n.new("ShaderNodeMath")
    ab.operation = "ABSOLUTE"
    l.new(sn.outputs[0], ab.inputs[0])
    pw = n.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 18.0
    l.new(ab.outputs[0], pw.inputs[0])
    # Color bands along the road.
    mv = n.new("ShaderNodeMath")
    mv.operation = "MULTIPLY"
    mv.inputs[1].default_value = 0.12
    l.new(sp.outputs["Y"], mv.inputs[0])
    fr = n.new("ShaderNodeMath")
    fr.operation = "FRACT"
    l.new(mv.outputs[0], fr.inputs[0])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = MAGENTA
    ramp.color_ramp.elements[1].position = 0.34
    ramp.color_ramp.elements[1].color = CYAN
    e2 = ramp.color_ramp.elements.new(0.67)
    e2.color = YELLOW
    l.new(fr.outputs[0], ramp.inputs["Fac"])
    base = n.new("ShaderNodeRGB")
    base.outputs[0].default_value = (0.008, 0.010, 0.020, 1)
    mix = n.new("ShaderNodeMixRGB")
    l.new(pw.outputs[0], mix.inputs[0])
    l.new(base.outputs[0], mix.inputs[1])
    l.new(ramp.outputs["Color"], mix.inputs[2])
    _out(nt, mix.outputs["Color"], 0.95)
    return mat


def image_mat(name, path, strength=1.6):
    mat = fresh(name)
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Linear"
    _out(nt, tex.outputs["Color"], strength)
    return mat


def soft_emit(name, color, strength, alpha):
    """Emission that stays see-through, so beams never become solid walls."""
    mat = fresh(name)
    nt = mat.node_tree
    n, l = nt.nodes, nt.links
    tr = n.new("ShaderNodeBsdfTransparent")
    em = n.new("ShaderNodeEmission")
    em.inputs[0].default_value = color
    em.inputs[1].default_value = strength
    mix = n.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = alpha
    l.new(tr.outputs[0], mix.inputs[1])
    l.new(em.outputs[0], mix.inputs[2])
    out = n.new("ShaderNodeOutputMaterial")
    l.new(mix.outputs[0], out.inputs[0])
    mat.blend_method = "BLEND"
    mat.surface_render_method = "BLENDED"
    mat.shadow_method = "NONE"
    return mat


def apply_blend(mat):
    mat.blend_method = "BLEND"
    mat.surface_render_method = "BLENDED"
    mat.shadow_method = "NONE"
    return mat


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

def _loop_from_half(y, pts):
    right = [(x, y, z) for x, z in pts]
    left = [(-x, y, z) for x, z in reversed(pts[1:-1])]
    return right + left


def _loft(loops):
    n = len(loops[0])
    verts = [p for loop in loops for p in loop]
    faces = []
    for i in range(len(loops) - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            c = (i + 1) * n + (j + 1) % n
            d = (i + 1) * n + j
            faces.append((a, b, c, d))
    faces.append(tuple(range(n)))
    last = (len(loops) - 1) * n
    faces.append(tuple(range(last + n - 1, last - 1, -1)))
    return verts, faces


def _profile(y, zb, xb, zs, xs, z1, x1, zsh, xsh, zbelt, xbelt, zrail, xrail, zt):
    pts = [
        (0.0, zb),
        (xb, zb + 0.015),
        (xs, zs),
        (x1, z1),
        (xsh, zsh),
        (xbelt, zbelt),
        (xrail, zrail),
        (xrail * 0.45, (zrail * 0.35 + zt * 0.65)),
        (0.0, zt),
    ]
    return _loop_from_half(y, pts)


def build_hero(mats):
    # Tail -> nose. Low nose, cabin set back, haunches over the wheels.
    # Args: y, zb, xb, zs, xs, z1, x1, zsh, xsh, zbelt, xbelt, zrail, xrail, zt
    sections = [
        _profile(-2.24, 0.24, 0.62, 0.34, 0.86, 0.50, 0.96, 0.64, 0.98, 0.72, 0.78, 0.74, 0.48, 0.76),
        _profile(-2.02, 0.20, 0.70, 0.30, 0.94, 0.48, 1.02, 0.68, 1.04, 0.80, 0.86, 0.84, 0.58, 0.86),
        _profile(-1.62, 0.46, 0.52, 0.56, 0.86, 0.68, 1.02, 0.82, 1.06, 0.88, 0.92, 0.90, 0.62, 0.92),
        _profile(-1.12, 0.15, 0.66, 0.24, 0.96, 0.46, 1.02, 0.74, 1.02, 0.90, 0.80, 0.98, 0.54, 1.02),
        _profile(-0.40, 0.14, 0.68, 0.22, 0.98, 0.44, 1.02, 0.72, 1.02, 0.92, 0.74, 1.02, 0.50, 1.05),
        _profile(0.12, 0.14, 0.66, 0.22, 0.97, 0.42, 1.01, 0.70, 1.01, 0.90, 0.74, 1.02, 0.48, 1.05),
        _profile(0.52, 0.14, 0.62, 0.22, 0.95, 0.40, 1.00, 0.60, 1.00, 0.70, 0.84, 0.82, 0.56, 0.90),
        _profile(0.92, 0.14, 0.58, 0.22, 0.92, 0.36, 0.98, 0.50, 1.00, 0.54, 0.88, 0.56, 0.62, 0.58),
        _profile(1.38, 0.40, 0.48, 0.50, 0.82, 0.62, 0.98, 0.76, 1.02, 0.70, 0.84, 0.60, 0.50, 0.58),
        _profile(1.78, 0.16, 0.50, 0.26, 0.80, 0.38, 0.90, 0.48, 0.92, 0.50, 0.72, 0.48, 0.42, 0.50),
        _profile(2.08, 0.15, 0.34, 0.22, 0.60, 0.32, 0.72, 0.40, 0.74, 0.42, 0.52, 0.40, 0.28, 0.42),
        _profile(2.32, 0.17, 0.10, 0.22, 0.26, 0.28, 0.34, 0.32, 0.36, 0.34, 0.22, 0.32, 0.10, 0.33),
    ]
    verts, faces = _loft(sections)
    me = new_mesh("HeroBody", verts, faces)
    # Open the greenhouse so the camera can pass through glass only.
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.normal_update()
    score = 0.0
    for f in bm.faces:
        c = f.calc_center_median()
        if abs(c.x) > 0.35:
            score += f.normal.x * (1.0 if c.x > 0.0 else -1.0)
    if score < 0.0:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
        bm.normal_update()
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        x, y, z = abs(c.x), c.y, c.z
        windshield = (0.38 < y < 1.02 and z > 0.60 and x < 0.70)
        roof = (-0.22 < y < 0.42 and z > 0.98 and x < 0.48)
        side = (0.00 < y < 0.55 and 0.64 < z < 0.98 and x > 0.58)
        if windshield or roof or side:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(me)
    bm.free()
    body = ob_from_mesh("HeroBody", me, mats["body"])
    _ink_shell(body, mats["ink"], 0.018)

    # Glass sheets.
    glass_parts = []
    ws = [
        [(-0.68, 0.96, 0.52), (0.68, 0.96, 0.52)],
        [(-0.52, 0.60, 0.90), (0.52, 0.60, 0.90)],
        [(-0.40, 0.32, 1.02), (0.40, 0.32, 1.02)],
    ]
    glass_parts.append(_sheet("Windshield", ws, mats["glass"]))
    roof = [
        [(-0.42, 0.30, 1.035), (0.42, 0.30, 1.035)],
        [(-0.40, 0.02, 1.045), (0.40, 0.02, 1.045)],
        [(-0.38, -0.24, 1.01), (0.38, -0.24, 1.01)],
    ]
    glass_parts.append(_sheet("RoofGlass", roof, mats["glass"]))
    for sign, x in ((1, 0.86), (-1, -0.86)):
        side = [
            [(x, 0.52, 0.66), (x * 0.78, 0.52, 0.98)],
            [(x, 0.02, 0.64), (x * 0.76, 0.02, 0.98)],
        ]
        glass_parts.append(_sheet(f"SideGlass{sign}", side, mats["glass"]))

    def part(name, kind, loc, scale, mat, rot=(0, 0, 0), **kw):
        mep = primitive(kind, **kw)
        o = ob_from_mesh(name, mep, mat)
        o.location = loc
        o.scale = scale
        o.rotation_euler = rot
        o.parent = body
        return o

    # Lights, aero, wheels. Slits sit on the surface, not through the body.
    # Sit the lamps on the nose surface, not inside the volume.
    part("HL", "cube", (0.15, 2.345, 0.27), (0.11, 0.018, 0.026), mats["cyan"])
    part("HR", "cube", (-0.15, 2.345, 0.27), (0.11, 0.018, 0.026), mats["cyan"])
    part("HL2", "cube", (0.52, 2.00, 0.38), (0.20, 0.02, 0.028), mats["cyan"])
    part("HR2", "cube", (-0.52, 2.00, 0.38), (0.20, 0.02, 0.028), mats["cyan"])
    part("Tail", "cube", (0.0, -2.27, 0.58), (1.78, 0.022, 0.05), mats["magenta"])
    part("TailC", "cube", (0.0, -2.27, 0.44), (1.15, 0.018, 0.016), mats["cyan"])
    part("Split", "cube", (0.0, 2.20, 0.13), (1.15, 0.22, 0.018), mats["ink"])
    part("SkirtL", "cube", (0.98, -0.15, 0.18), (0.045, 1.15, 0.06), mats["ink"])
    part("SkirtR", "cube", (-0.98, -0.15, 0.18), (0.045, 1.15, 0.06), mats["ink"])
    part("Duck", "cube", (0.0, -1.85, 0.90), (0.72, 0.18, 0.015), mats["ink"], rot=(0.35, 0, 0))
    for i, x in enumerate((-0.34, -0.12, 0.12, 0.34)):
        part(f"Fin{i}", "cube", (x, -2.10, 0.22), (0.012, 0.18, 0.09), mats["ink"])
    part("AccentL", "cube", (1.02, 0.85, 0.46), (0.018, 0.42, 0.012), mats["cyan"])
    part("AccentR", "cube", (-1.02, 0.85, 0.46), (0.018, 0.42, 0.012), mats["cyan"])

    # Interior.
    part("Dash", "cube", (0.0, 0.55, 0.62), (1.15, 0.28, 0.08), mats["ink"])
    part("Console", "cube", (0.0, 0.28, 0.48), (0.28, 0.45, 0.10), mats["ink"])
    part("SeatL", "cube", (-0.36, -0.05, 0.48), (0.32, 0.34, 0.28), mats["ink"])
    part("SeatR", "cube", (0.36, -0.05, 0.48), (0.32, 0.34, 0.28), mats["ink"])
    part("Wheel", "torus", (-0.36, 0.42, 0.70), (0.18, 0.18, 0.18), mats["ink"], rot=(0.2, 0, 0), R=1, r=0.14, maj=16, min=6, axis="Y")
    part("Hub", "cyl", (-0.36, 0.48, 0.70), (0.05, 0.05, 0.04), mats["cyan"], rot=(1.57, 0, 0), r=1, d=1, seg=12)
    holo = part("Holo", "plane", (0.02, 0.95, 0.68), (0.42, 0.22, 1), mats["holo"], rot=(1.05, 0, 0))
    holo.scale = (0.001, 0.001, 0.001)
    # Driver: helmet and shoulders, left seat (US).
    part("Shoulder", "uv", (-0.36, -0.02, 0.62), (0.22, 0.16, 0.16), mats["ink"], u=8, v=6)
    part("Helmet", "uv", (-0.36, 0.02, 0.82), (0.13, 0.14, 0.15), mats["ink"], u=10, v=8)
    part("Visor", "cube", (-0.36, 0.10, 0.84), (0.11, 0.02, 0.035), mats["cyan"])

    wheels = []
    for name, x, y, z, radius in (
        ("FL", 0.88, 1.40, 0.34, 0.34),
        ("FR", -0.88, 1.40, 0.34, 0.34),
        ("RL", 0.92, -1.58, 0.36, 0.36),
        ("RR", -0.92, -1.58, 0.36, 0.36),
    ):
        root = bpy.data.objects.new(name, None)
        link(root)
        root.parent = body
        root.location = (x, y, z)
        root.empty_display_size = 0.1
        tire = ob_from_mesh(name + "tire", primitive("torus", R=radius * 0.78, r=radius * 0.22, maj=18, min=8, axis="X"), mats["tire"])
        tire.parent = root
        rim = ob_from_mesh(name + "rim", primitive("torus", R=radius * 0.58, r=0.018, maj=18, min=6, axis="X"), mats["rim"])
        rim.parent = root
        hub = ob_from_mesh(name + "hub", primitive("cyl", r=radius * 0.14, d=0.08, seg=12), mats["rim"])
        hub.parent = root
        hub.rotation_euler = (0, math.radians(90), 0)
        ring = ob_from_mesh(name + "ring", primitive("torus", R=radius * 0.62, r=0.012, maj=20, min=6, axis="X"), mats["cyan"])
        ring.parent = root
        spokes = bpy.data.objects.new(name + "spokes", None)
        link(spokes)
        spokes.parent = root
        for k in range(5):
            sp = ob_from_mesh(name + f"sp{k}", primitive("cube"), mats["rim"])
            sp.parent = spokes
            # Long axis in the wheel plane (YZ), spin axis is X.
            sp.scale = (0.028, radius * 0.07, radius * 0.62)
            sp.rotation_euler = (k * math.radians(72), 0, 0)
            sp.location = (0, 0, 0)
        blur = ob_from_mesh(name + "blur", primitive("cyl", r=radius * 0.92, d=0.02, seg=24), mats["blurwheel"])
        blur.parent = root
        blur.rotation_euler = (0, math.radians(90), 0)
        blur.scale = (0.01, 0.01, 0.01)
        wheels.append({"root": root, "spokes": spokes, "blur": blur, "front": name.startswith("F")})

    # Thrusters and belly jets, hidden until they fire.
    flames = []
    for i, x in enumerate((-0.34, 0.34)):
        fl = ob_from_mesh(f"Flame{i}", primitive("cone", r=0.18, d=0.9, seg=8), mats["fire_y"])
        fl.parent = body
        fl.location = (x, -2.42, 0.36)
        fl.rotation_euler = (math.radians(90), 0, 0)
        fl.scale = (0.01, 0.01, 0.01)
        flames.append(fl)
    jets = []
    for i, x in enumerate((-0.4, 0.4)):
        jt = ob_from_mesh(f"Jet{i}", primitive("cone", r=0.16, d=0.7, seg=8), mats["fire_c"])
        jt.parent = body
        jt.location = (x, 0.2, 0.12)
        jt.rotation_euler = (math.radians(-90), 0, 0)
        jt.scale = (0.01, 0.01, 0.01)
        jets.append(jt)
    # Headlight beams.
    for i, x in enumerate((0.5, -0.5)):
        beam = ob_from_mesh(f"Beam{i}", primitive("cone", r=0.18, d=2.4, seg=8), mats["beam"])
        beam.parent = body
        beam.location = (x * 0.75, 3.15, 0.36)
        beam.rotation_euler = (math.radians(-90), 0, 0)

    return {"body": body, "wheels": wheels, "flames": flames, "jets": jets, "glass": glass_parts, "holo": holo}


def _sheet(name, rows, mat):
    verts = []
    for row in rows:
        verts.extend(row)
    w = len(rows[0])
    faces = []
    for i in range(len(rows) - 1):
        for j in range(w - 1):
            a = i * w + j
            faces.append((a, a + 1, a + 1 + w, a + w))
    me = new_mesh(name, verts, faces)
    ob = ob_from_mesh(name, me, mat)
    return ob


def _ink_shell(obj, mat, thickness):
    obj.data.materials.append(mat)
    mod = obj.modifiers.new("Ink", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.material_offset = 1
    mod.use_quality_normals = True


# ---------------------------------------------------------------------------
# Other vehicles
# ---------------------------------------------------------------------------

def build_police(mats, font):
    """One hierarchy. Call again with shared meshes via copy."""
    root = bpy.data.objects.new("Police", None)
    link(root)
    root.empty_display_size = 0.2

    def box(name, loc, scale, mat, rot=(0, 0, 0)):
        o = ob_from_mesh(name, primitive("cube"), mat)
        o.parent = root
        o.location = loc
        o.scale = scale
        o.rotation_euler = rot
        return o

    body = box("PBody", (0, 0, 0.72), (1.85, 4.7, 0.85), mats["police"])
    _ink_shell(body, mats["ink"], 0.02)
    box("PCab", (0, 0.15, 1.28), (1.6, 2.3, 0.55), mats["police"])
    glass = box("PGlass", (0, 0.35, 1.32), (1.45, 1.7, 0.42), mats["glass"])
    box("PStripe", (0, 0, 0.55), (1.88, 4.72, 0.08), mats["cyan_dim"])
    # Lightbar.
    box("Bar", (0, 0.15, 1.78), (0.55, 1.45, 0.16), mats["ink"])
    for i, x in enumerate(np.linspace(-0.55, 0.55, 5)):
        mat = mats["red"] if i % 2 == 0 else mats["blue"]
        box(f"LB{i}", (x, 0.15, 1.86), (0.18, 0.24, 0.09), mat)
    # Push bumper.
    box("Push", (0, 2.45, 0.45), (1.7, 0.12, 0.28), mats["ink"])
    # Wheels.
    for (x, y) in ((0.82, 1.45), (-0.82, 1.45), (0.82, -1.45), (-0.82, -1.45)):
        w = ob_from_mesh("PW", primitive("cyl", r=0.34, d=0.22, seg=12), mats["tire"])
        w.parent = root
        w.location = (x, y, 0.34)
        w.rotation_euler = (0, math.radians(90), 0)
    # 警 on the door and hood.
    badge = text_mesh("警", font, 0.78, 0.025)
    for i, loc, rot in (
        (0, (0.94, 0.15, 0.85), (math.radians(90), 0, math.radians(90))),
        (1, (-0.94, 0.15, 0.85), (math.radians(90), 0, math.radians(-90))),
        (2, (0.0, 1.55, 0.95), (0, 0, 0)),
        # Rear face, so a chase view reads the character.
        (3, (0.0, -2.45, 0.95), (math.radians(90), 0, 0)),
    ):
        o = ob_from_mesh(f"Jing{i}", badge, mats["ink"])
        o.parent = root
        o.location = loc
        o.rotation_euler = rot
    return root


def text_mesh(body, font, size, extrude):
    key = f"glyph_{body}_{size}"
    if key in bpy.data.meshes:
        return bpy.data.meshes[key]
    curve = bpy.data.curves.new(key, "FONT")
    curve.body = body
    curve.font = font
    curve.size = size
    curve.extrude = extrude
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.resolution_u = 6
    tmp = bpy.data.objects.new(key, curve)
    link(tmp)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = tmp
    tmp.select_set(True)
    bpy.ops.object.convert(target="MESH")
    me = tmp.data
    me.name = key
    bpy.data.objects.remove(tmp, do_unlink=True)
    return me


def build_heli(mats, font):
    root = bpy.data.objects.new("Heli", None)
    link(root)

    def part(name, kind, loc, scale, mat, rot=(0, 0, 0), **kw):
        o = ob_from_mesh(name, primitive(kind, **kw), mat)
        o.parent = root
        o.location = loc
        o.scale = scale
        o.rotation_euler = rot
        return o

    body = part("HBody", "uv", (0, 0.2, 0), (0.95, 2.3, 0.75), mats["heli"], u=12, v=8)
    _ink_shell(body, mats["ink"], 0.03)
    part("HNose", "uv", (0, 2.15, -0.1), (0.55, 0.7, 0.48), mats["glass"], u=10, v=6)
    part("Boom", "cyl", (0, -3.3, 0.15), (0.22, 0.22, 4.2), mats["heli"], rot=(math.radians(90), 0, 0), r=1, d=1)
    part("Fin", "cube", (0, -5.3, 0.55), (0.08, 0.45, 0.7), mats["heli"])
    part("TailR", "cube", (0.35, -5.2, 0.35), (0.55, 0.08, 0.16), mats["red"])
    part("SkidL", "cyl", (0.55, 0.2, -0.85), (0.05, 0.05, 2.4), mats["ink"], rot=(math.radians(90), 0, 0), r=1, d=1, seg=8)
    part("SkidR", "cyl", (-0.55, 0.2, -0.85), (0.05, 0.05, 2.4), mats["ink"], rot=(math.radians(90), 0, 0), r=1, d=1, seg=8)
    rotor = bpy.data.objects.new("Rotor", None)
    link(rotor)
    rotor.parent = root
    rotor.location = (0, 0.3, 0.85)
    for k in range(4):
        blade = ob_from_mesh(f"Blade{k}", primitive("cube"), mats["ink"])
        blade.parent = rotor
        blade.scale = (5.3, 0.16, 0.035)
        blade.rotation_euler = (0, 0, k * math.pi / 2)
    disc = ob_from_mesh("RotorDisc", primitive("cyl", r=5.1, d=0.02, seg=24), mats["disc"])
    disc.parent = rotor
    # Searchlight cone hangs under the nose, aimed down.
    cone = ob_from_mesh("Search", primitive("cone", r=2.2, d=18.0, seg=12), mats["search"])
    cone.parent = root
    cone.location = (0.0, 1.4, -9.0)
    badge = text_mesh("警", font, 0.4, 0.02)
    o = ob_from_mesh("HJing", badge, mats["yellow"])
    o.parent = root
    o.location = (0.96, 0.2, 0.1)
    o.rotation_euler = (math.radians(90), 0, math.radians(90))
    return {"root": root, "rotor": rotor, "search": cone}


def build_tanker(mats):
    root = bpy.data.objects.new("Tanker", None)
    link(root)

    def part(name, kind, loc, scale, mat, rot=(0, 0, 0), **kw):
        o = ob_from_mesh(name, primitive(kind, **kw), mat)
        o.parent = root
        o.location = loc
        o.scale = scale
        o.rotation_euler = rot
        return o

    # Long axis is local X so it can sit across the road.
    cab = part("Cab", "cube", (-5.3, 0, 1.3), (2.2, 2.4, 2.2), mats["police"])
    _ink_shell(cab, mats["ink"], 0.04)
    part("Tank", "cyl", (1.2, 0, 1.7), (1.35, 1.35, 8.5), mats["yellow"], rot=(0, math.radians(90), 0), r=1, d=1, seg=16)
    part("Stripe", "cyl", (1.2, 0, 1.7), (1.42, 1.42, 1.2), mats["magenta"], rot=(0, math.radians(90), 0), r=1, d=1, seg=16)
    for x in (-4.6, -2.2, 1.0, 3.6):
        w = part(f"TW{x}", "cyl", (x, 1.15, 0.45), (0.45, 0.45, 0.3), mats["tire"], rot=(math.radians(90), 0, 0), r=1, d=1, seg=10)
        w2 = part(f"TW2{x}", "cyl", (x, -1.15, 0.45), (0.45, 0.45, 0.3), mats["tire"], rot=(math.radians(90), 0, 0), r=1, d=1, seg=10)
    return root


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = T.FPS
    scene.render.fps_base = 1
    scene.frame_start = 1
    scene.frame_end = T.N
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 6
    scene.render.filter_size = 1.4
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.15
    scene.view_settings.gamma = 1.0
    ee = scene.eevee
    ee.use_raytracing = False
    ee.use_shadows = False
    ee.use_gtao = False
    ee.use_ssr = False
    ee.use_bloom = False
    ee.use_volumetric_lights = False
    ee.use_volumetric_shadows = False
    ee.taa_render_samples = 1
    scene.render.use_motion_blur = False
    world = bpy.data.worlds.new("Night")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.012, 0.016, 0.04, 1)
    bg.inputs[1].default_value = 1.0
    scene["sim_t"] = 0.0
    if scene.camera is None:
        cam_d = bpy.data.cameras.new("Cam")
        cam = bpy.data.objects.new("Cam", cam_d)
        link(cam)
        scene.camera = cam
    scene.camera.data.sensor_fit = "HORIZONTAL"
    scene.camera.data.sensor_width = 36.0
    scene.camera.data.clip_start = 0.04
    scene.camera.data.clip_end = 2500.0
    return scene


def window_mat(name="Bldg"):
    """Metre-space window grid on vertical walls. Roofs stay dark."""
    mat = fresh(name)
    nt = mat.node_tree
    n, l = nt.nodes, nt.links
    geo = n.new("ShaderNodeNewGeometry")
    nsep = n.new("ShaderNodeSeparateXYZ")
    l.new(geo.outputs["Normal"], nsep.inputs[0])
    psep = n.new("ShaderNodeSeparateXYZ")
    l.new(geo.outputs["Position"], psep.inputs[0])
    up = n.new("ShaderNodeVectorMath")
    up.operation = "CROSS_PRODUCT"
    up.inputs[1].default_value = (0.0, 0.0, 1.0)
    l.new(geo.outputs["Normal"], up.inputs[0])
    udot = n.new("ShaderNodeVectorMath")
    udot.operation = "DOT_PRODUCT"
    l.new(geo.outputs["Position"], udot.inputs[0])
    l.new(up.outputs["Vector"], udot.inputs[1])

    def math_node(op, a=None, b=None, bval=None):
        nd = n.new("ShaderNodeMath")
        nd.operation = op
        if a is not None:
            l.new(a, nd.inputs[0])
        if b is not None:
            l.new(b, nd.inputs[1])
        if bval is not None:
            nd.inputs[1].default_value = bval
        return nd.outputs[0]

    u = math_node("MULTIPLY", udot.outputs["Value"], bval=0.42)
    v = math_node("MULTIPLY", psep.outputs["Z"], bval=0.30)
    fu = math_node("FRACT", u if False else None)
    # FRACT takes one input; wire it explicitly.
    fu = n.new("ShaderNodeMath")
    fu.operation = "FRACT"
    l.new(math_node("MULTIPLY", udot.outputs["Value"], bval=0.42), fu.inputs[0])
    fv = n.new("ShaderNodeMath")
    fv.operation = "FRACT"
    l.new(math_node("MULTIPLY", psep.outputs["Z"], bval=0.30), fv.inputs[0])
    cu = n.new("ShaderNodeMath")
    cu.operation = "FLOOR"
    l.new(math_node("MULTIPLY", udot.outputs["Value"], bval=0.42), cu.inputs[0])
    cv = n.new("ShaderNodeMath")
    cv.operation = "FLOOR"
    l.new(math_node("MULTIPLY", psep.outputs["Z"], bval=0.30), cv.inputs[0])

    # hash = fract(sin(cx * 12.9898 + cz * 78.233) * 43758.5453)
    hx = n.new("ShaderNodeMath")
    hx.operation = "MULTIPLY"
    hx.inputs[1].default_value = 12.9898
    l.new(cu.outputs[0], hx.inputs[0])
    hz = n.new("ShaderNodeMath")
    hz.operation = "MULTIPLY"
    hz.inputs[1].default_value = 78.233
    l.new(cv.outputs[0], hz.inputs[0])
    hs = n.new("ShaderNodeMath")
    hs.operation = "ADD"
    l.new(hx.outputs[0], hs.inputs[0])
    l.new(hz.outputs[0], hs.inputs[1])
    sn = n.new("ShaderNodeMath")
    sn.operation = "SINE"
    l.new(hs.outputs[0], sn.inputs[0])
    hm = n.new("ShaderNodeMath")
    hm.operation = "MULTIPLY"
    hm.inputs[1].default_value = 43758.5453
    l.new(sn.outputs[0], hm.inputs[0])
    hf = n.new("ShaderNodeMath")
    hf.operation = "FRACT"
    l.new(hm.outputs[0], hf.inputs[0])

    # Frame: fract near 0 or 1.
    def away(fract_node, lo):
        gt = n.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = lo
        l.new(fract_node.outputs[0], gt.inputs[0])
        lt = n.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        lt.inputs[1].default_value = 1.0 - lo
        l.new(fract_node.outputs[0], lt.inputs[0])
        both = n.new("ShaderNodeMath")
        both.operation = "MULTIPLY"
        l.new(gt.outputs[0], both.inputs[0])
        l.new(lt.outputs[0], both.inputs[1])
        return both.outputs[0]

    in_u = away(fu, 0.16)
    in_v = away(fv, 0.18)
    inside = n.new("ShaderNodeMath")
    inside.operation = "MULTIPLY"
    l.new(in_u, inside.inputs[0])
    l.new(in_v, inside.inputs[1])
    lit = n.new("ShaderNodeMath")
    lit.operation = "GREATER_THAN"
    lit.inputs[1].default_value = 0.58
    l.new(hf.outputs[0], lit.inputs[0])
    win = n.new("ShaderNodeMath")
    win.operation = "MULTIPLY"
    l.new(inside.outputs[0], win.inputs[0])
    l.new(lit.outputs[0], win.inputs[1])

    # Four neon colors from the hash.
    c1 = n.new("ShaderNodeMixRGB")
    c1.inputs[1].default_value = MAGENTA
    c1.inputs[2].default_value = CYAN
    band = n.new("ShaderNodeMath")
    band.operation = "GREATER_THAN"
    band.inputs[1].default_value = 0.78
    l.new(hf.outputs[0], band.inputs[0])
    l.new(band.outputs[0], c1.inputs[0])
    c2 = n.new("ShaderNodeMixRGB")
    c2.inputs[1].default_value = YELLOW
    c2.inputs[2].default_value = PURPLE
    band2 = n.new("ShaderNodeMath")
    band2.operation = "GREATER_THAN"
    band2.inputs[1].default_value = 0.90
    l.new(hf.outputs[0], band2.inputs[0])
    l.new(band2.outputs[0], c2.inputs[0])
    pick = n.new("ShaderNodeMath")
    pick.operation = "GREATER_THAN"
    pick.inputs[1].default_value = 0.84
    l.new(hf.outputs[0], pick.inputs[0])
    neon = n.new("ShaderNodeMixRGB")
    l.new(pick.outputs[0], neon.inputs[0])
    l.new(c1.outputs["Color"], neon.inputs[1])
    l.new(c2.outputs["Color"], neon.inputs[2])

    wall = (0.018, 0.022, 0.045, 1)
    glass_dark = (0.012, 0.016, 0.035, 1)
    pane = n.new("ShaderNodeMixRGB")
    l.new(win.outputs[0], pane.inputs[0])
    pane.inputs[1].default_value = glass_dark
    l.new(neon.outputs["Color"], pane.inputs[2])

    absz = n.new("ShaderNodeMath")
    absz.operation = "ABSOLUTE"
    l.new(nsep.outputs["Z"], absz.inputs[0])
    roof = n.new("ShaderNodeMath")
    roof.operation = "GREATER_THAN"
    roof.inputs[1].default_value = 0.72
    l.new(absz.outputs[0], roof.inputs[0])
    facade = n.new("ShaderNodeMixRGB")
    l.new(roof.outputs[0], facade.inputs[0])
    l.new(pane.outputs["Color"], facade.inputs[1])
    facade.inputs[2].default_value = wall

    # Ink contour on the silhouette.
    lw = n.new("ShaderNodeLayerWeight")
    inv = n.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    l.new(lw.outputs["Facing"], inv.inputs[1])
    edge = n.new("ShaderNodeMath")
    edge.operation = "POWER"
    edge.inputs[1].default_value = 2.4
    l.new(inv.outputs[0], edge.inputs[0])
    inked = n.new("ShaderNodeMixRGB")
    l.new(edge.outputs[0], inked.inputs[0])
    l.new(facade.outputs["Color"], inked.inputs[1])
    inked.inputs[2].default_value = INK
    _out(nt, inked.outputs["Color"], 1.35)
    return mat


def make_mats():
    mats = {
        "body": toon("Body", (0.05, 0.055, 0.075, 1), (0.015, 0.018, 0.028, 1),
                     (0.32, 0.36, 0.48, 1), CYAN, MAGENTA, halftone=False),
        "ink": emission("Ink", INK, 1.0),
        # Strength stays near 1 so the hue survives the display transform.
        "cyan": emission("Cyan", CYAN, 1.65),
        "magenta": emission("Magenta", MAGENTA, 1.5),
        "yellow": emission("Yellow", YELLOW, 1.4),
        "purple": emission("Purple", PURPLE, 1.35),
        "cyan_dim": emission("CyanDim", CYAN, 0.7),
        "red": emission("Red", (1, 0.05, 0.08, 1), 2.2),
        "blue": emission("Blue", (0.15, 0.35, 1, 1), 2.2),
        "glass": glass_mat(),
        "tire": toon("Tire", (0.02, 0.02, 0.025, 1), (0.0, 0.0, 0.0, 1),
                     (0.08, 0.08, 0.09, 1), (0.02, 0.02, 0.03, 1), (0.02, 0.02, 0.03, 1)),
        "rim": emission("Rim", (0.16, 0.17, 0.20, 1), 1.0),
        "police": toon("Police", (0.93, 0.95, 0.97, 1), (0.62, 0.66, 0.74, 1),
                       (1, 1, 1, 1), (0.1, 0.3, 0.8, 1), (0.8, 0.1, 0.15, 1), halftone=False),
        "heli": toon("Heli", (0.05, 0.08, 0.16, 1), (0.02, 0.03, 0.06, 1),
                     (0.25, 0.32, 0.5, 1), CYAN, YELLOW, halftone=True),
        "road": road_mat(),
        "holo": apply_blend(emission("Holo", CYAN, 2.2)),
        "fire_y": emission("FireY", YELLOW, 4.5),
        "fire_m": emission("FireM", MAGENTA, 4.0),
        "fire_c": emission("FireC", CYAN, 3.6),
        "fire_p": emission("FireP", PURPLE, 3.2),
        "smoke": apply_blend(emission("Smoke", (0.08, 0.07, 0.12, 1), 1.2)),
        "beam": soft_emit("Beam", (0.45, 0.95, 1.0, 1), 1.3, 0.10),
        "search": soft_emit("Search", (1.0, 0.97, 0.82, 1), 1.6, 0.08),
        "disc": soft_emit("Disc", (0.8, 0.9, 1.0, 1), 0.6, 0.06),
        "blurwheel": emission("BlurW", (0.04, 0.05, 0.07, 1), 1.0),
        "bldg": window_mat("Bldg"),
        "sign": None,
    }
    # Blink drivers for police lights.
    for mat, phase in ((mats["red"], 0.0), (mats["blue"], math.pi)):
        node = None
        for nd in mat.node_tree.nodes:
            if nd.type == "EMISSION":
                node = nd
        if node is None:
            continue
        drv = node.inputs[1].driver_add("default_value").driver
        drv.expression = f"2.1 * max(0.12, sin(sim * 9.5 + {phase:.4f}))"
        var = drv.variables.new()
        var.name = "sim"
        var.targets[0].id_type = "SCENE"
        var.targets[0].id = bpy.context.scene
        var.targets[0].data_path = '["sim_t"]'
    return mats


def sample_road(data, s):
    pos, fwd, right, up = data["road_pos"], data["road_fwd"], data["road_right"], data["road_up"]
    s = float(np.clip(s, 0.0, len(pos) - 1.001))
    i = int(s)
    u = s - i
    j = min(i + 1, len(pos) - 1)
    def mix(a, b):
        return a * (1 - u) + b * u
    p = mix(pos[i], pos[j])
    f = mix(fwd[i], fwd[j])
    r = mix(right[i], right[j])
    f = f / np.linalg.norm(f)
    r = r - f * np.dot(r, f)
    r = r / np.linalg.norm(r)
    uu = np.cross(r, f)
    return p, f, r, uu


def build_road(data, mats):
    s_edge = data["s_edge"]
    ss = np.arange(0.0, s_edge, 2.0)
    verts, uvs, faces = [], [], []
    for k, s in enumerate(ss):
        p, f, r, u = sample_road(data, s)
        w = T.road_width(float(s)) * 0.5
        verts.append(tuple(p - r * w))
        verts.append(tuple(p + r * w))
        uvs.append((0.0, float(s) / 40.0))
        uvs.append((1.0, float(s) / 40.0))
    # Break the last edge so the cliff reads as a snapped road.
    verts[-1] = (verts[-1][0], verts[-1][1], verts[-1][2] - 1.4)
    verts[-2] = (verts[-2][0], verts[-2][1], verts[-2][2] - 0.6)
    for k in range(len(ss) - 1):
        a = 2 * k
        faces.append((a, a + 1, a + 3, a + 2))
    me = new_mesh("Road", verts, faces, uvs)
    ob = ob_from_mesh("Road", me, mats["road"])
    # Glowing center ribbon — the hijacked lane.
    verts, uvs, faces = [], [], []
    for k, s in enumerate(ss):
        p, f, r, u = sample_road(data, s)
        p = p + np.array([0, 0, 0.04])
        w = 0.55
        verts.append(tuple(p - r * w))
        verts.append(tuple(p + r * w))
        uvs.append((0.0, k))
        uvs.append((1.0, k))
    for k in range(len(ss) - 1):
        a = 2 * k
        faces.append((a, a + 1, a + 3, a + 2))
    ribbon_mat = emission("Ribbon", CYAN, 0.0)
    me2 = new_mesh("Ribbon", verts, faces, uvs)
    ribbon = ob_from_mesh("Ribbon", me2, ribbon_mat)
    # Cross street at the tanker.
    s_tank = data["s_tank"]
    p, f, r, u = sample_road(data, s_tank)
    cv, cf = [], []
    length, width = 70.0, 14.0
    for i, t in enumerate(np.linspace(-length, length, 16)):
        a = p + r * t - f * width * 0.5
        b = p + r * t + f * width * 0.5
        cv += [tuple(a), tuple(b)]
    for i in range(15):
        q = 2 * i
        cf.append((q, q + 1, q + 3, q + 2))
    cross = ob_from_mesh("Cross", new_mesh("Cross", cv, cf), mats["road"])
    return ribbon


def build_city(data, mats, font, images):
    cube = primitive("cube")
    shade_flat(cube)
    rng = np.random.default_rng(4)
    s = 8.0
    signs = ["龍門", "夜雨", "永樂", "雲吞", "酒吧", "藥房", "霓虹", "通宵", "金龍", "冰室", "電玩", "麵館", "旅店", "發財", "夜色", "星河"]
    sign_i = 0
    bill_i = 0
    plane = primitive("plane")
    while s < min(1180.0, data["s_edge"] - 30):
        # Keep the intersection clear.
        if abs(s - data["s_tank"]) < 22 or abs(s - data["s_bar"]) < 14:
            s += 18
            continue
        p, f, r, u = sample_road(data, s)
        half = T.road_width(float(s)) * 0.5
        for side in (-1.0, 1.0):
            depth = float(rng.uniform(12, 22))
            width = float(rng.uniform(10, 20))
            height = float(rng.uniform(16, 70))
            if rng.random() < 0.18:
                height = float(rng.uniform(80, 150))
            center = p + r * side * (half + depth * 0.55 + 1.2)
            center = center + np.array([0, 0, height * 0.5])
            ob = ob_from_mesh(f"B{int(s)}_{int(side)}", cube, mats["bldg"])
            ob.location = Vector(center)
            ob.scale = (width, depth, height)
            # Yaw the building along the street.
            yaw = math.atan2(f[0], f[1])
            ob.rotation_euler = (0, 0, yaw)
            if rng.random() < 0.55 and sign_i < len(signs):
                me = text_mesh(signs[sign_i], font, 1.1, 0.04)
                col = [mats["magenta"], mats["cyan"], mats["yellow"], mats["purple"]][sign_i % 4]
                so = ob_from_mesh(f"Sign{sign_i}", me, col)
                so.location = Vector(p + r * side * (half + 0.6) + np.array([0, 0, 6 + (sign_i % 4) * 3.5]))
                # Face the street.
                normal = -r * side
                so.rotation_euler = _align_z(normal, f)
                sign_i += 1
            if rng.random() < 0.45 and bill_i < len(images) * 2:
                img = images[bill_i % len(images)]
                bo = ob_from_mesh(f"Ad{bill_i}", plane, img)
                bw, bh = (10.0, 5.6) if bill_i % 5 else (4.2, 9.0)
                bo.scale = (bw, bh, 1)
                bo.location = Vector(center + r * (-side) * (depth * 0.5) + np.array([0, 0, height * 0.15]))
                bo.rotation_euler = _align_z(-r * side, np.array([0, 0, 1.0]))
                bill_i += 1
        # Lamps.
        if int(s) % 36 < 4:
            lamp = ob_from_mesh(f"Lamp{int(s)}", cube, mats["yellow"])
            lamp.location = Vector(p + r * (half - 0.4) + np.array([0, 0, 6.5]))
            lamp.scale = (0.25, 0.25, 0.25)
            pole = ob_from_mesh(f"Pole{int(s)}", cube, mats["ink"])
            pole.location = Vector(p + r * (half - 0.4) + np.array([0, 0, 3.2]))
            pole.scale = (0.08, 0.08, 3.2)
        s += float(rng.uniform(16, 26))


def _align_z(normal, up_hint):
    n = Vector(normal).normalized()
    up = Vector(up_hint).normalized()
    if abs(n.dot(up)) > 0.95:
        up = Vector((0, 1, 0))
    quat = n.to_track_quat("Z", "Y")
    # Keep the sign upright-ish: rebuild with a projected up.
    z = n
    x = up.cross(z)
    if x.length < 1e-4:
        x = Vector((1, 0, 0))
    x.normalize()
    y = z.cross(x)
    rot = Matrix((x, y, z)).transposed().to_euler()
    return rot


def build_slabs(data, mats):
    s0 = data["s_curl"] - 10
    slabs = []
    cube = primitive("cube")
    for i in range(18):
        s = s0 + i * 8.0
        if s > data["s_edge"] - 40:
            break
        p, f, r, u = sample_road(data, s)
        hinge = p + np.array([0, 0, 0.08]) - r * 1.2
        empty = bpy.data.objects.new(f"Hinge{i}", None)
        link(empty)
        empty.location = Vector(hinge)
        # Local Y along the road, so a Y rotation lifts the outer edge? We'll bake matrices.
        yaw = math.atan2(f[0], f[1])
        empty.rotation_euler = (0, 0, yaw)
        plate = ob_from_mesh(f"Slab{i}", cube, mats["road"])
        plate.parent = empty
        plate.location = (-3.6, 0, 0.05)
        plate.scale = (3.4, 3.6, 0.08)
        slabs.append(empty)
    return slabs


def build_screen(data, mats, face_mat):
    s = data["s_screen"] + 1.4
    p, f, r, u = sample_road(data, s)
    root = bpy.data.objects.new("Screen", None)
    link(root)
    root.location = Vector(p + np.array([0, 0, 4.6]))
    yaw = math.atan2(f[0], f[1])
    root.rotation_euler = (0, 0, yaw)
    panel = ob_from_mesh("ScreenPanel", primitive("plane"), face_mat)
    panel.parent = root
    panel.scale = (14.0, 8.0, 1)
    panel.rotation_euler = (math.radians(90), 0, 0)
    frame = ob_from_mesh("ScreenFrame", primitive("cube"), mats["ink"])
    frame.parent = root
    frame.location = (0, 0, 0)
    frame.scale = (14.6, 0.25, 8.5)
    return root, panel


def build_barricade(data, mats):
    s = data["s_bar"] + 2.2
    p, f, r, u = sample_road(data, s)
    bars = []
    cube = primitive("cube")
    for i, x in enumerate(np.linspace(-6.5, 6.5, 6)):
        o = ob_from_mesh(f"Bar{i}", cube, mats["yellow"] if i % 2 == 0 else mats["ink"])
        o.location = Vector(p + r * x + np.array([0, 0, 0.55]))
        yaw = math.atan2(f[0], f[1])
        o.rotation_euler = (0, 0, yaw)
        o.scale = (1.5, 0.25, 0.55)
        bars.append(o)
    return bars


def build_pads(data, mats):
    pads = []
    cube = primitive("cube")
    for name, s, yaw_extra in (
        ("Pad1", data["s"][int(round(T.T_LAUNCH1 * T.FPS))] - 1.0, 0.0),
        ("Pad2", data["s_tank"] - 1.0, 0.0),
    ):
        p, f, r, u = sample_road(data, float(s))
        o = ob_from_mesh(name, cube, mats["holo"])
        # A ramp: thin, long, tilted up toward +forward.
        o.location = Vector(p + f * 3.0 + np.array([0, 0, 0.7]))
        yaw = math.atan2(f[0], f[1])
        o.rotation_euler = (math.radians(-18), 0, yaw)
        o.scale = (3.2, 7.5, 0.08)
        pads.append(o)
    return pads


def build_tunnel(data, mats):
    """Towers around the flight path after ignition, with a light at the end."""
    hero = data["hero_pos"]
    fwd = data["hero_fwd"]
    i0 = int(round((T.T_IGNITE + 0.45) * T.FPS))
    i1 = T.N - 1
    origin = hero[i0]
    direction = hero[i1] - hero[i0]
    direction[2] = 0
    dist = np.linalg.norm(direction)
    if dist < 1:
        direction = fwd[i0]
        dist = 200
    direction = direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    cube = primitive("cube")
    rng = np.random.default_rng(11)
    length = max(float(dist) + 80.0, 220.0)
    yaw = math.atan2(direction[0], direction[1])
    s = 36.0
    while s < length:
        for side in (-1.0, 1.0):
            h = float(rng.uniform(70, 160))
            w = float(rng.uniform(12, 20))
            c = origin + direction * s + right * side * 32.0
            c[2] = origin[2] - 10 + h * 0.5
            ob = ob_from_mesh(f"Tower{int(s)}_{int(side)}", cube, mats["bldg"])
            ob.location = Vector(c)
            ob.rotation_euler = (0, 0, yaw)
            ob.scale = (w, 14, h)
        if int(s) % 48 < 8 and s > 20:
            # Overhead bridge, so the towers read as a tunnel.
            bridge = ob_from_mesh(f"Bridge{int(s)}", cube, mats["bldg"])
            c = origin + direction * s
            c[2] = origin[2] + 18
            bridge.location = Vector(c)
            bridge.rotation_euler = (0, 0, yaw)
            bridge.scale = (72, 5, 2.0)
        s += float(rng.uniform(22, 34))
    glow = ob_from_mesh("ExitLight", primitive("plane"), emission("Exit", (1.0, 0.92, 0.7, 1), 2.4))
    end = origin + direction * (length + 10)
    end[2] = origin[2] + 2
    glow.location = Vector(end)
    # Face back toward the car.
    glow.rotation_euler = _align_z(-direction, up)
    glow.scale = (16, 10, 1)
    # Abyss lights under the cliff.
    p, f, r, u = sample_road(data, data["s_edge"])
    for k in range(70):
        ox = float(rng.uniform(-80, 80))
        oy = float(rng.uniform(-40, 140))
        lamp = ob_from_mesh(f"Abyss{k}", cube, [mats["magenta"], mats["cyan"], mats["yellow"], mats["purple"]][k % 4])
        lamp.location = Vector(p + r * ox + f * oy + np.array([0, 0, -55]))
        lamp.scale = (1.2, 1.2, float(rng.uniform(2, 10)))


def build_rain(scene):
    """Camera-following streaks. Fall distance is driven by sim_t, so bullet-time hangs."""
    try:
        parent = bpy.data.objects.new("RainRig", None)
        link(parent)
        parent.empty_display_size = 0.2
        streak_me = primitive("cube")
        mat = emission("Rain", (0.75, 0.9, 1.0, 1), 1.8)
        rng = random.Random(3)
        streaks = []
        for i in range(220):
            o = ob_from_mesh(f"Drop{i}", streak_me, mat)
            o.parent = parent
            o.location = (rng.uniform(-18, 18), rng.uniform(-8, 28), 0.0)
            o.scale = (0.015, 0.015, rng.uniform(0.35, 0.9))
            phase = rng.uniform(0, 26)
            speed = rng.uniform(14, 22)
            drv = o.driver_add("location", 2).driver
            drv.expression = f"(({phase:.3f} - sim * {speed:.3f}) % 26.0) - 13.0"
            var = drv.variables.new()
            var.name = "sim"
            var.targets[0].id_type = "SCENE"
            var.targets[0].id = scene
            var.targets[0].data_path = '["sim_t"]'
            streaks.append(o)
        return parent
    except Exception as exc:
        print("RAIN FAILED", exc)
        return None


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def _quat_from_axes(right, forward, up):
    rot = Matrix((
        (right[0], forward[0], up[0]),
        (right[1], forward[1], up[1]),
        (right[2], forward[2], up[2]),
    ))
    return rot.to_quaternion()


def _look_quat(loc, target):
    direction = Vector(target) - Vector(loc)
    if direction.length < 1e-6:
        return Matrix.Identity(3).to_quaternion()
    # -Z looks at the target, Y stays world-up. Same-axis up rolls the camera.
    return direction.to_track_quat("-Z", "Y")


def bake_curves(obj, frames, mapping, step=1, interpolation="LINEAR"):
    """mapping: data_path -> (index -> values array)."""
    obj.animation_data_create()
    act = bpy.data.actions.new(obj.name + "_Act")
    obj.animation_data.action = act
    n = len(frames)
    use = list(range(0, n, step))
    fr = [frames[i] for i in use]
    for path, channels in mapping.items():
        for index, values in channels.items():
            try:
                fc = act.fcurves.new(path, index=index)
            except TypeError:
                fc = act.fcurves.new(path)
            fc.keyframe_points.add(len(fr))
            co = np.empty(len(fr) * 2, dtype=np.float64)
            co[0::2] = fr
            co[1::2] = [float(values[i]) for i in use]
            fc.keyframe_points.foreach_set("co", co.tolist())
            for kp in fc.keyframe_points:
                kp.interpolation = interpolation
            fc.update()


def bake(data, hero, cops, heli, rain, screen, barricade, pads, slabs):
    frames = list(range(1, T.N + 1))
    # Hero
    quats = []
    locs = data["hero_pos"]
    prev = None
    for i in range(T.N):
        q = _quat_from_axes(data["hero_right"][i], data["hero_fwd"][i], data["hero_up"][i])
        if prev is not None and prev.dot(q) < 0:
            q = -q
        # mathutils Quaternion dot via x
        if prev is not None:
            if (prev.x * q.x + prev.y * q.y + prev.z * q.z + prev.w * q.w) < 0:
                q.negate()
        quats.append(q)
        prev = q
    body = hero["body"]
    body.rotation_mode = "QUATERNION"
    bake_curves(
        body, frames,
        {
            "location": {0: locs[:, 0], 1: locs[:, 1], 2: locs[:, 2]},
            "rotation_quaternion": {
                0: [q.w for q in quats],
                1: [q.x for q in quats],
                2: [q.y for q in quats],
                3: [q.z for q in quats],
            },
        },
    )
    # Wheels: spin on local X, steer on local Z for the fronts. Blur vs spokes.
    for w in hero["wheels"]:
        root = w["root"]
        spin = -data["wheel"]
        steer = data["steer"] if w["front"] else np.zeros(T.N)
        bake_curves(
            root, frames,
            {"rotation_euler": {0: spin, 1: np.zeros(T.N), 2: steer}},
        )
        spoke_s = np.clip((18.0 - data["speed"]) / 8.0, 0.0, 1.0)
        blur_s = 1.0 - spoke_s
        bake_curves(w["spokes"], frames, {"scale": {0: spoke_s, 1: spoke_s, 2: spoke_s}})
        bake_curves(w["blur"], frames, {"scale": {0: blur_s, 1: blur_s, 2: np.full(T.N, 0.2)}})

    # Flames / jets on twos.
    def pulse(obj, t0, t1, radius):
        sx, sy, sz = [], [], []
        for i in range(T.N):
            tt = data["t"][i]
            if t0 <= tt <= t1 and (i % 2 == 0):
                flick = 0.75 + 0.35 * ((i // 2) % 3 == 0)
                sx.append(radius * flick)
                sy.append(radius * flick)
                sz.append(radius * (1.4 + 0.4 * ((i // 3) % 2)))
            elif t0 <= tt <= t1:
                # hold by copying later; filled below
                sx.append(sx[-1] if sx else 0.01)
                sy.append(sy[-1] if sy else 0.01)
                sz.append(sz[-1] if sz else 0.01)
            else:
                sx.append(0.01)
                sy.append(0.01)
                sz.append(0.01)
        bake_curves(obj, frames, {"scale": {0: sx, 1: sy, 2: sz}}, step=2, interpolation="CONSTANT")

    for fl in hero["flames"]:
        # Two burns: the tanker hop, and the cliff ignition through the end.
        # Bake as the max of both windows by doing a custom array.
        sx, sy, sz = [], [], []
        for i in range(T.N):
            tt = data["t"][i]
            on = (T.T_LAUNCH2 - 0.02) <= tt <= (T.T_LAND2 + 0.05) or tt >= T.T_IGNITE
            if on:
                flick = 0.8 + 0.45 * ((i // 2) % 2)
                sx.append(0.55 * flick)
                sy.append(0.55 * flick)
                sz.append(1.3 * flick)
            else:
                sx.append(0.01); sy.append(0.01); sz.append(0.01)
        bake_curves(fl, frames, {"scale": {0: sx, 1: sy, 2: sz}}, step=2, interpolation="CONSTANT")
    for jt in hero["jets"]:
        sx, sy, sz = [], [], []
        for i in range(T.N):
            tt = data["t"][i]
            if tt >= T.T_IGNITE:
                flick = 0.85 + 0.4 * ((i // 2) % 2)
                sx.append(0.7 * flick); sy.append(0.7 * flick); sz.append(1.1 * flick)
            else:
                sx.append(0.01); sy.append(0.01); sz.append(0.01)
        bake_curves(jt, frames, {"scale": {0: sx, 1: sy, 2: sz}}, step=2, interpolation="CONSTANT")

    # Cockpit map stays dark until the camera is inside.
    holo = hero["holo"]
    hon = [1.0 if data["t"][i] >= T.T_GLASS_IN + 0.15 else 0.001 for i in range(T.N)]
    bake_curves(
        holo, frames,
        {"scale": {0: [0.42 * s for s in hon], 1: [0.22 * s for s in hon], 2: hon}},
        interpolation="CONSTANT",
    )

    # Camera
    cam = bpy.context.scene.camera
    cam.rotation_mode = "QUATERNION"
    cqs = []
    prev = None
    lenses = []
    for i in range(T.N):
        q = _look_quat(data["cam_pos"][i], data["cam_tgt"][i])
        if prev is not None and (prev.x * q.x + prev.y * q.y + prev.z * q.z + prev.w * q.w) < 0:
            q.negate()
        cqs.append(q)
        prev = q
        fov = float(data["fov"][i])
        lenses.append(36.0 / (2.0 * math.tan(math.radians(fov) / 2.0)))
    bake_curves(
        cam, frames,
        {
            "location": {0: data["cam_pos"][:, 0], 1: data["cam_pos"][:, 1], 2: data["cam_pos"][:, 2]},
            "rotation_quaternion": {
                0: [q.w for q in cqs], 1: [q.x for q in cqs],
                2: [q.y for q in cqs], 3: [q.z for q in cqs],
            },
        },
    )
    bake_curves(cam.data, frames, {"lens": {0: lenses}})

    # sim_t on the scene
    scene = bpy.context.scene
    scene.animation_data_create()
    act = bpy.data.actions.new("SceneAct")
    scene.animation_data.action = act
    fc = act.fcurves.new('["sim_t"]', index=0)
    fc.keyframe_points.add(T.N)
    co = np.empty(T.N * 2, dtype=np.float64)
    co[0::2] = frames
    co[1::2] = data["simt"]
    fc.keyframe_points.foreach_set("co", co.tolist())
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
    fc.update()

    # Cops
    for cop, info, sideways_until in (
        (cops[0], data["cop_a"], T.T_DROP + 0.15),
        (cops[1], data["cop_b"], T.T_DROP + 0.25),
        (cops[2], data["cop_c"], 0.0),
    ):
        locs = info["pos"]
        qu = []
        prev = None
        scales = []
        for i in range(T.N):
            if data["t"][i] < sideways_until:
                # Parked across the alley: nose points along the road's right.
                f = data["road_right"][int(np.clip(locs[i][1], 0, len(data["road_fwd"]) - 1))] if False else None
            # Use the stored forward, but yaw them 90 degrees while they block.
            fwd = info["fwd"][i]
            if data["t"][i] < sideways_until:
                # Rotate forward onto the right axis.
                fvec = np.array([-fwd[1], fwd[0], 0.0])
                if np.linalg.norm(fvec) < 1e-4:
                    fvec = np.array([1.0, 0.0, 0.0])
                fvec /= np.linalg.norm(fvec)
            else:
                fvec = fwd
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(fvec, up)
            if np.linalg.norm(right) < 1e-4:
                right = np.array([1.0, 0.0, 0.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fvec)
            q = _quat_from_axes(right, fvec, up)
            if prev is not None and (prev.x * q.x + prev.y * q.y + prev.z * q.z + prev.w * q.w) < 0:
                q.negate()
            qu.append(q)
            prev = q
            scales.append(1.0 if info["live"][i] > 0.5 else 0.001)
        cop.rotation_mode = "QUATERNION"
        bake_curves(
            cop, frames,
            {
                "location": {0: locs[:, 0], 1: locs[:, 1], 2: locs[:, 2]},
                "rotation_quaternion": {
                    0: [q.w for q in qu], 1: [q.x for q in qu],
                    2: [q.y for q in qu], 3: [q.z for q in qu],
                },
                "scale": {0: scales, 1: scales, 2: scales},
            },
        )

    # Helicopter
    hroot = heli["root"]
    hroot.rotation_mode = "QUATERNION"
    hq, prev = [], None
    for i in range(T.N):
        fvec = data["heli_fwd"][i]
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fvec, up)
        right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, fvec)
        q = _quat_from_axes(right, fvec, up)
        if prev is not None and (prev.x * q.x + prev.y * q.y + prev.z * q.z + prev.w * q.w) < 0:
            q.negate()
        hq.append(q)
        prev = q
    bake_curves(
        hroot, frames,
        {
            "location": {
                0: data["heli_pos"][:, 0], 1: data["heli_pos"][:, 1], 2: data["heli_pos"][:, 2],
            },
            "rotation_quaternion": {
                0: [q.w for q in hq], 1: [q.x for q in hq],
                2: [q.y for q in hq], 3: [q.z for q in hq],
            },
        },
    )
    # Rotor uses sim time so it slows in the bullet-time and the screen smash.
    rotor = heli["rotor"]
    spin = data["simt"] * 32.0
    bake_curves(rotor, frames, {"rotation_euler": {2: spin}})

    if rain is not None:
        bake_curves(
            rain, frames,
            {"location": {
                0: data["cam_pos"][:, 0],
                1: data["cam_pos"][:, 1],
                2: data["cam_pos"][:, 2],
            }},
        )

    # Searchlight only while the helicopter is actually in the chase.
    beam_on = [
        1.0 if (T.T_CHOPPER - 0.35) <= data["t"][i] <= (T.T_BARRICADE + 0.5) else 0.001
        for i in range(T.N)
    ]
    bake_curves(
        heli["search"], frames,
        {"scale": {0: beam_on, 1: beam_on, 2: beam_on}},
        interpolation="CONSTANT",
    )

    # Screen smash: hide the panel on the hit, keep the frame as broken scale.
    panel = screen
    hide = [1.0 if data["t"][i] < T.T_SCREEN else 0.001 for i in range(T.N)]
    bake_curves(panel, frames, {"scale": {0: [14 if h > 0.5 else 0.001 for h in hide],
                                          1: [8 if h > 0.5 else 0.001 for h in hide],
                                          2: [1 if h > 0.5 else 0.001 for h in hide]}},
                interpolation="CONSTANT")
    frame_ob = bpy.data.objects.get("ScreenFrame")
    if frame_ob is not None:
        bake_curves(frame_ob, frames, {"scale": {
            0: [14.6 if h > 0.5 else 0.001 for h in hide],
            1: [0.25 if h > 0.5 else 0.001 for h in hide],
            2: [8.5 if h > 0.5 else 0.001 for h in hide],
        }}, interpolation="CONSTANT")

    # Barricade pieces fly back toward the camera on the hit, on twos.
    bi = int(round(T.T_BARRICADE * T.FPS))
    for k, bar in enumerate(barricade):
        origin = np.array(bar.location)
        rng = random.Random(20 + k)
        vel = -data["hero_fwd"][bi] * rng.uniform(6, 14) + data["hero_right"][bi] * rng.uniform(-4, 4)
        vel = vel + np.array([0, 0, rng.uniform(2, 7)])
        locs = np.zeros((T.N, 3))
        rot = np.zeros(T.N)
        for i in range(T.N):
            if i < bi:
                locs[i] = origin
            else:
                dt = (i - bi) / T.FPS
                locs[i] = origin + vel * dt + np.array([0, 0, -0.5 * 9 * dt * dt])
                rot[i] = dt * rng.uniform(2, 6)
        bake_curves(
            bar, frames,
            {"location": {0: locs[:, 0], 1: locs[:, 1], 2: locs[:, 2]},
             "rotation_euler": {0: rot, 1: rot * 0.3, 2: bar.rotation_euler.z + rot}},
            step=2, interpolation="CONSTANT",
        )

    # Pads pop in and stay.
    for pad, t0 in ((pads[0], T.T_LAUNCH1 - 0.16), (pads[1], T.T_LAUNCH2 - 0.14)):
        sc = [1.0 if data["t"][i] >= t0 else 0.001 for i in range(T.N)]
        base = np.array(pad.scale)
        bake_curves(pad, frames, {"scale": {0: [base[0] * s for s in sc],
                                            1: [base[1] * s for s in sc],
                                            2: [base[2] * s for s in sc]}},
                    interpolation="CONSTANT")

    # Slabs curl on a wave, smooth (not on twos — this is the world, not an effect).
    for i, hinge in enumerate(slabs):
        delay = (len(slabs) - 1 - i) * 0.09
        angs = []
        for k in range(T.N):
            tt = data["t"][k]
            u = (tt - (T.T_CURL + delay)) / 1.15
            u = max(0.0, min(1.0, u))
            u = u * u * (3 - 2 * u)
            angs.append(u * 2.6)  # radians, up and over
        bake_curves(hinge, frames, {"rotation_euler": {0: angs}})

    # Ribbon emission strength: off, then the network lights up.
    ribbon = bpy.data.materials["Ribbon"]
    emit = None
    for nd in ribbon.node_tree.nodes:
        if nd.type == "EMISSION":
            emit = nd
    if emit is not None:
        nt = ribbon.node_tree
        nt.animation_data_create()
        act = bpy.data.actions.new("RibbonAct")
        nt.animation_data.action = act
        fc = act.fcurves.new(f'nodes["{emit.name}"].inputs[1].default_value')
        fc.keyframe_points.add(T.N)
        co = np.empty(T.N * 2, dtype=np.float64)
        co[0::2] = frames
        vals = []
        for i in range(T.N):
            tt = data["t"][i]
            if tt < T.T_CURL - 0.3:
                vals.append(0.0)
            else:
                u = max(0.0, min(1.0, (tt - (T.T_CURL - 0.3)) / 0.6))
                vals.append(5.5 * u)
        co[1::2] = vals
        fc.keyframe_points.foreach_set("co", co.tolist())
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.update()


def add_fireball(name, origin, frame, mats, big=True):
    rng = random.Random(frame + int(origin[0] * 10))
    colors = [mats["fire_y"], mats["fire_m"], mats["fire_c"], mats["fire_p"]]
    frames = list(range(1, T.N + 1))
    count = 8 if big else 5
    for k in range(count):
        me = primitive("uv", u=8, v=6)
        mat = colors[k % len(colors)]
        o = ob_from_mesh(f"{name}{k}", me, mat)
        o.location = Vector(origin)
        life = 32 if k < 5 else 70
        direction = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.2, 1.4)])
        direction /= np.linalg.norm(direction)
        locs = []
        sc = []
        for i in range(T.N):
            if i < frame or i > frame + life:
                locs.append(origin)
                sc.append(0.001)
            else:
                u = (i - frame) / life
                pop = math.sin(min(0.18 + u * 2.4, 1.0) * math.pi)
                rad = (4.5 if big else 2.2) * (1.3 if k < 2 else 0.8) * pop
                drift = direction * (u * (6 if k < 5 else 3))
                drift[2] += u * (2.0 if k < 5 else 5.0)
                locs.append(origin + drift)
                sc.append(max(rad, 0.001))
        locs = np.array(locs)
        bake_curves(
            o, frames,
            {"location": {0: locs[:, 0], 1: locs[:, 1], 2: locs[:, 2]},
             "scale": {0: sc, 1: sc, 2: [c * 0.72 for c in sc]}},
            step=2, interpolation="CONSTANT",
        )
    # A small local shock ring, not a full-frame flash.
    ring = ob_from_mesh(name + "ring", primitive("torus", R=1.0, r=0.05, maj=24, min=6), mats["yellow"])
    ring.location = Vector((origin[0], origin[1], origin[2]))
    sc = []
    for i in range(T.N):
        if frame <= i <= frame + 8:
            sc.append(0.4 + (i - frame) * (0.7 if big else 0.35))
        else:
            sc.append(0.001)
    bake_curves(ring, frames, {"scale": {0: sc, 1: sc, 2: [0.2 if c > 0.01 else 0.001 for c in sc]}},
                step=2, interpolation="CONSTANT")


def add_shards(origin, forward, right, up, frame, mats, face_mat):
    rng = random.Random(99)
    frames = list(range(1, T.N + 1))
    for k in range(16):
        mat = face_mat if k < 4 else [mats["cyan"], mats["magenta"], mats["yellow"], mats["ink"]][k % 4]
        o = ob_from_mesh(f"Shard{k}", primitive("plane"), mat)
        vel = (-forward * rng.uniform(1.5, 5.5)
               + right * rng.uniform(-3.5, 3.5)
               + up * rng.uniform(-0.5, 2.5))
        spin = rng.uniform(-4, 4)
        locs, rots, sc = [], [], []
        for i in range(T.N):
            if i < frame:
                locs.append(origin)
                sc.append(0.001)
                rots.append(0.0)
            else:
                dt = (i - frame) / T.FPS
                locs.append(origin + vel * dt + np.array([0, 0, -2.5 * dt * dt]))
                sc.append(rng.uniform(0.35, 1.1))
                rots.append(spin * dt)
        locs = np.array(locs)
        bake_curves(
            o, frames,
            {"location": {0: locs[:, 0], 1: locs[:, 1], 2: locs[:, 2]},
             "rotation_euler": {0: rots, 1: rots, 2: rots},
             "scale": {0: sc, 1: [c * 0.7 for c in sc], 2: sc}},
            step=2, interpolation="CONSTANT",
        )


def add_traffic(data, mats):
    cube = primitive("cube")
    colors = [mats["magenta"], mats["cyan"], mats["yellow"], mats["purple"], mats["police"]]
    frames = list(range(1, T.N + 1))
    rng = random.Random(5)
    for n in range(9):
        lane = rng.choice([-6.2, -3.2, 3.2, 6.2])
        s0 = rng.uniform(180, 900)
        speed = rng.uniform(8, 14)
        root = ob_from_mesh(f"Car{n}", cube, colors[n % len(colors)])
        root.scale = (1.7, 4.2, 1.3)
        locs = np.zeros((T.N, 3))
        qu = []
        prev = None
        for i in range(T.N):
            s = s0 + speed * data["t"][i]
            p, f, r, uu = sample_road(data, s)
            p = p + r * lane + np.array([0, 0, 0.7])
            locs[i] = p
            right = np.cross(f, np.array([0.0, 0.0, 1.0]))
            right /= max(np.linalg.norm(right), 1e-6)
            up = np.cross(right, f)
            q = _quat_from_axes(right, f, up)
            if prev is not None and (prev.x * q.x + prev.y * q.y + prev.z * q.z + prev.w * q.w) < 0:
                q.negate()
            qu.append(q)
            prev = q
        root.rotation_mode = "QUATERNION"
        bake_curves(
            root, frames,
            {"location": {0: locs[:, 0], 1: locs[:, 1], 2: locs[:, 2]},
             "rotation_quaternion": {
                 0: [q.w for q in qu], 1: [q.x for q in qu],
                 2: [q.y for q in qu], 3: [q.z for q in qu]}},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for d in list(block):
            if d.users == 0:
                block.remove(d)


def main():
    mode = parse_args()
    print("MODE", mode)
    clear_scene()
    data = T.build()
    T.report(data)
    scene = setup_render()
    mats = make_mats()
    font = bpy.data.fonts.load(FONT_PATH)
    images = []
    for fn in BILLBOARDS:
        path = os.path.join(ROOT, "assets", "billboards", fn)
        images.append(image_mat(fn, path, 1.8))
    print("building hero")
    hero = build_hero(mats)
    # Parent glass to the body so it rides along.
    for g in hero["glass"]:
        # keep world transform
        mat = g.matrix_world.copy()
        g.parent = hero["body"]
        g.matrix_world = mat
    print("building pursuit")
    cops = [build_police(mats, font) for _ in range(3)]
    heli = build_heli(mats, font)
    tanker = build_tanker(mats)
    # Park the tanker across the road.
    p, f, r, u = sample_road(data, data["s_tank"] + 2.0)
    tanker.location = Vector(p + np.array([0, 0, 0]))
    yaw = math.atan2(r[0], r[1])
    tanker.rotation_euler = (0, 0, yaw)
    print("building city")
    ribbon = build_road(data, mats)
    build_city(data, mats, font, images)
    slabs = build_slabs(data, mats)
    screen_root, screen_panel = build_screen(data, mats, images[0])
    barricade = build_barricade(data, mats)
    pads = build_pads(data, mats)
    build_tunnel(data, mats)
    rain = build_rain(scene)
    add_traffic(data, mats)
    print("baking")
    bake(data, hero, cops, heli, rain, screen_panel, barricade, pads, slabs)
    # Explosions at the causal locations.
    boom_i = int(round(T.T_BOOM * T.FPS))
    add_fireball("TankBoom", data["cop_a"]["pos"][boom_i] + np.array([0, 0, 1.5]), boom_i, mats, big=True)
    fall_i = int(round(T.T_COPS_FALL * T.FPS))
    add_fireball("CurlBoom", data["cop_c"]["pos"][fall_i] + np.array([0, 0, 1.2]), fall_i, mats, big=True)
    # Small tire-hit burst on the landing, local.
    drop_i = int(round(T.T_DROP * T.FPS))
    add_fireball("Land", data["hero_pos"][drop_i] + np.array([0, 0, 0.2]), drop_i, mats, big=False)
    scr_i = int(round(T.T_SCREEN * T.FPS))
    p, f, r, u = sample_road(data, data["s_screen"] + 1.4)
    add_shards(p + np.array([0, 0, 4.5]), f, r, u, scr_i, mats, images[0])
    ign_i = int(round(T.T_IGNITE * T.FPS))
    add_fireball("Ignite", data["hero_pos"][ign_i] + np.array([0, 0, 0.3]), ign_i, mats, big=False)

    os.makedirs(SHOT_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    if mode == "preview":
        times = [0.5, 7.6, 12.55, 13.72, 18.5, 25.2, 28.4, 32.08, 36.57, 43.5, 52.6, 57.4, 59.5, 62.2]
        for tt in times:
            frame = T.nearest_frame(tt)
            scene.frame_set(frame)
            path = os.path.join(SHOT_DIR, f"t_{frame:04d}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            print("SHOT", path)
    else:
        for frame in range(1, T.N + 1):
            path = os.path.join(FRAME_DIR, f"f_{frame:04d}.png")
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                continue
            scene.frame_set(frame)
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            if frame % 30 == 0:
                print(f"FRAME {frame}/{T.N}")


if __name__ == "__main__":
    main()
