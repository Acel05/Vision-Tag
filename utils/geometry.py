import math

def get_local_coords(px, py, cx, cy, angle):
    """Translate pixel click ke koordinat rotasi box."""
    tx, ty = px - cx, py - cy
    rad = math.radians(-angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    lx = tx * cos_a - ty * sin_a
    ly = tx * sin_a + ty * cos_a
    return lx, ly

def get_rotated_points(cx, cy, w, h, angle):
    """Mendapatkan 4 titik sudut dari OBB yang dirotasi."""
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = []
    for rx, ry in corners:
        nx = cx + rx * cos_a - ry * sin_a
        ny = cy + rx * sin_a + ry * cos_a
        pts.append((nx, ny))
    return pts

def is_point_in_box(px, py, box):
    """Mengecek apakah koordinat klik mouse berada di dalam OBB."""
    cls, cx, cy, w, h, angle = box
    lx, ly = get_local_coords(px, py, cx, cy, angle)
    return (-w/2 <= lx <= w/2) and (-h/2 <= ly <= h/2)