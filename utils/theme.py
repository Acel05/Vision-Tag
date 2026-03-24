import hashlib
import colorsys

def generate_class_color(cls_name, color_cache):
    """Menghasilkan warna neon unik secara dinamis berdasarkan nama class."""
    if cls_name not in color_cache:
        hash_val = int(hashlib.md5(cls_name.encode()).hexdigest(), 16)
        hue = (hash_val % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95) 
        color_cache[cls_name] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return color_cache[cls_name]

def fade_hex_color(hex_color, step, max_steps=10):
    """Mesin kalkulasi untuk menggelapkan warna Hex menjadi hitam perlahan (Animasi)."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        factor = max(0.0, 1.0 - (step / max_steps))
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"
    except:
        return hex_color