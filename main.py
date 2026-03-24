import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
from collections import defaultdict
import json
import copy

# --- IMPORT MODULAR UTILITIES ---
from utils.geometry import get_rotated_points, get_local_coords, is_point_in_box
from utils.theme import generate_class_color, fade_hex_color

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class VisionTag_Enterprise:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionTag - Enterprise YOLOv8 OBB Annotator (AI Edition)")
        self.root.state('zoomed') 
        self.root.configure(bg="#121212")

        self.master_classes = []
        self.class_colors = {} 
        self.available_classes = []
        self.current_index = 0
        self.history_tags = {} 
        
        self.ai_model = None
        self.undo_stack = []
        self.redo_stack = []
        
        # Toast Message State
        self.toast_msg = ""
        self.toast_color = ""
        self.toast_timer = None
        
        self.source_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if not self.source_dir:
            self.root.destroy()
            return
            
        self.image_list = [f for f in os.listdir(self.source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_list:
            messagebox.showerror("Error", "No images found in the selected directory.")
            self.root.destroy()
            return

        self.load_external_config()
        self.scan_existing_labels()
        self.auto_skip_to_untagged()

        # Annotator States
        self.current_boxes = [] 
        self.action_mode = "none" 
        self.selected_box_idx = -1  
        self.clipboard_box = None   
        self.resize_axis = "none" 
        self.start_ox, self.start_oy = 0, 0
        self.interact_start_state = [] 
        
        self.temp_rect = None
        self.original_img = None
        self.zoom_level = 1.0
        self.img_x, self.img_y = 0, 0
        self.last_mouse_x, self.last_mouse_y = 0, 0
        self.current_mouse_x, self.current_mouse_y = 0, 0

        self.show_annotations = True
        self.brightness = 1.0
        self.contrast = 1.0
        self.untagged_only_var = tk.BooleanVar(value=False)

        self.setup_ui()
        self.setup_shortcuts()
        self.load_image()

    def run_auto_detect(self):
        if not HAS_YOLO:
            messagebox.showerror("Dependency Missing", "Library AI tidak ditemukan.\nInstall via terminal:\npip install ultralytics")
            return
            
        if self.ai_model is None:
            model_path = filedialog.askopenfilename(title="Select Trained YOLOv8 OBB Model (.pt)", filetypes=[("PyTorch Model", "*.pt")])
            if not model_path: return
            
            self.set_status_message("Loading AI Model...", "#ffd700", 3000)
            self.root.update()
            try:
                self.ai_model = YOLO(model_path)
            except Exception as e:
                messagebox.showerror("AI Error", f"Gagal meload model: {e}")
                return

        filepath = os.path.join(self.source_dir, self.image_list[self.current_index])
        self.set_status_message("Running Inference...", "#ffd700", 2000)
        self.root.update()
        
        try:
            results = self.ai_model(filepath)
            self.push_state() 
            
            boxes_added = 0
            import math
            for r in results:
                if r.obb is not None:
                    for i in range(len(r.obb)):
                        cx, cy, w, h, rad = r.obb.xywhr[i].tolist()
                        cls_id = int(r.obb.cls[i].item())
                        cls_name = r.names[cls_id]
                        
                        if cls_name not in self.master_classes:
                            self.master_classes.append(cls_name)
                            self._save_external_config()
                        
                        angle = math.degrees(rad)
                        self.current_boxes.append([cls_name, cx, cy, w, h, angle])
                        boxes_added += 1
                        
            self.render_tags()
            self.show_image()
            self.set_status_message(f"AI Auto-Detect Selesai: {boxes_added} Objek ditemukan!", "#00ff00", 3000)
            
        except Exception as e:
            messagebox.showerror("Inference Error", f"Terjadi kesalahan saat deteksi AI:\n{e}")

    def show_bulk_rename_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("📂 Bulk Rename Class")
        dialog.geometry("400x250")
        dialog.configure(bg="#1e1e1e")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Ubah Nama Class Secara Massal", bg="#1e1e1e", fg="#00f2ff", font=("Segoe UI", 12, "bold")).pack(pady=(15, 10))

        tk.Label(dialog, text="Pilih Class Lama:", bg="#1e1e1e", fg="#ccc", font=("Segoe UI", 9)).pack(anchor="w", padx=30)
        old_class_var = tk.StringVar()
        combo = ttk.Combobox(dialog, textvariable=old_class_var, values=self.master_classes, state="readonly", font=("Segoe UI", 10))
        combo.pack(fill="x", padx=30, pady=(0, 15))

        tk.Label(dialog, text="Ketik Nama Class Baru:", bg="#1e1e1e", fg="#ccc", font=("Segoe UI", 9)).pack(anchor="w", padx=30)
        new_class_entry = tk.Entry(dialog, font=("Segoe UI", 10), bg="#2d2d2d", fg="white", insertbackground="white", borderwidth=1)
        new_class_entry.pack(fill="x", padx=30, pady=(0, 20), ipady=4)

        def execute():
            old_name = old_class_var.get()
            new_name = new_class_entry.get().strip()
            if not old_name or not new_name:
                messagebox.showwarning("Warning", "Pilih class lama dan ketik nama baru!")
                return
            if old_name == new_name: return
            
            if messagebox.askyesno("Confirm", f"Apakah Anda yakin ingin mengubah seluruh dataset '{old_name}' menjadi '{new_name}'?"):
                self.execute_bulk_rename(old_name, new_name, dialog)

        tk.Button(dialog, text="APPLY RENAME", bg="#ff8c00", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", command=execute).pack(fill="x", padx=30, pady=5)

    def execute_bulk_rename(self, old_name, new_name, window):
        old_dir = os.path.join(self.source_dir, old_name)
        new_dir = os.path.join(self.source_dir, new_name)
        
        if os.path.exists(old_dir):
            if os.path.exists(new_dir):
                messagebox.showerror("Error", f"Folder tujuan '{new_name}' sudah ada! Penggabungan otomatis belum didukung.")
                return
            os.rename(old_dir, new_dir)
            
            cls_txt = os.path.join(new_dir, 'classes.txt')
            if os.path.exists(cls_txt):
                with open(cls_txt, 'w') as f:
                    f.write(f"{new_name}\n")
                    
            notes_json = os.path.join(new_dir, 'notes.json')
            if os.path.exists(notes_json):
                try:
                    with open(notes_json, 'r') as f:
                        data = json.load(f)
                    if 'categories' in data and len(data['categories']) > 0:
                        data['categories'][0]['name'] = new_name
                    with open(notes_json, 'w') as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    pass
                    
        if old_name in self.master_classes:
            idx = self.master_classes.index(old_name)
            self.master_classes[idx] = new_name
            self._save_external_config()
            
        for img_idx, tags in self.history_tags.items():
            if old_name in tags:
                tags.remove(old_name)
                tags.add(new_name)
                
        for box in self.current_boxes:
            if box[0] == old_name:
                box[0] = new_name
                
        self.render_tags()
        self.show_image()
        
        if self.current_index in self.history_tags:
            classes_str = ", ".join(self.history_tags[self.current_index])
            self.history_label.config(text=f"Labeled in: [{classes_str}]")
                
        window.destroy()
        messagebox.showinfo("Success", f"Berhasil merename '{old_name}' menjadi '{new_name}'!")

    def show_augmentation_dialog(self):
        if not self.current_boxes:
            messagebox.showwarning("Warning", "Gambar ini belum memiliki label.\nSilakan beri label OBB terlebih dahulu sebelum melakukan augmentasi.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("🎛️ Augment Data (Rotation)")
        dialog.geometry("380x280")
        dialog.configure(bg="#1e1e1e")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Duplikasi Data & Label (Rotate)", bg="#1e1e1e", fg="#00f2ff", font=("Segoe UI", 12, "bold")).pack(pady=(15, 5))
        tk.Label(dialog, text="Pilih satu atau lebih rotasi:", bg="#1e1e1e", fg="#aaa", font=("Segoe UI", 9)).pack()

        self.rot_90_var = tk.BooleanVar(value=False)
        self.rot_180_var = tk.BooleanVar(value=False)
        self.rot_270_var = tk.BooleanVar(value=False)
        
        frame_opts = tk.Frame(dialog, bg="#1e1e1e")
        frame_opts.pack(pady=15)
        
        tk.Checkbutton(frame_opts, text="Rotasi 90° Clockwise", variable=self.rot_90_var, bg="#1e1e1e", fg="#fff", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=2)
        tk.Checkbutton(frame_opts, text="Rotasi 180° Upside Down", variable=self.rot_180_var, bg="#1e1e1e", fg="#fff", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=2)
        tk.Checkbutton(frame_opts, text="Rotasi 270° Clockwise", variable=self.rot_270_var, bg="#1e1e1e", fg="#fff", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff", font=("Segoe UI", 10)).grid(row=2, column=0, sticky="w", pady=2)

        def execute():
            angles = []
            if self.rot_90_var.get(): angles.append(90)
            if self.rot_180_var.get(): angles.append(180)
            if self.rot_270_var.get(): angles.append(270)
            
            if not angles:
                messagebox.showwarning("Warning", "Pilih setidaknya satu sudut rotasi!")
                return
                
            for angle in angles:
                self.execute_augmentation(angle)
                
            dialog.destroy()
            self.set_status_message(f"Berhasil membuat {len(angles)} gambar augmentasi!", "#00ff00", 3000)

        tk.Button(dialog, text="GENERATE AUGMENTED COPY", bg="#008080", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", command=execute).pack(fill="x", padx=30, pady=5)

    def execute_augmentation(self, angle):
        filename = self.image_list[self.current_index]
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_rot{angle}{ext}"
        new_basename = f"{base_name}_rot{angle}"
        
        if new_filename in self.image_list:
            return 
            
        old_w, old_h = self.original_img.size
        new_img = self.original_img.rotate(-angle, expand=True)
        new_w, new_h = new_img.size

        new_boxes = []
        for box in self.current_boxes:
            cls_name, cx, cy, w, h, box_angle = box
            
            if angle == 90:
                new_cx, new_cy = old_h - cy, cx
            elif angle == 180:
                new_cx, new_cy = old_w - cx, old_h - cy
            elif angle == 270:
                new_cx, new_cy = cy, old_w - cx
                
            new_box_angle = (box_angle + angle) % 360
            new_boxes.append([cls_name, new_cx, new_cy, w, h, new_box_angle])

        new_src_path = os.path.join(self.source_dir, new_filename)
        new_img.save(new_src_path)
        self.image_list.append(new_filename)
        
        boxes_by_class = defaultdict(list)
        for box in new_boxes:
            boxes_by_class[box[0]].append(box)

        new_classes = set(boxes_by_class.keys())

        for cls_name, boxes in boxes_by_class.items():
            class_dir = os.path.join(self.source_dir, cls_name)
            img_dir = os.path.join(class_dir, "images")
            lbl_dir = os.path.join(class_dir, "labels")
            
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            
            dest_img_path = os.path.join(img_dir, new_filename)
            shutil.copy(new_src_path, dest_img_path)

            with open(os.path.join(class_dir, 'classes.txt'), 'w') as f:
                f.write(f"{cls_name}\n")
                
            dest_txt_path = os.path.join(lbl_dir, f"{new_basename}.txt")
            with open(dest_txt_path, 'w') as f:
                for box in boxes:
                    cls, cx, cy, w, h, box_ang = box
                    cls_id = 0 
                    pts = get_rotated_points(cx, cy, w, h, box_ang)
                    norm_pts = []
                    for px, py in pts:
                        nx, ny = px / new_w, py / new_h
                        norm_pts.extend([max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))])
                    
                    line = f"{cls_id} " + " ".join(f"{p:.6f}" for p in norm_pts)
                    f.write(line + "\n")
                    
        new_index = len(self.image_list) - 1
        self.history_tags[new_index] = new_classes
        self.root.update()

    def load_external_config(self):
        config_path = os.path.join(self.source_dir, "classes.txt")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.master_classes = [line.strip() for line in f if line.strip()]
        
        if not self.master_classes:
            self.master_classes = [
                'Accu', 'Aerosol', 'Alkohol', 'Bor Portable', 'Cairan Non MSDS', 
                'Cat', 'Catridge', 'Electronic Device', 'Freon', 'Handphone', 
                'Korek Api Zippo', 'Laptop', 'Lem', 'Medicine', 'Oli', 
                'Parfume', 'Pembersih', 'Petasan', 'Power Bank', 'Pupuk', 
                'Rokok', 'Serbuk', 'Tabung', 'Tanaman', 'Tinta', 'Live Animal', 'Kapasitor'
            ]
            self._save_external_config()

    def _save_external_config(self):
        config_path = os.path.join(self.source_dir, "classes.txt")
        with open(config_path, "w") as f:
            for cls in self.master_classes:
                f.write(f"{cls}\n")

    def push_state(self):
        self.undo_stack.append(copy.deepcopy(self.current_boxes))
        self.redo_stack.clear() 

    def perform_undo(self):
        if self.undo_stack:
            self.redo_stack.append(copy.deepcopy(self.current_boxes))
            self.current_boxes = self.undo_stack.pop()
            self.selected_box_idx = -1
            self.show_image()
            self.set_status_message("UNDO successful", "#ffd700")

    def perform_redo(self):
        if self.redo_stack:
            self.undo_stack.append(copy.deepcopy(self.current_boxes))
            self.current_boxes = self.redo_stack.pop()
            self.selected_box_idx = -1
            self.show_image()
            self.set_status_message("REDO successful", "#00f2ff")

    def show_dashboard(self):
        dash = tk.Toplevel(self.root)
        dash.title("📊 Dataset Analytics Dashboard")
        dash.geometry("900x700")
        dash.configure(bg="#121212")
        
        counts = defaultdict(int)
        total_duplicated_images = 0
        total_original_labeled = len(self.history_tags)
        total_images_in_dir = len(self.image_list)

        for idx, tags in self.history_tags.items():
            total_duplicated_images += len(tags) 
            for t in tags:
                counts[t] += 1
                
        header_frame = tk.Frame(dash, bg="#121212")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        def create_card(parent, title, value, color):
            card = tk.Frame(parent, bg="#1e1e1e", highlightbackground="#333", highlightthickness=1, bd=0)
            card.pack(side="left", fill="both", expand=True, padx=5)
            tk.Label(card, text=title, fg="#888", bg="#1e1e1e", font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
            tk.Label(card, text=str(value), fg=color, bg="#1e1e1e", font=("Segoe UI", 24, "bold")).pack(pady=(0, 15))
            return card
            
        create_card(header_frame, "ORIGINAL IMAGES LABELED", f"{total_original_labeled} / {total_images_in_dir}", "#00f2ff")
        create_card(header_frame, "TOTAL GENERATED IMAGES", f"{total_duplicated_images}", "#ff8c00")
        create_card(header_frame, "ACTIVE CLASSES", f"{len(counts)}", "#2ea44f")
                
        if not counts:
            tk.Label(dash, text="Belum ada data yang diberi label.", fg="#888", bg="#121212", font=("Segoe UI", 12)).pack(pady=50)
            return

        tk.Label(dash, text="CLASS DISTRIBUTION", fg="#fff", bg="#121212", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", padx=25, pady=(15, 0))
        
        c_frame = tk.Frame(dash, bg="#1e1e1e", highlightbackground="#333", highlightthickness=1)
        c_frame.pack(fill="both", expand=True, padx=25, pady=(10, 25))
        
        canvas = tk.Canvas(c_frame, bg="#1e1e1e", highlightthickness=0)
        scroll = ttk.Scrollbar(c_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#1e1e1e")
        
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", width=800)
        canvas.configure(yscrollcommand=scroll.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scroll.pack(side="right", fill="y")
        
        max_count = max(counts.values()) if counts else 1
        
        header_row = tk.Frame(inner, bg="#1e1e1e")
        header_row.pack(fill="x", pady=(0, 10))
        tk.Label(header_row, text="CLASS NAME", fg="#888", bg="#1e1e1e", font=("Segoe UI", 9, "bold"), width=22, anchor="w").pack(side="left")
        tk.Label(header_row, text="DISTRIBUTION BAR", fg="#888", bg="#1e1e1e", font=("Segoe UI", 9, "bold")).pack(side="left", padx=10)
        
        for i, cls_name in enumerate(sorted(counts.keys(), key=lambda x: counts[x], reverse=True)):
            row_bg = "#252525" if i % 2 == 0 else "#1e1e1e"
            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill="x", pady=2)
            
            tk.Label(row, text=cls_name, fg="#ccc", bg=row_bg, font=("Segoe UI", 10), width=22, anchor="w").pack(side="left", padx=(5,0), pady=8)
            
            bar_container = tk.Frame(row, bg=row_bg, width=400, height=20)
            bar_container.pack(side="left", padx=10)
            bar_container.pack_propagate(False) 
            
            bar_width = int((counts[cls_name] / max_count) * 400)
            bar_color = generate_class_color(cls_name, self.class_colors)
            
            if bar_width > 0:
                bar = tk.Label(bar_container, bg=bar_color)
                bar.place(x=0, y=2, width=bar_width, height=16)
            
            percent = (counts[cls_name] / total_duplicated_images) * 100 if total_duplicated_images > 0 else 0
            stats_text = f"{counts[cls_name]} imgs  ({percent:.1f}%)"
            tk.Label(row, text=stats_text, fg="#fff", bg=row_bg, font=("Consolas", 10)).pack(side="left", padx=10)

    def scan_existing_labels(self):
        self.history_tags.clear()
        image_basename_map = {os.path.splitext(filename)[0]: idx for idx, filename in enumerate(self.image_list)}
        
        for cls in self.master_classes:
            labels_dir = os.path.join(self.source_dir, cls, 'labels')
            if os.path.exists(labels_dir):
                for filename in os.listdir(labels_dir):
                    if filename.endswith('.txt'):
                        base_name = os.path.splitext(filename)[0]
                        if base_name in image_basename_map:
                            idx = image_basename_map[base_name]
                            if idx not in self.history_tags:
                                self.history_tags[idx] = set()
                            self.history_tags[idx].add(cls)

    def auto_skip_to_untagged(self):
        for i in range(len(self.image_list)):
            if i not in self.history_tags:
                self.current_index = i
                break
        else:
            self.current_index = len(self.image_list) - 1 if self.image_list else 0

    def setup_shortcuts(self):
        def is_not_typing(e):
            return e.widget.winfo_class() != 'Entry'

        self.root.bind("<Return>", lambda e: self.final_save() if is_not_typing(e) else None)
        self.root.bind("<Right>", lambda e: self.next_image() if is_not_typing(e) else None)
        self.root.bind("<Left>", lambda e: self.prev_image() if is_not_typing(e) else None)
        self.root.bind("<Control-f>", lambda e: self.search_entry.focus())
        
        self.root.bind("<Control-z>", lambda e: self.perform_undo() if is_not_typing(e) else None) 
        self.root.bind("<Control-y>", lambda e: self.perform_redo() if is_not_typing(e) else None) 
        
        self.root.bind("<Control-c>", lambda e: self.copy_box() if is_not_typing(e) else None)
        self.root.bind("<Control-C>", lambda e: self.copy_box() if is_not_typing(e) else None)
        self.root.bind("<Control-v>", lambda e: self.paste_box() if is_not_typing(e) else None)
        self.root.bind("<Control-V>", lambda e: self.paste_box() if is_not_typing(e) else None)
        
        self.root.bind("<v>", lambda e: self.toggle_visibility() if is_not_typing(e) else None)
        self.root.bind("<V>", lambda e: self.toggle_visibility() if is_not_typing(e) else None)
        
        self.root.bind("-", lambda e: self.adjust_enhancement('brightness', -0.2) if is_not_typing(e) else None)
        self.root.bind("=", lambda e: self.adjust_enhancement('brightness', 0.2) if is_not_typing(e) else None)
        self.root.bind("+", lambda e: self.adjust_enhancement('brightness', 0.2) if is_not_typing(e) else None)
        self.root.bind("[", lambda e: self.adjust_enhancement('contrast', -0.2) if is_not_typing(e) else None)
        self.root.bind("]", lambda e: self.adjust_enhancement('contrast', 0.2) if is_not_typing(e) else None)

    def setup_ui(self):
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self.main_container, bg="#1e1e1e", width=380)
        self.sidebar.pack(side="right", fill="y")
        self.sidebar.pack_propagate(False)

        top_btns = tk.Frame(self.sidebar, bg="#1e1e1e")
        top_btns.pack(fill="x", padx=20, pady=5)
        
        tk.Button(top_btns, text="📊 DASHBOARD", bg="#3d3d3d", fg="#00f2ff", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.show_dashboard).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(top_btns, text="ℹ️ HELP", bg="#2d2d2d", fg="#aaa", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.show_help_dialog).pack(side="right", fill="x", expand=True, padx=(4, 0))

        action_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        action_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Button(action_frame, text="🪄 AUTO-DETECT (AI)", bg="#8a2be2", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.run_auto_detect).pack(fill="x", pady=(0, 6))
        tk.Button(action_frame, text="📂 BULK RENAME CLASS", bg="#ff8c00", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.show_bulk_rename_dialog).pack(fill="x", pady=(0, 6))
        tk.Button(action_frame, text="🎛️ AUGMENT DATA (ROTATE)", bg="#008080", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.show_augmentation_dialog).pack(fill="x")

        tk.Label(self.sidebar, text="ACTIVE / SELECTED CLASS", bg="#1e1e1e", fg="#00f2ff", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(10, 2))
        self.queue_list = tk.Listbox(self.sidebar, height=1, bg="#121212", fg="#00f2ff", borderwidth=0, highlightthickness=1, font=("Segoe UI", 12, "bold"))
        self.queue_list.pack(fill="x", padx=20, pady=5)
        
        tk.Label(self.sidebar, text="STATUS", bg="#1e1e1e", fg="#ffd700", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(0, 2))
        self.history_container = tk.Frame(self.sidebar, bg="#1e1e1e")
        self.history_container.pack(fill="x", padx=20, pady=5)
        self.history_label = tk.Label(self.history_container, text="", bg="#1e1e1e", fg="#ffd700", font=("Segoe UI", 9, "italic"), justify="left", anchor="nw")
        self.history_label.pack(fill="both", expand=True)

        tk.Label(self.sidebar, text="SEARCH & SELECT CLASS", bg="#1e1e1e", fg="#888", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(0, 5))
        search_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        search_frame.pack(fill="x", padx=20, pady=5)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda n, i, m: self.render_tags())
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var, bg="#2d2d2d", fg="white", insertbackground="white", borderwidth=0, font=("Segoe UI", 11))
        self.search_entry.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.search_entry.bind("<Return>", self.add_to_master)
        tk.Button(search_frame, text="ADD", bg="#0078d4", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), width=6, command=self.add_to_master).pack(side="right", fill="y")

        btn_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        btn_frame.pack(fill="x", side="bottom", padx=20, pady=5)
        self.btn_clear = tk.Button(btn_frame, text="CLEAR ALL LABELS", bg="#ff4444", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.clear_all_boxes)
        self.btn_clear.pack(fill="x", pady=(0, 6))
        self.btn_save = tk.Button(btn_frame, text="CONFIRM & EXPORT (ENTER)", bg="#2ea44f", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), pady=4, command=self.final_save)
        self.btn_save.pack(fill="x")

        self.list_container = tk.Frame(self.sidebar, bg="#1e1e1e")
        self.list_container.pack(fill="both", expand=True, padx=20, pady=5)
        self.tag_canvas = tk.Canvas(self.list_container, bg="#1e1e1e", highlightthickness=0)
        self.tag_scroll = ttk.Scrollbar(self.list_container, orient="vertical", command=self.tag_canvas.yview)
        self.tag_inner = tk.Frame(self.tag_canvas, bg="#1e1e1e")
        self.tag_canvas.create_window((0, 0), window=self.tag_inner, anchor="nw", width=320)
        self.tag_canvas.configure(yscrollcommand=self.tag_scroll.set)
        
        self.tag_canvas.bind("<Enter>", lambda e: self.tag_canvas.bind_all("<MouseWheel>", self._on_tag_scroll))
        self.tag_canvas.bind("<Leave>", lambda e: self.tag_canvas.unbind_all("<MouseWheel>"))
        self.tag_canvas.pack(side="left", fill="both", expand=True)
        self.tag_scroll.pack(side="right", fill="y")
        self.tag_inner.bind("<Configure>", lambda e: self.tag_canvas.configure(scrollregion=self.tag_canvas.bbox("all")))

        self.viewer_frame = tk.Frame(self.main_container, bg="#121212")
        self.viewer_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(self.viewer_frame, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        
        self.vline = self.canvas.create_line(0, 0, 0, 0, fill="#00f2ff", dash=(4, 4), tags="crosshair")
        self.hline = self.canvas.create_line(0, 0, 0, 0, fill="#00f2ff", dash=(4, 4), tags="crosshair")
        
        self.canvas.bind("<Motion>", self.update_crosshair)
        self.canvas.bind("<MouseWheel>", self.handle_scroll)
        self.canvas.bind("<ButtonPress-3>", self.start_drag)
        self.canvas.bind("<B3-Motion>", self.do_drag)
        
        self.canvas.bind("<ButtonPress-1>", self.start_interaction)
        self.canvas.bind("<B1-Motion>", self.do_interaction)
        self.canvas.bind("<ButtonRelease-1>", self.end_interaction)

        self.nav_info_bar = tk.Frame(self.viewer_frame, bg="#1e1e1e", height=50)
        self.nav_info_bar.pack(fill="x", pady=(5,0))

        tk.Button(self.nav_info_bar, text="<< PREV", bg="#333", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), command=self.prev_image, width=12).pack(side="left", padx=20, pady=10)

        btn_skip = tk.Button(self.nav_info_bar, text="SKIP >>", bg="#333", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), command=self.next_image, width=12)
        btn_skip.pack(side="right", padx=(5, 20))
        
        self.status_text = tk.Label(self.nav_info_bar, text="", bg="#1e1e1e", fg="#888", font=("Consolas", 10))
        self.status_text.pack(side="right", padx=10)

        info_mid = tk.Frame(self.nav_info_bar, bg="#1e1e1e")
        info_mid.pack(side="left", fill="x", expand=True)

        tk.Label(info_mid, text="Go to / Search:", bg="#1e1e1e", fg="#888", font=("Segoe UI", 9)).pack(side="left")
        self.jump_var = tk.StringVar()
        self.jump_entry = tk.Entry(info_mid, textvariable=self.jump_var, width=15, bg="#2d2d2d", fg="#00f2ff", borderwidth=0, justify="center", font=("Consolas", 11, "bold"))
        self.jump_entry.pack(side="left", padx=10, ipady=3)
        self.jump_entry.bind("<Return>", self.jump_to_image)

        tk.Checkbutton(info_mid, text="Show Untagged Only", variable=self.untagged_only_var, bg="#1e1e1e", fg="#aaa", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff", cursor="hand2").pack(side="left", padx=(10, 0))

        self.save_indicator = tk.Label(info_mid, text="", bg="#1e1e1e", font=("Segoe UI", 12))
        self.save_indicator.pack(side="left", padx=2)
        
        self.filename_label = tk.Label(info_mid, text="", bg="#1e1e1e", fg="#00f2ff", font=("Segoe UI", 9, "bold"), anchor="w")
        self.filename_label.pack(side="left", fill="x", expand=True)

    def show_help_dialog(self):
        help_text = (
            "Mouse Controls:\n"
            "• L-Click empty space : Draw Box\n"
            "• L-Click a box : Select Box (Can Change Class or Copy)\n"
            "• L-Click edge of box : Resize Box\n"
            "• R-Click & Drag : Pan Image\n"
            "• Scroll : Zoom In/Out\n"
            "• Ctrl + Scroll : Rotate Box\n"
            "• Shift + L-Click : Delete specific Box\n\n"
            "Keyboard Shortcuts:\n"
            "• 'V' : Toggle Hide/Show Annotations\n"
            "• '- / =' : Decrease / Increase Brightness\n"
            "• '[ / ]' : Decrease / Increase Contrast\n"
            "• Ctrl + C : Copy Selected Box\n"
            "• Ctrl + V : Paste Box at Cursor location\n"
            "• Enter : Confirm & Export\n"
            "• Arrow Left / Right : Prev / Next Image\n"
            "• Ctrl + F : Focus Search Bar\n"
            "• Ctrl + Z : Undo | Ctrl + Y : Redo"
        )
        messagebox.showinfo("VisionTag Controls & Shortcuts", help_text)

    def set_status_message(self, text, color="#00ff00", duration=2000):
        self.status_text.config(text=f"{self.current_index+1}/{len(self.image_list)}", fg="#888")
        
        self.toast_msg = text
        self.toast_color = color
        self.toast_step = 0  
        self.show_image() 
        
        if hasattr(self, 'toast_timer') and self.toast_timer:
            self.root.after_cancel(self.toast_timer)
        self.toast_timer = self.root.after(duration, self.fade_out_toast)

    def fade_out_toast(self):
        self.toast_step += 1
        if self.toast_step > 10: 
            self.toast_msg = ""
            self.show_image()
        else:
            self.show_image()
            self.toast_timer = self.root.after(40, self.fade_out_toast)

    def clear_toast_message(self):
        self.toast_msg = ""
        self.show_image()

    def copy_box(self):
        if self.selected_box_idx != -1 and self.selected_box_idx < len(self.current_boxes):
            self.clipboard_box = list(self.current_boxes[self.selected_box_idx])
            self.set_status_message(f"COPIED: {self.clipboard_box[0]}", "#00ff00")
        else:
            self.set_status_message("Select a box first to copy!", "#ff4444")

    def paste_box(self):
        if self.clipboard_box:
            self.push_state()
            new_box = list(self.clipboard_box)
            ox, oy = self.canvas_to_img(self.current_mouse_x, self.current_mouse_y)
            new_box[1] = ox
            new_box[2] = oy
            
            self.current_boxes.append(new_box)
            self.selected_box_idx = len(self.current_boxes) - 1 
            self.show_image()
            self.set_status_message("PASTED!", "#00ff00")
        else:
            self.set_status_message("Clipboard empty!", "#ff4444")

    def toggle_visibility(self):
        self.show_annotations = not self.show_annotations
        self.show_image()

    def adjust_enhancement(self, target, amount):
        if target == 'brightness':
            self.brightness = max(0.2, min(3.0, self.brightness + amount))
        elif target == 'contrast':
            self.contrast = max(0.2, min(3.0, self.contrast + amount))
        self.show_image()

    def _on_tag_scroll(self, event):
        self.tag_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def render_tags(self):
        for widget in self.tag_inner.winfo_children(): widget.destroy()
        search_query = self.search_var.get().lower()
        filtered = [c for c in self.master_classes if search_query in c.lower()]
        
        for cls in sorted(filtered):
            row = tk.Frame(self.tag_inner, bg="#1e1e1e")
            row.pack(fill="x", pady=3)
            
            btn = tk.Button(row, text=f"  {cls}", anchor="w", bg="#2d2d2d", fg="#ccc", relief="flat", font=("Segoe UI", 8), command=lambda c=cls: self.set_active_class(c))
            btn.pack(side="left", fill="both", expand=True)
            
            del_btn = tk.Button(row, text="🗑️", bg="#2d2d2d", fg="#ff4444", relief="flat", font=("Segoe UI", 9), anchor="center", command=lambda c=cls: self.delete_master_class(c))
            del_btn.pack(side="right", fill="y", padx=(6, 0))
            
            bg_hover = generate_class_color(cls, self.class_colors)
            btn.bind("<Enter>", lambda e, b=btn, color=bg_hover: b.configure(bg=color, fg="#000"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#2d2d2d", fg="#ccc"))
            
            del_btn.bind("<Enter>", lambda e, b=del_btn: b.configure(bg="#ff4444", fg="white"))
            del_btn.bind("<Leave>", lambda e, b=del_btn: b.configure(bg="#2d2d2d", fg="#ff4444"))

    def delete_master_class(self, cls_name):
        confirm = messagebox.askyesno("Delete Class", f"Hapus '{cls_name}' dari daftar Master Class?")
        if confirm:
            if cls_name in self.master_classes:
                self.master_classes.remove(cls_name)
                self._save_external_config() 
                self.render_tags() 
                
                if self.get_active_class() == cls_name:
                    self.queue_list.delete(0, tk.END)
                    self.selected_box_idx = -1
                    self.show_image()
            self.set_status_message(f"Class '{cls_name}' deleted!", "#ff4444")

    def set_active_class(self, tag):
        self.queue_list.delete(0, tk.END)
        self.queue_list.insert(tk.END, tag)
        
        if self.selected_box_idx != -1 and self.selected_box_idx < len(self.current_boxes):
            self.push_state()
            self.current_boxes[self.selected_box_idx][0] = tag
            self.show_image()
            self.set_status_message(f"Label updated to {tag}", "#00ff00")
            
        self.canvas.focus_set() 

    def get_active_class(self):
        items = self.queue_list.get(0, tk.END)
        return items[0] if items else None

    def add_to_master(self, event=None):
        val = self.search_var.get().strip()
        if val and val not in self.master_classes:
            self.master_classes.append(val)
            self._save_external_config()
            self.search_var.set("")
            self.render_tags()
            
        self.canvas.focus_set()
        return "break"

    def jump_to_image(self, event=None):
        query = self.jump_var.get().strip().lower()
        if not query: 
            self.canvas.focus_set()
            return "break"

        if query.isdigit():
            target = int(query) - 1
            if 0 <= target < len(self.image_list):
                self.current_index = target
                self.load_image()
        else:
            for i, filename in enumerate(self.image_list):
                if query in filename.lower():
                    self.current_index = i
                    self.load_image()
                    self.canvas.focus_set()
                    return "break"
            messagebox.showinfo("Not Found", f"No image found containing: '{query}'")
        
        self.canvas.focus_set()
        return "break"

    def canvas_to_img(self, cx, cy):
        w_disp = self.original_img.size[0] * self.zoom_level
        h_disp = self.original_img.size[1] * self.zoom_level
        ox = (cx - (self.img_x - w_disp / 2)) / self.zoom_level
        oy = (cy - (self.img_y - h_disp / 2)) / self.zoom_level
        return ox, oy

    def img_to_canvas(self, ox, oy):
        w_disp = self.original_img.size[0] * self.zoom_level
        h_disp = self.original_img.size[1] * self.zoom_level
        cx = (ox * self.zoom_level) + (self.img_x - w_disp / 2)
        cy = (oy * self.zoom_level) + (self.img_y - h_disp / 2)
        return cx, cy

    def update_crosshair(self, event):
        self.current_mouse_x = event.x
        self.current_mouse_y = event.y
        if self.show_annotations:
            self.canvas.coords(self.vline, event.x, 0, event.x, self.canvas.winfo_height())
            self.canvas.coords(self.hline, 0, event.y, self.canvas.winfo_width(), event.y)
            self.canvas.tag_raise("crosshair")

    def handle_scroll(self, event):
        ctrl_pressed = (event.state & 0x0004) != 0
        delta = event.delta if not (hasattr(event, 'num') and event.num in (4, 5)) else (1 if event.num == 4 else -1)

        if ctrl_pressed:
            ox, oy = self.canvas_to_img(event.x, event.y)
            for i in reversed(range(len(self.current_boxes))):
                if is_point_in_box(ox, oy, self.current_boxes[i]):
                    self.push_state() 
                    
                    rotated_box = self.current_boxes.pop(i)
                    self.current_boxes.append(rotated_box)
                    self.selected_box_idx = len(self.current_boxes) - 1
                    
                    step = 5 if delta > 0 else -5 
                    self.current_boxes[-1][5] = (self.current_boxes[-1][5] + step) % 360
                    self.show_image()
                    break
        else:
            self.zoom_level *= (1.1 if delta > 0 else 0.9)
            self.show_image()

    def start_drag(self, event):
        self.canvas.focus_set()
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def do_drag(self, event):
        self.img_x += event.x - self.last_mouse_x
        self.img_y += event.y - self.last_mouse_y
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.show_image()

    def start_interaction(self, event):
        self.canvas.focus_set()
        ox, oy = self.canvas_to_img(event.x, event.y)

        if event.state & 0x0001: 
            for i in reversed(range(len(self.current_boxes))):
                if is_point_in_box(ox, oy, self.current_boxes[i]):
                    self.push_state()
                    self.current_boxes.pop(i)
                    if self.selected_box_idx == i:
                        self.selected_box_idx = -1
                    self.show_image()
                    break
            self.action_mode = "none"
            return

        for i in reversed(range(len(self.current_boxes))):
            box = self.current_boxes[i]
            if is_point_in_box(ox, oy, box):
                self.push_state()
                
                clicked_box = self.current_boxes.pop(i)
                self.current_boxes.append(clicked_box)
                self.selected_box_idx = len(self.current_boxes) - 1  
                
                cls, cx, cy, w, h, angle = clicked_box
                lx, ly = get_local_coords(ox, oy, cx, cy, angle)
                
                self.queue_list.delete(0, tk.END)
                self.queue_list.insert(tk.END, cls)

                margin = 15 / self.zoom_level 
                is_edge_x = abs(lx) > (w/2 - margin)
                is_edge_y = abs(ly) > (h/2 - margin)

                self.start_ox, self.start_oy = ox, oy
                self.interact_start_state = [cx, cy, w, h, angle]

                if is_edge_x and is_edge_y:
                    self.action_mode = "resize"
                    self.resize_axis = "both"
                elif is_edge_x:
                    self.action_mode = "resize"
                    self.resize_axis = "width"
                elif is_edge_y:
                    self.action_mode = "resize"
                    self.resize_axis = "height"
                else:
                    self.action_mode = "move"
                
                self.show_image() 
                return

        self.selected_box_idx = -1
        self.show_image()
        
        if not self.get_active_class():
            messagebox.showwarning("No Class", "Please select an Active Class from the right panel first.")
            self.action_mode = "none"
            return
            
        self.action_mode = "draw"
        self.start_x, self.start_y = event.x, event.y
        self.temp_rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline=generate_class_color(self.get_active_class(), self.class_colors), width=2, tags="temp_box")

    def do_interaction(self, event):
        ox, oy = self.canvas_to_img(event.x, event.y)

        if self.action_mode == "move":
            dx = ox - self.start_ox
            dy = oy - self.start_oy
            orig_cx, orig_cy, _, _, _ = self.interact_start_state
            
            self.current_boxes[self.selected_box_idx][1] = orig_cx + dx
            self.current_boxes[self.selected_box_idx][2] = orig_cy + dy
            self.show_image()

        elif self.action_mode == "resize":
            orig_cx, orig_cy, orig_w, orig_h, orig_angle = self.interact_start_state
            lx, ly = get_local_coords(ox, oy, orig_cx, orig_cy, orig_angle)
            
            if self.resize_axis in ("both", "width"):
                self.current_boxes[self.selected_box_idx][3] = max(10, abs(lx) * 2)
            if self.resize_axis in ("both", "height"):
                self.current_boxes[self.selected_box_idx][4] = max(10, abs(ly) * 2)
            self.show_image()

        elif self.action_mode == "draw" and self.temp_rect:
            self.canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def end_interaction(self, event):
        if self.action_mode == "draw":
            self.canvas.delete("temp_box")
            ox1, oy1 = self.canvas_to_img(self.start_x, self.start_y)
            ox2, oy2 = self.canvas_to_img(event.x, event.y)
            ox1, ox2 = min(ox1, ox2), max(ox1, ox2)
            oy1, oy2 = min(oy1, oy2), max(oy1, oy2)
            ow = ox2 - ox1
            oh = oy2 - oy1
            
            if ow > 5 and oh > 5:
                self.push_state()
                active_class = self.get_active_class()
                ocx, ocy = ox1 + ow / 2, oy1 + oh / 2
                self.current_boxes.append([active_class, ocx, ocy, ow, oh, 0.0])
                self.selected_box_idx = len(self.current_boxes) - 1
            self.show_image()
            
        self.action_mode = "none"

    def clear_all_boxes(self):
        self.push_state()
        self.current_boxes.clear()
        self.selected_box_idx = -1
        self.show_image()

    def draw_all_boxes(self):
        if not self.show_annotations: return
        
        for i, box in enumerate(self.current_boxes):
            cls_name, ocx, ocy, ow, oh, angle = box
            pts = get_rotated_points(ocx, ocy, ow, oh, angle)
            
            c_pts = []
            for ox, oy in pts:
                cx, cy = self.img_to_canvas(ox, oy)
                c_pts.extend([cx, cy])
            
            base_color = generate_class_color(cls_name, self.class_colors)
            outline_color = "#000000" if i == self.selected_box_idx else base_color
            outline_width = 3 if i == self.selected_box_idx else 2
            
            self.canvas.create_polygon(c_pts, outline=outline_color, fill="", width=outline_width, tags="box")
            
            text_x, text_y = self.img_to_canvas(pts[0][0], pts[0][1])
            font_size = max(7, int(11 * self.zoom_level))
            
            self.canvas.create_text(text_x + 1, text_y - 2 + 1, text=cls_name, fill="#000000", anchor="sw", font=("Segoe UI", font_size, "bold"), tags="box")
            self.canvas.create_text(text_x, text_y - 2, text=cls_name, fill=outline_color, anchor="sw", font=("Segoe UI", font_size, "bold"), tags="box")

    def load_image(self):
        if not self.image_list: return
        if 0 <= self.current_index < len(self.image_list):
            filename = self.image_list[self.current_index]
            filepath = os.path.join(self.source_dir, filename)
            
            self.original_img = Image.open(filepath)
            
            self.brightness = 1.0
            self.contrast = 1.0

            self.root.update()
            
            self.current_boxes = [] 
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.selected_box_idx = -1
            
            img_w, img_h = self.original_img.size

            base_name = os.path.splitext(filename)[0]
            for cls in self.master_classes:
                label_path = os.path.join(self.source_dir, cls, 'labels', f"{base_name}.txt")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 9:
                                nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4 = map(float, parts[1:9])
                                ox1, oy1 = nx1 * img_w, ny1 * img_h
                                ox2, oy2 = nx2 * img_w, ny2 * img_h
                                ox3, oy3 = nx3 * img_w, ny3 * img_h
                                ox4, oy4 = nx4 * img_w, ny4 * img_h
                                
                                ocx = (ox1 + ox2 + ox3 + ox4) / 4
                                ocy = (oy1 + oy2 + oy3 + oy4) / 4
                                ow = math.hypot(ox2 - ox1, oy2 - oy1)
                                oh = math.hypot(ox3 - ox2, oy3 - oy2)
                                angle = math.degrees(math.atan2(oy2 - oy1, ox2 - ox1))
                                
                                self.current_boxes.append([cls, ocx, ocy, ow, oh, angle])

            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            ratio = min(cw/img_w, ch/img_h)
            self.zoom_level, self.img_x, self.img_y = ratio * 0.95, cw // 2, ch // 2
            
            self.current_mouse_x = cw // 2
            self.current_mouse_y = ch // 2
            
            self.available_classes = list(self.master_classes)
            self.jump_var.set(str(self.current_index + 1))
            
            display_name = filename
            if len(display_name) > 40:
                display_name = display_name[:20] + "..." + display_name[-15:]
                
            self.filename_label.config(text=f"FILE: {display_name}")
            self.status_text.config(text=f"{self.current_index+1}/{len(self.image_list)}")
            
            if self.current_index in self.history_tags:
                classes_str = ", ".join(self.history_tags[self.current_index])
                self.save_indicator.config(text="✅", fg="#2ea44f")
                self.history_label.config(text=f"Labeled in: [{classes_str}]")
            else:
                self.save_indicator.config(text="⭕", fg="#555")
                self.history_label.config(text="No labels yet.")
            
            self.show_image()
            self.render_tags()
            self.canvas.focus_set()

    def show_image(self):
        if not self.original_img: return
        w, h = self.original_img.size
        resized = self.original_img.resize((max(1, int(w * self.zoom_level)), max(1, int(h * self.zoom_level))), Image.Resampling.LANCZOS)
        
        if self.brightness != 1.0:
            resized = ImageEnhance.Brightness(resized).enhance(self.brightness)
        if self.contrast != 1.0:
            resized = ImageEnhance.Contrast(resized).enhance(self.contrast)

        self.tk_img = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("img", "box", "hud")
        self.canvas.create_image(self.img_x, self.img_y, image=self.tk_img, anchor="center", tags="img")
        
        self.draw_all_boxes()
        
        # --- DESAIN HUD MODERN & RAPI ---
        pad_x, pad_y = 12, 10
        hud_font = ("Consolas", 9, "bold") 
        
        # 1. HUD KIRI ATAS
        hud_text = (
            f"ZOOM       : {int(self.zoom_level*100)}%\n"
            f"BRIGHTNESS : {self.brightness:.1f}x\n"
            f"CONTRAST   : {self.contrast:.1f}x"
        )
        
        t_hud = self.canvas.create_text(25, 20, text=hud_text, fill="#ffffff", anchor="nw", font=hud_font, tags="hud")
        bbox_hud = self.canvas.bbox(t_hud)
        if bbox_hud:
            bg_hud = self.canvas.create_rectangle(bbox_hud[0]-pad_x, bbox_hud[1]-pad_y, bbox_hud[2]+pad_x, bbox_hud[3]+pad_y, fill="#1c1c1c", outline="#333333", width=1, tags="hud")
            acc_hud = self.canvas.create_line(bbox_hud[0]-pad_x, bbox_hud[1]-pad_y, bbox_hud[0]-pad_x, bbox_hud[3]+pad_y, fill="#00f2ff", width=4, tags="hud")
            self.canvas.tag_lower(acc_hud, t_hud)
            self.canvas.tag_lower(bg_hud, acc_hud)
            
        # 2. HUD KANAN ATAS (Toast)
        if hasattr(self, 'toast_msg') and self.toast_msg:
            cw = self.canvas.winfo_width()
            if cw <= 1: cw = self.root.winfo_screenwidth() - 400 
            
            step = getattr(self, 'toast_step', 0)
            
            text_color = fade_hex_color("#ffffff", step)
            bg_color = fade_hex_color("#1c1c1c", step)
            outline_color = fade_hex_color("#333333", step)
            acc_color = fade_hex_color(self.toast_color, step)
            
            t_toast = self.canvas.create_text(cw - 25, 20, text=f"🔔  {self.toast_msg}", fill=text_color, anchor="ne", font=hud_font, tags="hud")
            bbox_toast = self.canvas.bbox(t_toast)
            if bbox_toast:
                bg_toast = self.canvas.create_rectangle(bbox_toast[0]-pad_x, bbox_toast[1]-pad_y, bbox_toast[2]+pad_x, bbox_toast[3]+pad_y, fill=bg_color, outline=outline_color, width=1, tags="hud")
                acc_toast = self.canvas.create_line(bbox_toast[2]+pad_x, bbox_toast[1]-pad_y, bbox_toast[2]+pad_x, bbox_toast[3]+pad_y, fill=acc_color, width=4, tags="hud")
                self.canvas.tag_lower(acc_toast, t_toast)
                self.canvas.tag_lower(bg_toast, acc_toast)
        
        if self.show_annotations:
            self.canvas.itemconfig("crosshair", state="normal")
            self.canvas.tag_raise("crosshair")
        else:
            self.canvas.itemconfig("crosshair", state="hidden")
            
        self.canvas.tag_raise("hud")

    def next_image(self):
        if self.untagged_only_var.get():
            for i in range(self.current_index + 1, len(self.image_list)):
                if i not in self.history_tags:
                    self.current_index = i
                    self.load_image()
                    return
            messagebox.showinfo("Complete", "Semua gambar sudah di-label!")
        else:
            if self.current_index < len(self.image_list) - 1:
                self.current_index += 1
                self.load_image()

    def prev_image(self):
        if self.untagged_only_var.get():
            for i in range(self.current_index - 1, -1, -1):
                if i not in self.history_tags:
                    self.current_index = i
                    self.load_image()
                    return
        else:
            if self.current_index > 0:
                self.current_index -= 1
                self.load_image()

    def _remove_files_for_class(self, cls_name, filename, base_name):
        txt_path = os.path.join(self.source_dir, cls_name, 'labels', f"{base_name}.txt")
        img_path = os.path.join(self.source_dir, cls_name, 'images', filename)
        if os.path.exists(txt_path): os.remove(txt_path)
        if os.path.exists(img_path): os.remove(img_path)

    def final_save(self):
        filename = self.image_list[self.current_index]
        base_name = os.path.splitext(filename)[0]
        src_filepath = os.path.join(self.source_dir, filename)
        img_w, img_h = self.original_img.size
        
        old_classes = self.history_tags.get(self.current_index, set())

        if not self.current_boxes:
            if old_classes:
                if messagebox.askyesno("Remove All Labels", "Hapus semua label untuk gambar ini? File pada folder akan ikut terhapus."):
                    for cls in old_classes:
                        self._remove_files_for_class(cls, filename, base_name)
                    if self.current_index in self.history_tags:
                        del self.history_tags[self.current_index]
                    self.next_image()
                return
            else:
                self.next_image()
                return

        boxes_by_class = defaultdict(list)
        for box in self.current_boxes:
            boxes_by_class[box[0]].append(box)

        new_classes = set(boxes_by_class.keys())

        for cls_to_remove in old_classes - new_classes:
            self._remove_files_for_class(cls_to_remove, filename, base_name)

        for cls_name, boxes in boxes_by_class.items():
            class_dir = os.path.join(self.source_dir, cls_name)
            img_dir = os.path.join(class_dir, "images")
            lbl_dir = os.path.join(class_dir, "labels")
            
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            
            dest_img_path = os.path.join(img_dir, filename)
            shutil.copy(src_filepath, dest_img_path)

            with open(os.path.join(class_dir, 'classes.txt'), 'w') as f:
                f.write(f"{cls_name}\n")

            notes_data = {"categories": [{"id": 0, "name": cls_name}], "info": {"year": 2026, "version": "1.0", "contributor": "Label Studio"}}
            with open(os.path.join(class_dir, 'notes.json'), 'w') as f:
                json.dump(notes_data, f, indent=2)
                
            dest_txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
            with open(dest_txt_path, 'w') as f:
                for box in boxes:
                    cls, cx, cy, w, h, angle = box
                    cls_id = 0 
                    pts = get_rotated_points(cx, cy, w, h, angle)
                    norm_pts = []
                    for px, py in pts:
                        nx, ny = px / img_w, py / img_h
                        norm_pts.extend([max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))])
                    
                    line = f"{cls_id} " + " ".join(f"{p:.6f}" for p in norm_pts)
                    f.write(line + "\n")
                
        self.history_tags[self.current_index] = new_classes
        self.next_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionTag_Enterprise(root)
    root.after(500, app.load_image)
    root.mainloop()