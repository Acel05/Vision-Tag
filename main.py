import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import defaultdict
import json
import copy
import math
import threading
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageOps

from utils.geometry import get_rotated_points, get_local_coords, is_point_in_box
from utils.theme import generate_class_color, fade_hex_color

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VisionTag_Enterprise:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionTag - Enterprise YOLOv8 OBB Annotator (AI Edition)")
        self.root.state('zoomed') 
        self.root.configure(bg="#121212")
        
        # Core State
        self.master_classes = []
        self.class_colors = {}
        self.available_classes = []
        self.current_index = 0
        self.history_tags = {}
        self.ai_model = None
        
        # History & UI State
        self.undo_stack = []
        self.redo_stack = []
        self.toast_msg = ""
        self.toast_color = ""
        self.toast_timer = None
        self.toast_step = 0
        
        self.source_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if not self.source_dir: 
            self.root.destroy()
            return
            
        self.image_list = [f for f in os.listdir(self.source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not self.image_list: 
            messagebox.showerror("Error", "No images found.")
            self.root.destroy()
            return

        self.raw_basenames = {os.path.splitext(f)[0] for f in self.image_list}
        self.load_external_config()
        self.scan_existing_labels()
        self.auto_skip_to_untagged()

        # Canvas & Interaction State
        self.current_boxes = []
        self.action_mode = "none"
        self.selected_box_idx = -1  
        self.clipboard_box = None
        self.resize_axis = "none" 
        self.start_ox = 0
        self.start_oy = 0
        self.interact_start_state = []
        self.temp_rect = None
        self.original_img = None
        self.zoom_level = 1.0
        self.img_x = 0
        self.img_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.current_mouse_x = 0
        self.current_mouse_y = 0
        self.show_annotations = True
        self.brightness = 1.0
        self.contrast = 1.0
        self.enhance_timer = None
        
        # Variables
        self.untagged_only_var = tk.BooleanVar(value=False)
        self.filter_class_var = tk.StringVar(value="All Classes")
        self.search_var = tk.StringVar()
        self.jump_var = tk.StringVar()

        self.setup_ui()
        self.setup_shortcuts()
        self.load_image()

    def _is_matching_file(self, filename, base_name, extension=".txt"):
        target = f"{base_name}{extension}"
        if filename == target: 
            return True
            
        if filename.endswith(target):
            full_label_basename = filename[:-len(extension)]
            if full_label_basename in getattr(self, 'raw_basenames', set()) and full_label_basename != base_name: 
                return False
            prefix_len = len(filename) - len(target)
            if filename[prefix_len - 1] in ['-', '_']: 
                return True
                
        return False

    def scan_existing_labels(self):
        self.history_tags.clear()
        # Create a map of base names to their index
        img_map = {os.path.splitext(f)[0]: idx for idx, f in enumerate(self.image_list)}
        
        for cls in self.master_classes:
            lbl_dir = os.path.join(self.source_dir, cls, 'labels')
            if os.path.exists(lbl_dir):
                for f in os.listdir(lbl_dir):
                    if f.endswith('.txt'):
                        base = os.path.splitext(f)[0]
                        
                        # 1. Direct O(1) lookup first
                        match_idx = img_map.get(base)
                        
                        # 2. If not found, strip augmentation suffixes and check again (O(1) lookup)
                        if match_idx is None:
                            pure_base = self._get_pure_basename(base)
                            match_idx = img_map.get(pure_base)
                                    
                        # 3. If a match is found, record it
                        if match_idx is not None:
                            if match_idx not in self.history_tags: 
                                self.history_tags[match_idx] = set()
                            self.history_tags[match_idx].add(cls)

    def _get_pure_basename(self, filename):
        base = os.path.splitext(filename)[0]
        for suf in ['_rot90', '_rot180', '_rot270', '_hflip', '_clahe', '_invert', '_bright', '_noise']:
            if base.endswith(suf): 
                return base[:-len(suf)]
        return base

    def load_image(self):
        if not self.image_list: 
            return
            
        if 0 <= self.current_index < len(self.image_list):
            filename = self.image_list[self.current_index]
            filepath = os.path.join(self.source_dir, filename)
            
            self.canvas.delete("box", "img", "hud")
            self.current_boxes.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.selected_box_idx = -1
            self.root.update_idletasks() 
            
            try:
                self.original_img = Image.open(filepath)
                self.brightness = 1.0
                self.contrast = 1.0
                img_w, img_h = self.original_img.size
                base_name = os.path.splitext(filename)[0]
                
                for cls in self.master_classes:
                    lbl_dir = os.path.join(self.source_dir, cls, 'labels')
                    if os.path.exists(lbl_dir):
                        for lbl_file in os.listdir(lbl_dir):
                            if self._is_matching_file(lbl_file, base_name, ".txt"):
                                try:
                                    with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
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
                                except Exception as e: 
                                    print(f"Error reading {lbl_file}: {e}")
            except Exception as e: 
                messagebox.showerror("Error", f"Gagal memuat gambar: {e}")
                return
                
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw <= 1: 
                cw = self.root.winfo_screenwidth() - 400
                ch = self.root.winfo_screenheight() - 200
                
            self.zoom_level = min(cw / img_w, ch / img_h) * 0.95
            self.img_x = cw // 2
            self.img_y = ch // 2
            self.available_classes = list(self.master_classes)
            self.jump_var.set(str(self.current_index + 1))
            
            display_name = filename if len(filename) <= 40 else filename[:20] + "..." + filename[-15:]
            self.filename_label.config(text=f"FILE: {display_name}")
            self.status_text.config(text=f"{self.current_index + 1}/{len(self.image_list)}")
            
            if self.current_index in self.history_tags:
                self.save_indicator.config(text="✅", fg="#2ea44f")
                self.history_label.config(text=f"Labeled in: {list(self.history_tags[self.current_index])}")
            else:
                self.save_indicator.config(text="⭕", fg="#555")
                self.history_label.config(text="No labels yet.")
            
            # INITIALIZE REPROCESS FLAG
            self.needs_reprocess = True
            self.show_image()
            self.render_tags()
            self.canvas.focus_set()

    def _remove_files_for_class(self, cls_name, filename, base_name):
        lbl_dir = os.path.join(self.source_dir, cls_name, 'labels')
        img_dir = os.path.join(self.source_dir, cls_name, 'images')
        
        if os.path.exists(lbl_dir):
            for f in os.listdir(lbl_dir):
                if self._is_matching_file(f, base_name, ".txt"):
                    os.remove(os.path.join(lbl_dir, f))
                    img_f = os.path.splitext(f)[0] + os.path.splitext(filename)[1]
                    if os.path.exists(os.path.join(img_dir, img_f)): 
                        os.remove(os.path.join(img_dir, img_f))

    def _cleanup_duplicates(self, pure_base, action):
        expected_prefix = f"{pure_base}_{action}"
        duplicates = []
        
        for img_name in self.image_list:
            img_base = os.path.splitext(img_name)[0]
            if img_base.startswith(expected_prefix):
                duplicates.append(img_name)
                
        if len(duplicates) > 1:
            duplicates.sort(key=len)
            
            for dup in duplicates[1:]:
                dup_base = os.path.splitext(dup)[0]
                img_path = os.path.join(self.source_dir, dup)
                
                if os.path.exists(img_path): 
                    os.remove(img_path)
                
                dup_idx = self.image_list.index(dup)
                for cls in self.history_tags.get(dup_idx, set()):
                    self._remove_files_for_class(cls, dup, dup_base)
                
                if dup_idx in self.history_tags:
                    del self.history_tags[dup_idx]
                self.image_list.remove(dup)
                
            self.scan_existing_labels()
            return len(duplicates) - 1
            
        return 0

    def delete_current_image(self):
        if not self.image_list: 
            return
            
        filename = self.image_list[self.current_index]
        if messagebox.askyesno("Delete Image", f"Hapus file '{filename}' secara permanen beserta labelnya?"):
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(self.source_dir, filename)
            
            if os.path.exists(img_path): 
                os.remove(img_path)
                
            for cls in self.history_tags.get(self.current_index, set()): 
                self._remove_files_for_class(cls, filename, base_name)
                
            if self.current_index in self.history_tags: 
                del self.history_tags[self.current_index]
                
            del self.image_list[self.current_index]
            self.scan_existing_labels() 
            
            if self.current_index >= len(self.image_list): 
                self.current_index = max(0, len(self.image_list) - 1)
                
            self.load_image()
            self.set_status_message(f"Deleted: {filename}", "#ff4444")

    def final_save(self):
        filename = self.image_list[self.current_index]
        base_name, ext = os.path.splitext(filename)
        src_filepath = os.path.join(self.source_dir, filename)
        img_w, img_h = self.original_img.size
        old_classes = self.history_tags.get(self.current_index, set())
        
        if not self.current_boxes:
            if old_classes and messagebox.askyesno("Remove All Labels", "Hapus semua label untuk gambar ini?"):
                for cls in old_classes: 
                    self._remove_files_for_class(cls, filename, base_name)
                if self.current_index in self.history_tags: 
                    del self.history_tags[self.current_index]
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
            
            target_basename = base_name
            if os.path.exists(lbl_dir):
                for existing in os.listdir(lbl_dir):
                    if self._is_matching_file(existing, base_name, ".txt"): 
                        target_basename = os.path.splitext(existing)[0]
                        break
                        
            self._remove_files_for_class(cls_name, filename, base_name)
            shutil.copy2(src_filepath, os.path.join(img_dir, target_basename + ext))
            
            # Only write static configs if the directory is brand new
            notes_path = os.path.join(class_dir, 'notes.json')
            if not os.path.exists(notes_path):
                with open(os.path.join(class_dir, 'classes.txt'), 'w') as f: 
                    f.write(f"{cls_name}\n")
                with open(notes_path, 'w') as f: 
                    json.dump({"categories": [{"id": 0, "name": cls_name}], "info": {"year": 2026, "version": "1.0", "contributor": "Label Studio"}}, f, indent=2)
                
            with open(os.path.join(lbl_dir, f"{target_basename}.txt"), 'w') as f:
                for box in boxes:
                    pts = get_rotated_points(box[1], box[2], box[3], box[4], box[5])
                    norm_pts = []
                    for px, py in pts: 
                        norm_pts.extend([max(0.0, min(1.0, px / img_w)), max(0.0, min(1.0, py / img_h))])
                    f.write("0 " + " ".join(f"{p:.6f}" for p in norm_pts) + "\n")
                    
        self.history_tags[self.current_index] = new_classes
        self.set_status_message(f"SAVED: {target_basename}", "#00ff00")
        self.next_image()

    def run_auto_detect(self):
        if not HAS_YOLO: 
            messagebox.showerror("Dependency Missing", "Install via terminal:\npip install ultralytics")
            return
            
        # 1. If model isn't loaded yet, ask for path and load it in a background thread
        if self.ai_model is None:
            model_path = filedialog.askopenfilename(title="Select Trained YOLOv8 OBB Model (.pt)", filetypes=[("PyTorch Model", "*.pt")])
            if not model_path: 
                return
                
            self.set_status_message("Mounting PyTorch Model... Please wait.", "#ffd700", 10000)
            self.root.update()
            
            def load_model_task():
                try: 
                    # Load heavy model in background
                    loaded_model = YOLO(model_path)
                    # Safely pass back to main thread to start inference
                    self.root.after(0, self._on_model_loaded, loaded_model)
                except Exception as e: 
                    self.root.after(0, lambda: messagebox.showerror("AI Error", f"Gagal meload model: {e}"))
            
            threading.Thread(target=load_model_task, daemon=True).start()
            return # Exit function early; _on_model_loaded will trigger inference

        # 2. If model is ALREADY loaded, just jump straight to inference
        self._execute_inference_thread()

    def _on_model_loaded(self, loaded_model):
        """Callback when PyTorch finishes loading the model into memory"""
        self.ai_model = loaded_model
        self.set_status_message("Model Loaded! Starting inference...", "#00f2ff", 2000)
        self._execute_inference_thread()

    def _execute_inference_thread(self):
        """The actual threaded inference logic you already built"""
        filepath = os.path.join(self.source_dir, self.image_list[self.current_index])
        self.set_status_message("AI Running Inference... Please wait.", "#8a2be2", 10000) 
        
        target_idx = self.current_index 
        
        def inference_task(expected_idx):
            try:
                results = self.ai_model(filepath)
                if self.current_index == expected_idx:
                    self.root.after(0, self._process_ai_results, results)
            except Exception as e: 
                self.root.after(0, lambda: messagebox.showerror("Inference Error", f"Error:\n{e}"))
                
        threading.Thread(target=inference_task, args=(target_idx,), daemon=True).start()

    def _process_ai_results(self, results):
        self.push_state()
        boxes_added = 0
        
        for r in results:
            if r.obb is not None:
                for i in range(len(r.obb)):
                    cx, cy, w, h, rad = r.obb.xywhr[i].tolist()
                    cls_name = r.names[int(r.obb.cls[i].item())]
                    
                    if cls_name not in self.master_classes:
                        self.master_classes.append(cls_name)
                        self._save_external_config()
                        self._update_filter_combo_values()
                        
                    self.current_boxes.append([cls_name, cx, cy, w, h, math.degrees(rad)])
                    boxes_added += 1
                    
        self.render_tags()
        self.show_image()
        self.set_status_message(f"AI Auto-Detect Selesai: {boxes_added} Objek ditemukan!", "#00ff00", 3000)

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
        ttk.Combobox(dialog, textvariable=old_class_var, values=self.master_classes, state="readonly", font=("Segoe UI", 10)).pack(fill="x", padx=30, pady=(0, 15))
        
        tk.Label(dialog, text="Ketik Nama Class Baru:", bg="#1e1e1e", fg="#ccc", font=("Segoe UI", 9)).pack(anchor="w", padx=30)
        new_class_entry = tk.Entry(dialog, font=("Segoe UI", 10), bg="#2d2d2d", fg="white", insertbackground="white", borderwidth=1)
        new_class_entry.pack(fill="x", padx=30, pady=(0, 20), ipady=4)
        
        def execute():
            old_name = old_class_var.get()
            new_name = new_class_entry.get().strip()
            if old_name and new_name and old_name != new_name:
                if messagebox.askyesno("Confirm", f"Ubah '{old_name}' menjadi '{new_name}'?"): 
                    self.execute_bulk_rename(old_name, new_name, dialog)
                    
        tk.Button(dialog, text="APPLY RENAME", bg="#ff8c00", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", command=execute).pack(fill="x", padx=30, pady=5)

    def execute_bulk_rename(self, old_name, new_name, window):
        old_dir = os.path.join(self.source_dir, old_name)
        new_dir = os.path.join(self.source_dir, new_name)
        
        if os.path.exists(old_dir):
            if os.path.exists(new_dir): 
                messagebox.showerror("Error", f"Folder tujuan '{new_name}' sudah ada!")
                return
            os.rename(old_dir, new_dir)
            with open(os.path.join(new_dir, 'classes.txt'), 'w') as f: 
                f.write(f"{new_name}\n")
            
            notes_json = os.path.join(new_dir, 'notes.json')
            if os.path.exists(notes_json):
                try:
                    with open(notes_json, 'r') as f: 
                        data = json.load(f)
                    if data.get('categories'): 
                        data['categories'][0]['name'] = new_name
                    with open(notes_json, 'w') as f: 
                        json.dump(data, f, indent=2)
                except: 
                    pass
                    
        if old_name in self.master_classes:
            self.master_classes[self.master_classes.index(old_name)] = new_name
            self._save_external_config()
            self._update_filter_combo_values()
            
        for tags in self.history_tags.values():
            if old_name in tags: 
                tags.remove(old_name)
                tags.add(new_name)
                
        for box in self.current_boxes:
            if box[0] == old_name: 
                box[0] = new_name
                
        self.render_tags()
        self.show_image()
        
        if self.current_index in self.history_tags: 
            self.history_label.config(text=f"Labeled in: [{', '.join(self.history_tags[self.current_index])}]")
            
        window.destroy()
        messagebox.showinfo("Success", f"Berhasil merename '{old_name}' menjadi '{new_name}'!")

    def show_augmentation_dialog(self):
        if not self.history_tags and not self.current_boxes: 
            messagebox.showwarning("Warning", "Belum ada gambar yang dilabeli di dataset ini!")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("🎛️ Augment Data (X-Ray Suite)")
        dialog.geometry("500x480")
        dialog.configure(bg="#1e1e1e")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Generate Varian Data Baru", bg="#1e1e1e", fg="#00f2ff", font=("Segoe UI", 12, "bold")).pack(pady=(15, 5))
        tk.Label(dialog, text="Target Scope:", bg="#1e1e1e", fg="#fff", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=30, pady=(5, 0))
        
        self.aug_scope_var = tk.StringVar(value="current")
        tk.Radiobutton(dialog, text="Current Image Only", variable=self.aug_scope_var, value="current", bg="#1e1e1e", fg="#aaa", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff").pack(anchor="w", padx=40)
        tk.Radiobutton(dialog, text="Bulk: All Classes (Massal)", variable=self.aug_scope_var, value="all", bg="#1e1e1e", fg="#ff4444", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#ff4444").pack(anchor="w", padx=40)
        tk.Radiobutton(dialog, text="Bulk: Specific Classes", variable=self.aug_scope_var, value="specific", bg="#1e1e1e", fg="#00ff00", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00ff00").pack(anchor="w", padx=40)

        list_frame = tk.Frame(dialog, bg="#1e1e1e")
        list_frame.pack(fill="x", padx=50, pady=5)
        
        self.class_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, bg="#2d2d2d", fg="#fff", height=4, font=("Segoe UI", 9), highlightthickness=1)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_listbox.yview)
        self.class_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.class_listbox.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for cls in sorted(self.master_classes): 
            self.class_listbox.insert(tk.END, cls)
        
        def on_scope_change(*args):
            if self.aug_scope_var.get() == "specific": 
                self.class_listbox.configure(state=tk.NORMAL)
            else: 
                self.class_listbox.selection_clear(0, tk.END)
                self.class_listbox.configure(state=tk.DISABLED)
                
        self.aug_scope_var.trace("w", on_scope_change)
        on_scope_change()

        tk.Label(dialog, text="Pilih Teknik:", bg="#1e1e1e", fg="#fff", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=30, pady=(10, 0))
        self.rot_90_var, self.rot_180_var, self.rot_270_var, self.hflip_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.clahe_var, self.invert_var, self.bright_var, self.noise_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()

        main_frame = tk.Frame(dialog, bg="#1e1e1e")
        main_frame.pack(fill="x", padx=30, pady=5)
        
        col1 = tk.Frame(main_frame, bg="#1e1e1e")
        col1.pack(side="left", fill="y", expand=True)
        
        col2 = tk.Frame(main_frame, bg="#1e1e1e")
        col2.pack(side="right", fill="y", expand=True)

        for t, v in [("Rotasi 90°", self.rot_90_var), ("Rotasi 180°", self.rot_180_var), ("Rotasi 270°", self.rot_270_var), ("Horizontal Flip", self.hflip_var)]:
            tk.Checkbutton(col1, text=t, variable=v, bg="#1e1e1e", fg="#aaa", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff").pack(anchor="w")

        for t, v, c in [("CLAHE (Perjelas)", self.clahe_var, "#ff4444"), ("Invert (Negatif)", self.invert_var, "#d32f2f"), ("Brightness (+30%)", self.bright_var, "#ffd700"), ("Gaussian Noise", self.noise_var, "#ff8c00")]:
            tk.Checkbutton(col2, text=t, variable=v, bg="#1e1e1e", fg=c, selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground=c).pack(anchor="w")

        def execute():
            actions = []
            if self.rot_90_var.get(): actions.append('rot90')
            if self.rot_180_var.get(): actions.append('rot180')
            if self.rot_270_var.get(): actions.append('rot270')
            if self.hflip_var.get(): actions.append('hflip')
            if self.clahe_var.get(): actions.append('clahe')
            if self.invert_var.get(): actions.append('invert')
            if self.bright_var.get(): actions.append('bright')
            if self.noise_var.get(): actions.append('noise')
            
            if not actions: 
                messagebox.showwarning("Warning", "Pilih teknik augmentasi!")
                return
            
            scope = self.aug_scope_var.get()
            targets = []
            
            if scope == "current": 
                targets.append(self.current_index)
            else:
                sel_classes = set(self.class_listbox.get(i) for i in self.class_listbox.curselection()) if scope == "specific" else set()
                if scope == "specific" and not sel_classes: 
                    messagebox.showwarning("Warning", "Pilih minimal satu class dari list!")
                    return
                
                for idx, tags in self.history_tags.items():
                    filename = self.image_list[idx]
                    if self._get_pure_basename(filename) != os.path.splitext(filename)[0]: 
                        continue 
                    if scope == "specific" and not tags.intersection(sel_classes): 
                        continue
                    pure_base = self._get_pure_basename(filename)
                    if any(img != filename and self._get_pure_basename(img) == pure_base for img in self.image_list): 
                        continue
                    targets.append(idx)
            
            if not targets:
                messagebox.showinfo("Info", "Tidak ada gambar target yang valid atau gambar sudah pernah diaugmentasi sebelumnya.")
                return
                
            target_filenames = [self.image_list[idx] for idx in targets]
            
            # --- OPTIMIZATION: Threading the Heavy Loop ---
            self.set_status_message(f"Augmenting {len(target_filenames)} images in background...", "#ffd700", 5000)
            dialog.destroy()
            
            def background_augmentation():
                success_count = 0
                deleted_dups_count = 0
                
                for t_filename in target_filenames:
                    if t_filename not in self.image_list: 
                        continue 
                    pure_base = self._get_pure_basename(t_filename)
                    
                    for act in actions:
                        deleted_dups_count += self._cleanup_duplicates(pure_base, act)
                        current_idx = self.image_list.index(t_filename)
                        if self.execute_augmentation(act, current_idx): 
                            success_count += 1
                            
                # Safely update the UI from the background thread once completely finished
                self.root.after(0, lambda: self.set_status_message(f"Selesai! {success_count} augment baru. {deleted_dups_count} duplikat dihapus.", "#00ff00", 6000))
                self.root.after(0, self.show_image)
                
            threading.Thread(target=background_augmentation, daemon=True).start()
            
        tk.Button(dialog, text="GENERATE AUGMENTED COPY", bg="#008080", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", command=execute).pack(fill="x", padx=30, pady=25, ipady=5)

    def execute_augmentation(self, action, target_idx):
        filename = self.image_list[target_idx]
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{action}{ext}"
        new_basename = f"{base_name}_{action}"
        
        if new_filename in self.image_list: 
            return False
            
        if target_idx == self.current_index:
            target_img = self.original_img
            target_boxes = self.current_boxes
        else:
            filepath = os.path.join(self.source_dir, filename)
            if not os.path.exists(filepath): 
                return False
                
            target_img = Image.open(filepath)
            target_boxes = []
            img_w, img_h = target_img.size
            
            for cls in self.master_classes:
                lbl_dir = os.path.join(self.source_dir, cls, 'labels')
                if os.path.exists(lbl_dir):
                    for lbl_file in os.listdir(lbl_dir):
                        if self._is_matching_file(lbl_file, base_name, ".txt"):
                            with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 9:
                                        nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4 = map(float, parts[1:9])
                                        ox1, oy1 = nx1 * img_w, ny1 * img_h
                                        ox2, oy2 = nx2 * img_w, ny2 * img_h
                                        ox3, oy3 = nx3 * img_w, ny3 * img_h
                                        ox4, oy4 = nx4 * img_w, ny4 * img_h
                                        
                                        cx = (ox1 + ox2 + ox3 + ox4) / 4
                                        cy = (oy1 + oy2 + oy3 + oy4) / 4
                                        ow = math.hypot(ox2 - ox1, oy2 - oy1)
                                        oh = math.hypot(ox3 - ox2, oy3 - oy2)
                                        angle = math.degrees(math.atan2(oy2 - oy1, ox2 - ox1))
                                        
                                        target_boxes.append([cls, cx, cy, ow, oh, angle])
                                        
        if not target_boxes: 
            return False 
            
        old_w, old_h = target_img.size
        new_img = None
        new_boxes = []

        if action.startswith('rot'):
            angle = int(action.replace('rot', ''))
            new_img = target_img.rotate(-angle, expand=True)
            for box in target_boxes:
                if angle == 90:
                    new_cx, new_cy = (old_h - box[2], box[1])
                elif angle == 180:
                    new_cx, new_cy = (old_w - box[1], old_h - box[2])
                else:
                    new_cx, new_cy = (box[2], old_w - box[1])
                    
                new_boxes.append([box[0], new_cx, new_cy, box[3], box[4], (box[5] + angle) % 360])
                
        elif action == 'hflip':
            new_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
            for box in target_boxes: 
                new_boxes.append([box[0], old_w - box[1], box[2], box[3], box[4], (180 - box[5]) % 360])
                
        elif action == 'clahe':
            if not HAS_CV2: 
                messagebox.showerror("Error", "Library 'opencv-python' dibutuhkan untuk CLAHE.")
                return False
            img_cv = cv2.cvtColor(np.array(target_img.convert('RGB')), cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img_cv[:,:,0] = clahe.apply(img_cv[:,:,0])
            new_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB))
            new_boxes = [list(box) for box in target_boxes]
            
        elif action == 'invert':
            if target_img.mode == 'RGBA':
                r, g, b, a = target_img.split()
                r2, g2, b2 = ImageOps.invert(Image.merge('RGB', (r, g, b))).split()
                new_img = Image.merge('RGBA', (r2, g2, b2, a))
            else: 
                new_img = ImageOps.invert(target_img.convert('RGB'))
            new_boxes = [list(box) for box in target_boxes]
            
        elif action == 'bright':
            new_img = ImageEnhance.Brightness(target_img).enhance(1.3)
            new_boxes = [list(box) for box in target_boxes]
            
        elif action == 'noise':
            img_arr = np.array(target_img)
            noise = np.random.randint(-30, 30, img_arr.shape, dtype='int16')
            noisy_img = np.clip(img_arr.astype('int16') + noise, 0, 255).astype('uint8')
            new_img = Image.fromarray(noisy_img)
            new_boxes = [list(box) for box in target_boxes]

        if not new_img: 
            return False
            
        new_w, new_h = new_img.size
        new_img.save(os.path.join(self.source_dir, new_filename))
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
            
            shutil.copy2(os.path.join(self.source_dir, new_filename), os.path.join(img_dir, new_filename))
            
            # Only write static configs if the directory is brand new
            notes_path = os.path.join(class_dir, 'notes.json')
            if not os.path.exists(notes_path):
                with open(os.path.join(class_dir, 'classes.txt'), 'w') as f: 
                    f.write(f"{cls_name}\n")
                with open(notes_path, 'w') as f: 
                    json.dump({"categories": [{"id": 0, "name": cls_name}], "info": {"year": 2026, "version": "1.0", "contributor": "Label Studio"}}, f, indent=2)
                
            with open(os.path.join(lbl_dir, f"{new_basename}.txt"), 'w') as f:
                for box in boxes:
                    pts = get_rotated_points(box[1], box[2], box[3], box[4], box[5])
                    norm_pts = []
                    for px, py in pts: 
                        norm_pts.extend([max(0.0, min(1.0, px / new_w)), max(0.0, min(1.0, py / new_h))])
                    f.write(f"0 " + " ".join(f"{p:.6f}" for p in norm_pts) + "\n")

        self.history_tags[len(self.image_list) - 1] = new_classes
        if target_idx == self.current_index: 
            self.root.update()
            
        return True

    def load_external_config(self):
        config_path = os.path.join(self.source_dir, "classes.txt")
        if os.path.exists(config_path):
            with open(config_path, "r") as f: 
                self.master_classes = [line.strip() for line in f if line.strip()]
                
        if not self.master_classes:
            self.master_classes = [
                'Accu', 'Aerosol', 'Alkohol', 'Bor Portable', 'Cairan Non MSDS', 
                'Cat', 'Catridge', 'Electronic Device', 'Freon', 'Handphone', 
                'Korek Api Zippo', 'Laptop', 'Lem', 'Medicine', 'Oli', 'Parfume', 
                'Pembersih', 'Petasan', 'Power Bank', 'Pupuk', 'Rokok', 'Serbuk', 
                'Tabung', 'Tanaman', 'Tinta', 'Live Animal', 'Kapasitor'
            ]
            self._save_external_config()

    def _save_external_config(self):
        with open(os.path.join(self.source_dir, "classes.txt"), "w") as f:
            for cls in self.master_classes: 
                f.write(f"{cls}\n")

    def _update_filter_combo_values(self):
        if hasattr(self, 'filter_combo'):
            vals = ["All Classes"] + sorted(self.master_classes)
            self.filter_combo['values'] = vals
            if self.filter_class_var.get() not in vals: 
                self.filter_class_var.set("All Classes")

    def _is_image_valid_for_filter(self, idx):
        if self.untagged_only_var.get() and idx in self.history_tags: 
            return False
            
        filter_cls = getattr(self, 'filter_class_var', None)
        if filter_cls and filter_cls.get() != "All Classes":
            if idx not in self.history_tags or filter_cls.get() not in self.history_tags[idx]: 
                return False
                
        return True

    def _on_untagged_toggle(self):
        if self.untagged_only_var.get(): 
            self.filter_class_var.set("All Classes")
        self.apply_filter()

    def apply_filter(self, event=None):
        if self.filter_class_var.get() != "All Classes": 
            self.untagged_only_var.set(False)
            
        if self._is_image_valid_for_filter(self.current_index): 
            return
            
        for i in range(self.current_index + 1, len(self.image_list)):
            if self._is_image_valid_for_filter(i): 
                self.current_index = i
                self.load_image()
                return
                
        for i in range(0, self.current_index):
            if self._is_image_valid_for_filter(i): 
                self.current_index = i
                self.load_image()
                return
                
        messagebox.showinfo("Not Found", "Tidak ada gambar yang cocok dengan filter yang dipilih.")
        self.filter_class_var.set("All Classes")
        self.untagged_only_var.set(False)

    def next_image(self):
        for i in range(self.current_index + 1, len(self.image_list)):
            if self._is_image_valid_for_filter(i): 
                self.current_index = i
                self.load_image()
                return
        messagebox.showinfo("Akhir Galeri", "Sudah mencapai gambar terakhir untuk kriteria ini.")

    def prev_image(self):
        for i in range(self.current_index - 1, -1, -1):
            if self._is_image_valid_for_filter(i): 
                self.current_index = i
                self.load_image()
                return
        messagebox.showinfo("Awal Galeri", "Sudah mencapai gambar pertama untuk kriteria ini.")

    def auto_skip_to_untagged(self):
        for i in range(len(self.image_list)):
            if i not in self.history_tags: 
                self.current_index = i
                break
        else: 
            self.current_index = len(self.image_list) - 1 if self.image_list else 0

    def show_dashboard(self):
        dash = tk.Toplevel(self.root)
        dash.title("📊 Dataset Analytics Dashboard")
        dash.geometry("950x700")
        dash.configure(bg="#121212")
        
        counts_total = defaultdict(int)
        counts_orig = defaultdict(int)
        counts_aug = defaultdict(int)
        
        total_original_labeled = 0
        total_augmented_images = 0
        total_original_in_dir = 0
        
        for img in self.image_list:
            if self._get_pure_basename(img) == os.path.splitext(img)[0]: 
                total_original_in_dir += 1
                
        for idx, tags in self.history_tags.items():
            if idx >= len(self.image_list): 
                continue
            filename = self.image_list[idx]
            is_orig = (self._get_pure_basename(filename) == os.path.splitext(filename)[0])
            
            if is_orig: 
                total_original_labeled += 1
            else: 
                total_augmented_images += 1
                
            for t in tags: 
                counts_total[t] += 1
                if is_orig:
                    counts_orig[t] += 1
                else:
                    counts_aug[t] += 1
                    
        header_frame = tk.Frame(dash, bg="#121212")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        def create_card(parent, title, value, color):
            card = tk.Frame(parent, bg="#1e1e1e", highlightbackground="#333", highlightthickness=1, bd=0)
            card.pack(side="left", fill="both", expand=True, padx=5)
            tk.Label(card, text=title, fg="#888", bg="#1e1e1e", font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
            tk.Label(card, text=str(value), fg=color, bg="#1e1e1e", font=("Segoe UI", 24, "bold")).pack(pady=(0, 15))
            
        create_card(header_frame, "ORIGINAL IMAGES LABELED", f"{total_original_labeled} / {total_original_in_dir}", "#00f2ff")
        create_card(header_frame, "TOTAL GENERATED IMAGES", f"{total_augmented_images}", "#ff8c00")
        create_card(header_frame, "ACTIVE CLASSES", f"{len(counts_total)}", "#2ea44f")
        
        if not counts_total: 
            tk.Label(dash, text="Belum ada data yang diberi label.", fg="#888", bg="#121212", font=("Segoe UI", 12)).pack(pady=50)
            return
            
        tk.Label(dash, text="CLASS DISTRIBUTION (Total = Original + Augmented)", fg="#fff", bg="#121212", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", padx=25, pady=(15, 0))
        
        c_frame = tk.Frame(dash, bg="#1e1e1e", highlightbackground="#333", highlightthickness=1)
        c_frame.pack(fill="both", expand=True, padx=25, pady=(10, 25))
        
        canvas = tk.Canvas(c_frame, bg="#1e1e1e", highlightthickness=0)
        scroll = ttk.Scrollbar(c_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#1e1e1e")
        
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", width=880)
        canvas.configure(yscrollcommand=scroll.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scroll.pack(side="right", fill="y")
        
        max_count = max(counts_total.values()) if counts_total else 1
        
        header_row = tk.Frame(inner, bg="#1e1e1e")
        header_row.pack(fill="x", pady=(0, 10))
        tk.Label(header_row, text="CLASS NAME", fg="#888", bg="#1e1e1e", font=("Segoe UI", 9, "bold"), width=20, anchor="w").pack(side="left")
        tk.Label(header_row, text="DISTRIBUTION BAR", fg="#888", bg="#1e1e1e", font=("Segoe UI", 9, "bold")).pack(side="left", padx=10)
        
        for i, cls_name in enumerate(sorted(counts_total.keys(), key=lambda x: counts_total[x], reverse=True)):
            row_bg = "#252525" if i % 2 == 0 else "#1e1e1e"
            row = tk.Frame(inner, bg=row_bg)
            row.pack(fill="x", pady=2)
            
            tk.Label(row, text=cls_name, fg="#ccc", bg=row_bg, font=("Segoe UI", 10), width=20, anchor="w").pack(side="left", padx=(5,0), pady=8)
            
            bar_container = tk.Frame(row, bg=row_bg, width=300, height=20)
            bar_container.pack(side="left", padx=10)
            bar_container.pack_propagate(False) 
            
            bar_width = int((counts_total[cls_name] / max_count) * 300)
            bar_color = generate_class_color(cls_name, self.class_colors)
            
            if bar_width > 0: 
                bar = tk.Label(bar_container, bg=bar_color)
                bar.place(x=0, y=2, width=bar_width, height=16)
                
            stats_text = f"Total: {counts_total[cls_name]} imgs  (Orig: {counts_orig[cls_name]} | Aug: {counts_aug[cls_name]})"
            tk.Label(row, text=stats_text, fg="#fff", bg=row_bg, font=("Consolas", 10)).pack(side="left", padx=15)

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
        self.status_text.config(text=f"{self.current_index + 1}/{len(self.image_list)}", fg="#888")
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

    def push_state(self):
        self.undo_stack.append([list(box) for box in self.current_boxes])
        self.redo_stack.clear()

    def perform_undo(self):
        if self.undo_stack:
            self.redo_stack.append([list(box) for box in self.current_boxes])
            self.current_boxes = self.undo_stack.pop()
            self.selected_box_idx = -1
            self.show_image()
            self.set_status_message("UNDO successful", "#ffd700")

    def perform_redo(self):
        if self.redo_stack:
            self.undo_stack.append([list(box) for box in self.current_boxes])
            self.current_boxes = self.redo_stack.pop()
            self.selected_box_idx = -1
            self.show_image()
            self.set_status_message("REDO successful", "#00f2ff")

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
            new_box[1], new_box[2] = ox, oy
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
            
        # Update the HUD text instantly without recalculating the heavy PIL image
        self.set_status_message(f"Adjusting {target}...", "#ffd700", 500)
        
        # Debounce: Cancel the previous timer if the user is still actively pressing the key
        if self.enhance_timer:
            self.root.after_cancel(self.enhance_timer)
            
        # Schedule the heavy processing to run 150ms AFTER they stop pressing the key
        self.enhance_timer = self.root.after(150, self._apply_enhancement)
    
    def _apply_enhancement(self):
        # Tell the show_image function it is allowed to recalculate the pixels
        self.needs_reprocess = True
        self.show_image()

    def _on_tag_scroll(self, event): 
        self.tag_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def render_tags(self):
        for widget in self.tag_inner.winfo_children(): 
            widget.destroy()
            
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
        if messagebox.askyesno("Delete Class", f"Hapus '{cls_name}' dari daftar Master Class?"):
            if cls_name in self.master_classes:
                self.master_classes.remove(cls_name)
                self._save_external_config()
                self.render_tags()
                self._update_filter_combo_values()
                
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
            self._update_filter_combo_values()
            
        self.canvas.focus_set()
        return "break"

    def jump_to_image(self, event=None):
        query = self.jump_var.get().strip().lower()
        if not query: 
            self.canvas.focus_set()
            return "break"
            
        if query.isdigit() and 0 <= int(query) - 1 < len(self.image_list):
            self.current_index = int(query) - 1
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
        
        new_cx = (cx - (self.img_x - w_disp / 2)) / self.zoom_level
        new_cy = (cy - (self.img_y - h_disp / 2)) / self.zoom_level
        return new_cx, new_cy

    def img_to_canvas(self, ox, oy):
        w_disp = self.original_img.size[0] * self.zoom_level
        h_disp = self.original_img.size[1] * self.zoom_level
        
        new_ox = (ox * self.zoom_level) + (self.img_x - w_disp / 2)
        new_oy = (oy * self.zoom_level) + (self.img_y - h_disp / 2)
        return new_ox, new_oy

    def update_crosshair(self, event):
        self.current_mouse_x = event.x
        self.current_mouse_y = event.y
        if self.show_annotations:
            self.canvas.coords(self.vline, event.x, 0, event.x, self.canvas.winfo_height())
            self.canvas.coords(self.hline, 0, event.y, self.canvas.winfo_width(), event.y)

    def quick_select_class(self, idx):
        # Sort classes alphabetically to match how they are displayed in the UI list
        sorted_classes = sorted(self.master_classes)
        if idx < len(sorted_classes):
            self.set_active_class(sorted_classes[idx])

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
                    self.current_boxes[-1][5] = (self.current_boxes[-1][5] + (5 if delta > 0 else -5)) % 360
                    self.show_image()
                    break
        else:
            self.zoom_level *= (1.1 if delta > 0 else 0.9)
            # FLAG: Recalculate pixels because zoom changed
            self.needs_reprocess = True
            self.show_image()

    def start_drag(self, event): 
        self.canvas.focus_set()
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
    def do_drag(self, event): 
        # Calculate movement delta
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        # Update internal tracking
        self.img_x += dx
        self.img_y += dy
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        # INSTANT PANNING: Move canvas elements natively without redrawing PIL images
        self.canvas.move("img", dx, dy)
        self.canvas.move("box", dx, dy)

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
                is_edge_x = abs(lx) > (w / 2 - margin)
                is_edge_y = abs(ly) > (h / 2 - margin)
                
                self.start_ox = ox
                self.start_oy = oy
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
        self.start_x = event.x
        self.start_y = event.y
        
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, 
            outline=generate_class_color(self.get_active_class(), self.class_colors), 
            width=2, 
            tags="temp_box"
        )

    def do_interaction(self, event):
        ox, oy = self.canvas_to_img(event.x, event.y)
        if self.action_mode == "move":
            self.current_boxes[self.selected_box_idx][1] = self.interact_start_state[0] + (ox - self.start_ox)
            self.current_boxes[self.selected_box_idx][2] = self.interact_start_state[1] + (oy - self.start_oy)
            self.show_image()
        elif self.action_mode == "resize":
            lx, ly = get_local_coords(ox, oy, self.interact_start_state[0], self.interact_start_state[1], self.interact_start_state[4])
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
                self.current_boxes.append([self.get_active_class(), ox1 + ow / 2, oy1 + oh / 2, ow, oh, 0.0])
                self.selected_box_idx = len(self.current_boxes) - 1
                
            self.show_image()
        self.action_mode = "none"

    def clear_all_boxes(self):
        self.push_state()
        self.current_boxes.clear()
        self.selected_box_idx = -1
        self.show_image()

    def draw_all_boxes(self):
        if not self.show_annotations: 
            return
            
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
            
            self.canvas.create_text(text_x + 1, text_y - 1, text=cls_name, fill="#000000", anchor="sw", font=("Segoe UI", font_size, "bold"), tags="box")
            self.canvas.create_text(text_x, text_y - 2, text=cls_name, fill=outline_color, anchor="sw", font=("Segoe UI", font_size, "bold"), tags="box")

    def show_image(self):
        if not self.original_img: 
            return
            
        self.canvas.delete("img", "box", "hud")
        
        # HEAVY PROCESSING BLOCK - Only runs if specifically flagged
        if getattr(self, 'needs_reprocess', True):
            w, h = self.original_img.size
            resized = self.original_img.resize((max(1, int(w * self.zoom_level)), max(1, int(h * self.zoom_level))), Image.Resampling.BILINEAR)
            
            if self.brightness != 1.0: 
                resized = ImageEnhance.Brightness(resized).enhance(self.brightness)
            if self.contrast != 1.0: 
                resized = ImageEnhance.Contrast(resized).enhance(self.contrast)
                
            self.cached_tk_img = ImageTk.PhotoImage(resized)
            self.needs_reprocess = False  # Reset flag after processing
            
        # Draw the instantly available cached image
        self.canvas.create_image(self.img_x, self.img_y, image=self.cached_tk_img, anchor="center", tags="img")
        self.draw_all_boxes()
        
        # ----------------- HUD Drawing Logic -----------------
        filename = self.image_list[self.current_index]
        pure_base = self._get_pure_basename(filename)
        aug_count = max(0, sum(1 for img in self.image_list if self._get_pure_basename(img) == pure_base) - 1)
        
        pad_x, pad_y = 12, 10
        hud_font = ("Consolas", 9, "bold") 
        hud_text = f"ZOOM       : {int(self.zoom_level * 100)}%\nBRIGHTNESS : {self.brightness:.1f}x\nCONTRAST   : {self.contrast:.1f}x\nAUGS FOUND : {aug_count} variants"
        
        t_hud = self.canvas.create_text(25, 20, text=hud_text, fill="#ffffff", anchor="nw", font=hud_font, tags="hud")
        bbox_hud = self.canvas.bbox(t_hud)
        
        if bbox_hud:
            bg_hud = self.canvas.create_rectangle(bbox_hud[0] - pad_x, bbox_hud[1] - pad_y, bbox_hud[2] + pad_x, bbox_hud[3] + pad_y, fill="#1c1c1c", outline="#333333", width=1, tags="hud")
            acc_hud = self.canvas.create_line(bbox_hud[0] - pad_x, bbox_hud[1] - pad_y, bbox_hud[0] - pad_x, bbox_hud[3] + pad_y, fill="#00f2ff", width=4, tags="hud")
            self.canvas.tag_lower(acc_hud, t_hud)
            self.canvas.tag_lower(bg_hud, acc_hud)
            
        if hasattr(self, 'toast_msg') and self.toast_msg:
            cw = self.canvas.winfo_width()
            if cw <= 1: 
                cw = self.root.winfo_screenwidth() - 400 
            step = getattr(self, 'toast_step', 0)
            
            text_color = fade_hex_color("#ffffff", step)
            bg_color = fade_hex_color("#1c1c1c", step)
            outline_color = fade_hex_color("#333333", step)
            acc_color = fade_hex_color(self.toast_color, step)
            
            t_toast = self.canvas.create_text(cw - 25, 20, text=f"🔔  {self.toast_msg}", fill=text_color, anchor="ne", font=hud_font, tags="hud")
            bbox_toast = self.canvas.bbox(t_toast)
            
            if bbox_toast:
                bg_toast = self.canvas.create_rectangle(bbox_toast[0] - pad_x, bbox_toast[1] - pad_y, bbox_toast[2] + pad_x, bbox_toast[3] + pad_y, fill=bg_color, outline=outline_color, width=1, tags="hud")
                acc_toast = self.canvas.create_line(bbox_toast[2] + pad_x, bbox_toast[1] - pad_y, bbox_toast[2] + pad_x, bbox_toast[3] + pad_y, fill=acc_color, width=4, tags="hud")
                self.canvas.tag_lower(acc_toast, t_toast)
                self.canvas.tag_lower(bg_toast, acc_toast)
        
        self.canvas.itemconfig("crosshair", state="normal" if self.show_annotations else "hidden")
        if self.show_annotations: 
            self.canvas.tag_raise("crosshair")
        self.canvas.tag_raise("hud")

    def reset_view(self, event=None):
        if not self.original_img: 
            return
            
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw <= 1: 
            return
            
        img_w, img_h = self.original_img.size
        
        # Reset zoom and center coordinates
        self.zoom_level = min(cw / img_w, ch / img_h) * 0.95
        self.img_x = cw // 2
        self.img_y = ch // 2
        
        self.needs_reprocess = True
        self.show_image()
        self.set_status_message("View Reset", "#00f2ff", 1500)

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
        self.root.bind("<Control-v>", lambda e: self.paste_box() if is_not_typing(e) else None)
        self.root.bind("<v>", lambda e: self.toggle_visibility() if is_not_typing(e) else None)
        self.root.bind("<r>", lambda e: self.reset_view() if is_not_typing(e) else None)
        self.root.bind("-", lambda e: self.adjust_enhancement('brightness', -0.2) if is_not_typing(e) else None)
        self.root.bind("=", lambda e: self.adjust_enhancement('brightness', 0.2) if is_not_typing(e) else None)
        self.root.bind("+", lambda e: self.adjust_enhancement('brightness', 0.2) if is_not_typing(e) else None)
        self.root.bind("[", lambda e: self.adjust_enhancement('contrast', -0.2) if is_not_typing(e) else None)
        self.root.bind("]", lambda e: self.adjust_enhancement('contrast', 0.2) if is_not_typing(e) else None)

        # Bind keys 1 through 9 for fast class switching
        for i in range(1, 10):
            # i-1 converts key '1' to index 0, key '2' to index 1, etc.
            self.root.bind(str(i), lambda e, idx=i-1: self.quick_select_class(idx) if is_not_typing(e) else None)

    def setup_ui(self):
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(fill="both", expand=True)
        
        self._setup_sidebar()
        self._setup_viewer()

    def _setup_sidebar(self):
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
        tk.Button(action_frame, text="🎛️ AUGMENT DATA", bg="#008080", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.show_augmentation_dialog).pack(fill="x")

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
        self.search_var.trace("w", lambda n, i, m: self.render_tags())
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var, bg="#2d2d2d", fg="white", insertbackground="white", borderwidth=0, font=("Segoe UI", 11))
        self.search_entry.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.search_entry.bind("<Return>", self.add_to_master)
        tk.Button(search_frame, text="ADD", bg="#0078d4", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), width=6, command=self.add_to_master).pack(side="right", fill="y")

        btn_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        btn_frame.pack(fill="x", side="bottom", padx=20, pady=5)
        self.btn_delete_img = tk.Button(btn_frame, text="🗑️ DELETE IMAGE", bg="#8B0000", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=3, command=self.delete_current_image)
        self.btn_delete_img.pack(fill="x", pady=(0, 6))
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

    def _setup_viewer(self):
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

        self._setup_nav_bar()

    def _setup_nav_bar(self):
        self.nav_info_bar = tk.Frame(self.viewer_frame, bg="#1e1e1e", height=50)
        self.nav_info_bar.pack(fill="x", pady=(5,0))
        
        tk.Button(self.nav_info_bar, text="<< PREV", bg="#333", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), command=self.prev_image, width=12).pack(side="left", padx=20, pady=10)
        tk.Button(self.nav_info_bar, text="SKIP >>", bg="#333", fg="white", relief="flat", font=("Segoe UI", 8, "bold"), command=self.next_image, width=12).pack(side="right", padx=(5, 20))
        self.status_text = tk.Label(self.nav_info_bar, text="", bg="#1e1e1e", fg="#888", font=("Consolas", 10))
        self.status_text.pack(side="right", padx=10)

        info_mid = tk.Frame(self.nav_info_bar, bg="#1e1e1e")
        info_mid.pack(side="left", fill="x", expand=True)
        tk.Label(info_mid, text="Go to:", bg="#1e1e1e", fg="#888", font=("Segoe UI", 9)).pack(side="left")
        self.jump_entry = tk.Entry(info_mid, textvariable=self.jump_var, width=10, bg="#2d2d2d", fg="#00f2ff", borderwidth=0, justify="center", font=("Consolas", 11, "bold"))
        self.jump_entry.pack(side="left", padx=10, ipady=3)
        self.jump_entry.bind("<Return>", self.jump_to_image)

        tk.Checkbutton(info_mid, text="Untagged", variable=self.untagged_only_var, bg="#1e1e1e", fg="#aaa", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#00f2ff", cursor="hand2", command=self._on_untagged_toggle).pack(side="left", padx=(5, 0))
        tk.Label(info_mid, text=" | Filter:", bg="#1e1e1e", fg="#888", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(10, 5))
        self.filter_combo = ttk.Combobox(info_mid, textvariable=self.filter_class_var, state="readonly", width=15, font=("Segoe UI", 9))
        self.filter_combo.pack(side="left", padx=(0, 10))
        self.filter_combo.bind("<<ComboboxSelected>>", self.apply_filter)
        self._update_filter_combo_values()
        
        self.save_indicator = tk.Label(info_mid, text="", bg="#1e1e1e", font=("Segoe UI", 12))
        self.save_indicator.pack(side="left", padx=2)
        self.filename_label = tk.Label(info_mid, text="", bg="#1e1e1e", fg="#00f2ff", font=("Segoe UI", 9, "bold"), anchor="w")
        self.filename_label.pack(side="left", fill="x", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionTag_Enterprise(root)
    root.after(500, app.load_image)
    root.mainloop()