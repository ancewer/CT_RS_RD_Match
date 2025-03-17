import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from CT_RS_RD_Match import *
from PIL import Image, ImageTk
import os
import pydicom

# 默认窗位窗宽预设值 (Window Level, Window Width)
WINDOW_PRESETS = {
    "Lung": (-600, 1500),      # 肺窗
    "Soft Tissue": (40, 400),  # 软组织窗
    "Bone": (300, 1500),       # 骨窗
    "Brain": (40, 80),         # 脑窗
    "Custom": (0, 1000)       # 默认自定义
}

def run(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd, window_level=0, window_width=1000):
    """主函数：加载 CT、RT Structure、RT Dose 并进行可视化"""
    def custom_print(*args, **kwargs):
        text_output.insert(tk.END, " ".join(map(str, args)) + "\n")
        text_output.see(tk.END)
        root.update()

    import builtins
    original_print = builtins.print
    builtins.print = custom_print

    try:
        ct_array, ct_origin, ct_spacing, _ = load_ct_images(ct_folder)
        contours = load_rt_structure(rt_structure_file)
        mask = convert_contours_to_mask(ct_array, ct_origin, ct_spacing, contours, roi_name)
        dose_array, dose_origin, dose_spacing = load_dose(dose_file)

        if write_mhd:
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            save_as_mhd(ct_array, ct_origin, ct_spacing, os.path.join(tmp_folder, "ct_array.mhd"))
            save_as_mhd(dose_array, dose_origin, dose_spacing, os.path.join(tmp_folder, "dose_array.mhd"))
            save_as_mhd(mask, ct_origin, ct_spacing, os.path.join(tmp_folder, "mask_array.mhd"))

        print(f"✅ct_origin:{ct_origin}, ct_spacing:{ct_spacing}, ct_shape:{ct_array.shape[::-1]}")
        print(f"✅dose_origin:{dose_origin}, dose_spacing:{dose_spacing}, dose_shape:{dose_array.shape[::-1]}")
        ct_range = compute_physical_range(ct_origin, ct_spacing, ct_array.shape[::-1])
        dose_range = compute_physical_range(dose_origin, dose_spacing, dose_array.shape[::-1])
        print(f"✅ CT 物理范围 (mm): X:[{ct_range[0]}, {ct_range[1]}], Y:[{ct_range[2]}, {ct_range[3]}], Z:[{ct_range[4]}, {ct_range[5]}]")
        print(f"✅ Dose 物理范围 (mm): X:[{dose_range[0]}, {dose_range[1]}], Y:[{dose_range[2]}, {dose_range[3]}], Z:[{dose_range[4]}, {dose_range[5]}]")

        ct_resampled = resample_to_dose_grid(ct_array, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing, is_mask=False)
        print(f"✅Before Resampling, Mask Sum: {np.sum(mask)}")
        mask_resampled = resample_to_dose_grid(mask, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing, is_mask=True)
        print(f"✅After Resampling, Mask Sum: {np.sum(mask_resampled)}")
        volume = compute_roi_volume(mask, ct_spacing)
        print(f"✅Before Resampling, {roi_name} volume: {volume}cc")
        volume = compute_roi_volume(mask_resampled, dose_spacing)
        print(f"✅After Resampling, {roi_name} volume: {volume}cc")
        if write_mhd:
            save_as_mhd(ct_resampled, dose_origin, dose_spacing, os.path.join(tmp_folder, "ct_resampled.mhd"))
            save_as_mhd(mask_resampled, dose_origin, dose_spacing, os.path.join(tmp_folder, "mask_resampled.mhd"))

        plot_ct_contour_dose_interactive_best1(ct_resampled, mask_resampled, dose_array, window_width, window_level)

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        builtins.print = original_print

# GUI 界面设计
def create_gui():
    global root, text_output, ct_combo, rs_combo, rd_combo, roi_combo

    root = tk.Tk()
    root.title("CT + Contour + Dose Visualization")
    try:
        icon = Image.open("2023-06-02_105437.ico")
        icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(icon)
        root.iconphoto(True, photo)
    except Exception as e:
        print(f"Failed to load icon: {e}")
    root.geometry("600x750")  # 初始大小
    root.minsize(600, 750)  # 最小大小

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill="both", expand=True)

    # 定义辅助函数
    def update_roi_options(*args):
        rs_file = rs_combo.get()
        if rs_file and os.path.exists(rs_file):
            try:
                contours = load_rt_structure(rs_file)
                roi_names = list(contours.keys())
                roi_menu = tk.OptionMenu(frame, roi_combo, *roi_names)
                roi_menu.grid(row=4, column=1, sticky="w")
                if roi_names:
                    roi_combo.set(roi_names[0])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROI names: {e}")
        else:
            roi_combo.set("")

    def load_files(main_folder_var):
        folder = main_folder_var.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Input Error", "Please select a valid folder!")
            return
        progress_bar["value"] = 0
        root.update()

        ct_folders = set()
        rs_files = []
        rd_files = []

        all_files = []
        for subdir, _, files in os.walk(folder):
            for f in files:
                all_files.append(os.path.join(subdir, f))

        total_files = len(all_files)
        for i, file_path in enumerate(all_files):
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                modality = ds.get((0x0008, 0x0060), None)
                if modality:
                    modality = modality.value
                    if modality == "CT":
                        ct_folders.add(os.path.dirname(file_path))
                    elif modality == "RTSTRUCT":
                        rs_files.append(file_path)
                    elif modality == "RTDOSE":
                        rd_files.append(file_path)
            except Exception:
                continue
            # 模拟进度
            progress_bar["value"] = (i + 1) * 80 / total_files
            root.update()

        ct_list = sorted(list(ct_folders))
        ct_menu = tk.OptionMenu(frame, ct_combo, *ct_list)
        ct_menu.grid(row=1, column=1, sticky="w")
        if ct_list:
            ct_combo.set(ct_list[0])

        rs_menu = tk.OptionMenu(frame, rs_combo, *rs_files)
        rs_menu.grid(row=2, column=1, sticky="w")
        if rs_files:
            rs_combo.set(rs_files[0])

        rd_menu = tk.OptionMenu(frame, rd_combo, *rd_files)
        rd_menu.grid(row=3, column=1, sticky="w")
        if rd_files:
            rd_combo.set(rd_files[0])
        progress_bar["value"] = 90
        update_roi_options()
        progress_bar["value"] = 100
        root.update()
        adjust_window_size()

    def run_program(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd):
        if not all([ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder]):
            messagebox.showwarning("Input Error", "Please select all required options!")
            return
        text_output.delete(1.0, tk.END)
        progress_bar["value"] = 0
        root.update()
        # 获取窗位窗宽
        preset_name = window_preset_combo.get()
        window_level, window_width = WINDOW_PRESETS[preset_name]
        run(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd, window_level, window_width)
        progress_bar["value"] = 100
        root.update()

    def adjust_window_size():
        """动态调整窗口大小以适应内容"""
        # 计算所有控件的总宽度和高度
        frame.update_idletasks()  # 更新布局以获取实际大小
        required_width = frame.winfo_reqwidth() + 20  # 加上边距
        required_height = frame.winfo_reqheight() + 20

        # 设置最小窗口大小
        min_width, min_height = 600, 750
        new_width = max(min_width, required_width)
        new_height = max(min_height, required_height)

        # 更新窗口大小
        root.geometry(f"{new_width}x{new_height}")

    # GUI 布局
    tk.Label(frame, text="Main Folder:").grid(row=0, column=0, sticky="w")
    main_folder_var = tk.StringVar()
    tk.Entry(frame, textvariable=main_folder_var, width=40).grid(row=0, column=1, sticky="w")
    tk.Button(frame, text="Browse", command=lambda:main_folder_var.set(filedialog.askdirectory())).grid(row=0, column=2, padx=5)
    tk.Button(frame, text="Load...", command=lambda: load_files(main_folder_var)).grid(row=0, column=3)

    tk.Label(frame, text="CT Folder:").grid(row=1, column=0, sticky="w")
    ct_combo = tk.StringVar()
    tk.OptionMenu(frame, ct_combo, "").grid(row=1, column=1, sticky="w")

    tk.Label(frame, text="RT Structure File:").grid(row=2, column=0, sticky="w")
    rs_combo = tk.StringVar()
    rs_menu = tk.OptionMenu(frame, rs_combo, "")
    rs_menu.grid(row=2,columnspan=2, column=1, sticky="w")
    rs_combo.trace("w", update_roi_options)

    tk.Label(frame, text="RT Dose File:").grid(row=3, column=0, sticky="w")
    rd_combo = tk.StringVar()
    tk.OptionMenu(frame, rd_combo, "").grid(row=3, column=1, sticky="w")

    tk.Label(frame, text="ROI Name:").grid(row=4, column=0, sticky="w")
    roi_combo = tk.StringVar()
    tk.OptionMenu(frame, roi_combo, "").grid(row=4, column=1, sticky="w")

    tk.Label(frame, text="Output Folder:").grid(row=5, column=0, sticky="w")
    tmp_folder_var = tk.StringVar(value="Results")
    tk.Entry(frame, textvariable=tmp_folder_var, width=50).grid(row=5, column=1)
    tk.Button(frame, text="Browse", command=lambda: tmp_folder_var.set(filedialog.askdirectory())).grid(row=5, column=2)

    # 添加窗位窗宽预设选择
    tk.Label(frame, text="Window Preset:").grid(row=6, column=0, sticky="w")
    window_preset_combo = tk.StringVar(value="Custom")  # 默认选择 Custom
    preset_menu = tk.OptionMenu(frame, window_preset_combo, *WINDOW_PRESETS.keys())
    preset_menu.grid(row=6, column=1, sticky="w")

    write_mhd_var = tk.BooleanVar(value=True)
    tk.Checkbutton(frame, text="Save MHD Files", variable=write_mhd_var).grid(row=7, column=1, sticky="w")

    tk.Label(frame, text="Output:").grid(row=8, column=0, sticky="nw")
    text_output = scrolledtext.ScrolledText(frame, width=70, height=20)
    text_output.grid(row=9, column=0, columnspan=4, pady=5)

    tk.Button(frame, text="Run", command=lambda: run_program(
        ct_combo.get(), rs_combo.get(), rd_combo.get(),
        roi_combo.get(), tmp_folder_var.get(), write_mhd_var.get()
    )).grid(row=10, column=1, pady=10)
    tk.Button(frame, text="Exit", command=root.quit).grid(row=10, column=2)

    # 添加进度条
    progress_bar = ttk.Progressbar(frame, length=400, mode="determinate", maximum=100)
    progress_bar.grid(row=11, column=0, columnspan=4, pady=10)

    # 配置行列权重，使窗口可伸缩
    for i in range(12):  # 行数
        frame.grid_rowconfigure(i, weight=1 if i == 8 else 0)  # 文本框行可伸缩
    for i in range(4):  # 列数
        frame.grid_columnconfigure(i, weight=1 if i == 1 else 0)  # 下拉菜单列可伸缩

    root.mainloop()

def main():
    create_gui()

if __name__ == "__main__":
    main()