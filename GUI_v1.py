import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from CT_RS_RD_Match import *
from PIL import Image, ImageTk
import sys

def main(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd):
    """主函数：加载 CT、RT Structure、RT Dose 并进行可视化"""
    # 重定向 print 输出到 GUI 的文本框
    def custom_print(*args, **kwargs):
        text_output.insert(tk.END, " ".join(map(str, args)) + "\n")
        text_output.see(tk.END)
        root.update()

    # 临时替换内置 print 函数
    import builtins
    original_print = builtins.print
    builtins.print = custom_print

    try:
        # 1️⃣ 读取 CT 影像
        ct_array, ct_origin, ct_spacing, _ = load_ct_images(ct_folder)
        # 2️⃣ 读取 RT Structure 并转换为掩码
        contours = load_rt_structure(rt_structure_file)
        mask = convert_contours_to_mask(ct_array, ct_origin, ct_spacing, contours, roi_name)

        # 3️⃣ 读取 RT Dose
        dose_array, dose_origin, dose_spacing = load_dose(dose_file)

        if write_mhd:
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            save_as_mhd(ct_array, ct_origin, ct_spacing, os.path.join(tmp_folder, "ct_array.mhd"))
            save_as_mhd(dose_array, dose_origin, dose_spacing, os.path.join(tmp_folder, "dose_array.mhd"))
            save_as_mhd(mask, ct_origin, ct_spacing, os.path.join(tmp_folder, "mask_array.mhd"))

        # 4️⃣ 重新采样 CT 和掩码到剂量坐标系
        print(f"ct_origin:{ct_origin}, ct_spacing:{ct_spacing}, ct_shape:{ct_array.shape[::-1]}")
        print(f"dose_origin:{dose_origin}, dose_spacing:{dose_spacing}, dose_shape:{dose_array.shape[::-1]}")
        ct_range = compute_physical_range(ct_origin, ct_spacing, ct_array.shape[::-1])
        dose_range = compute_physical_range(dose_origin, dose_spacing, dose_array.shape[::-1])
        print(f"✅ CT 物理范围 (mm): X:[{ct_range[0]}, {ct_range[1]}], Y:[{ct_range[2]}, {ct_range[3]}], Z:[{ct_range[4]}, {ct_range[5]}]")
        print(f"✅ Dose 物理范围 (mm): X:[{dose_range[0]}, {dose_range[1]}], Y:[{dose_range[2]}, {dose_range[3]}], Z:[{dose_range[4]}, {dose_range[5]}]")

        ct_resampled = resample_to_dose_grid(ct_array, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing, is_mask=False)
        print(f"Before Resampling, Mask Sum: {np.sum(mask)}")
        mask_resampled = resample_to_dose_grid(mask, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing, is_mask=True)
        print(f"After Resampling, Mask Sum: {np.sum(mask_resampled)}")
        volume = compute_roi_volume(mask, ct_spacing)
        print(f"Before Resampling, {roi_name} volume: {volume}cc")
        volume = compute_roi_volume(mask_resampled, dose_spacing)
        print(f"After Resampling, {roi_name} volume: {volume}cc")
        if write_mhd:
            save_as_mhd(ct_resampled, dose_origin, dose_spacing, os.path.join(tmp_folder, "ct_resampled.mhd"))
            save_as_mhd(mask_resampled, dose_origin, dose_spacing, os.path.join(tmp_folder, "mask_resampled.mhd"))

        # 5️⃣ 进行可视化
        plot_ct_contour_dose_interactive_best1(ct_resampled, mask_resampled, dose_array)

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        # 恢复原始 print 函数
        builtins.print = original_print

# GUI 界面设计
def create_gui():
    global root, text_output

    root = tk.Tk()
    root.title("CT + Contour + Dose Visualization")
    # 设置窗口图标
    try:
        # 加载图标文件（支持 .png 或 .gif）
        icon = Image.open("2023-06-02_105437.ico")  # 替换为你的图标路径
        icon = icon.resize((32, 32), Image.Resampling.LANCZOS)  # 调整大小为 32x32
        photo = ImageTk.PhotoImage(icon)
        root.iconphoto(True, photo)  # 设置图标，True 表示应用于所有窗口
    except Exception as e:
        print(f"Failed to load icon: {e}")  # 如果加载失败，打印错误
    root.geometry("600x600")

    # 框架布局
    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill="both", expand=True)

    # CT 文件夹
    tk.Label(frame, text="CT Folder:").grid(row=0, column=0, sticky="w")
    ct_folder_var = tk.StringVar()
    tk.Entry(frame, textvariable=ct_folder_var, width=50).grid(row=0, column=1)
    tk.Button(frame, text="Browse", command=lambda: ct_folder_var.set(filedialog.askdirectory())).grid(row=0, column=2)

    # RT Structure 文件
    tk.Label(frame, text="RT Structure File:").grid(row=1, column=0, sticky="w")
    rt_structure_var = tk.StringVar()
    tk.Entry(frame, textvariable=rt_structure_var, width=50).grid(row=1, column=1)
    tk.Button(frame, text="Browse", command=lambda: rt_structure_var.set(filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")]))).grid(row=1, column=2)

    # RT Dose 文件
    tk.Label(frame, text="RT Dose File:").grid(row=2, column=0, sticky="w")
    dose_file_var = tk.StringVar()
    tk.Entry(frame, textvariable=dose_file_var, width=50).grid(row=2, column=1)
    tk.Button(frame, text="Browse", command=lambda: dose_file_var.set(filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")]))).grid(row=2, column=2)

    # ROI 名称
    tk.Label(frame, text="ROI Name:").grid(row=3, column=0, sticky="w")
    roi_name_var = tk.StringVar(value="CtvLung")
    tk.Entry(frame, textvariable=roi_name_var, width=50).grid(row=3, column=1)

    # 输出文件夹
    tk.Label(frame, text="Output Folder:").grid(row=4, column=0, sticky="w")
    tmp_folder_var = tk.StringVar(value="Results")
    tk.Entry(frame, textvariable=tmp_folder_var, width=50).grid(row=4, column=1)
    tk.Button(frame, text="Browse", command=lambda: tmp_folder_var.set(filedialog.askdirectory())).grid(row=4, column=2)

    # 是否保存 MHD
    write_mhd_var = tk.BooleanVar(value=True)
    tk.Checkbutton(frame, text="Save MHD Files", variable=write_mhd_var).grid(row=5, column=1, sticky="w")

    # 输出文本框
    tk.Label(frame, text="Output:").grid(row=6, column=0, sticky="nw")
    text_output = scrolledtext.ScrolledText(frame, width=70, height=20)
    text_output.grid(row=7, column=0, columnspan=3, pady=5)

    # 运行和退出按钮
    tk.Button(frame, text="Run", command=lambda: run_program(
        ct_folder_var.get(), rt_structure_var.get(), dose_file_var.get(),
        roi_name_var.get(), tmp_folder_var.get(), write_mhd_var.get()
    )).grid(row=8, column=1, pady=10)
    tk.Button(frame, text="Exit", command=root.quit).grid(row=8, column=2)

    # 运行程序的函数
    def run_program(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd):
        if not all([ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder]):
            messagebox.showwarning("Input Error", "Please fill in all fields!")
            return
        text_output.delete(1.0, tk.END)  # 清空输出框
        main(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd)

    root.mainloop()

def check_python_version():
    """检查 Python 版本，并在使用 3.12 时给出警告"""
    python_version = sys.version_info
    current_version = f"{python_version.major}.{python_version.minor}"
    if python_version.major == 3 and python_version.minor == 12:
        warning_msg = (
            f"警告：当前运行的 Python 版本为 {current_version}。\n"
            "此程序可能在 Python 3.12 上存在兼容性问题。\n"
            "建议使用 Python 3.10 以确保最佳性能和稳定性。\n"
            "您可以继续运行，但可能会遇到意外错误。"
        )
        messagebox.showwarning("Python 版本警告", warning_msg)
    print(f"当前 Python 版本: {current_version}")

def main():
    # 检查 Python 版本
    check_python_version()
    create_gui()

if __name__ == "__main__":
    main()