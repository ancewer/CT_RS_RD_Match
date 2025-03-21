import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from functions import extract_spot_data, plot_spot_distribution, convert_to_csv, parse_csv_data, merge_repaint_spots, split_repaint_spots
from PIL import Image, ImageTk
import os
import sys
import pydicom
import io  # 新增导入 io 模块用于重定向

class RTIonSpotComparer:
    def __init__(self, root):
        self.root = root
        self.root.title("RT Ion Plan Analysis")
        self.root.configure(bg="#f0f4f8")
        try:
            icon = Image.open("2023-06-02_105437.ico")
            icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(icon)
            root.iconphoto(True, photo)
        except Exception as e:
            print(f"Failed to load icon: {e}")

        self.root.geometry("600x730")  # 增加高度以容纳新文本框

        self.files = []
        self.beam_data = []
        self.file_types = []

        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.root, bg="#4682b4", fg="white")
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, bg="#4682b4", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Files", command=self.browse_files, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")

        help_menu = tk.Menu(menubar, tearoff=0, bg="#4682b4", fg="white")
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        self.root.bind("<Control-o>", lambda e: self.browse_files())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="RT Ion Plan Analysis", font=("Arial", 14, "bold"), bg="#4682b4", fg="white", pady=10)
        title_label.pack(fill=tk.X)

        # 文件选择区域（居中）
        file_frame = tk.Frame(self.root, bg="#f0f4f8")
        file_frame.pack(expand=True, pady=2)
        tk.Label(file_frame, text="Please Import RT Ion Plan Files or CSV Files", font=("Arial", 12), bg="#f0f4f8").pack(pady=5)
        button_subframe = tk.Frame(file_frame, bg="#f0f4f8")
        button_subframe.pack()
        tk.Button(button_subframe, text="Browse Files", command=self.browse_files, bg="#4682b4", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.file_count_label = tk.Label(button_subframe, text="Selected files: 0", font=("Arial", 10), bg="#f0f4f8")
        self.file_count_label.pack(side=tk.LEFT, padx=5)

        # 文件列表区域
        list_frame = tk.Frame(self.root, bg="#f0f4f8")
        list_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(list_frame, text="Loaded Files:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(anchor="w")
        self.file_list_frame = tk.Frame(list_frame, bg="#ffffff", bd=1, relief=tk.SUNKEN)
        self.file_list_frame.pack(fill=tk.X)
        self.file_list_text = tk.Text(self.file_list_frame, height=5, width=50, font=("Arial", 9), bg="#ffffff")
        scrollbar = tk.Scrollbar(self.file_list_frame, command=self.file_list_text.yview)
        self.file_list_text.config(yscrollcommand=scrollbar.set)
        self.file_list_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list_text.insert(tk.END, "No files loaded yet.")
        self.file_list_text.config(state=tk.DISABLED)

        # 绑定鼠标滚轮到文件列表
        self.bind_mouse_wheel(self.file_list_text)

        # Spot Size Multiplier
        size_frame = tk.Frame(self.root, bg="#f0f4f8", pady=5)
        size_frame.pack(fill=tk.X, padx=10)
        tk.Label(size_frame, text="Spot Size Multiplier:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(side=tk.LEFT)
        self.spot_size_slider = tk.Scale(size_frame, from_=1, to=200, orient=tk.HORIZONTAL, resolution=10, length=200, bg="#f0f4f8")
        self.spot_size_slider.set(50)
        self.spot_size_slider.pack(side=tk.LEFT, padx=5)

        # Plot Type
        plot_frame = tk.Frame(self.root, bg="#f0f4f8", pady=5)
        plot_frame.pack(fill=tk.X, padx=10)
        tk.Label(plot_frame, text="Plot Type:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(side=tk.LEFT)
        self.plot_type_var = tk.StringVar(value="line")
        tk.Radiobutton(plot_frame, text="Scatter Plot", variable=self.plot_type_var, value="scatter", bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(plot_frame, text="Line Plot", variable=self.plot_type_var, value="line", bg="#f0f4f8").pack(side=tk.LEFT, padx=5)

        # Axes Options
        axes_frame = tk.Frame(self.root, bg="#f0f4f8", pady=5)
        axes_frame.pack(fill=tk.X, padx=10)
        tk.Label(axes_frame, text="Axes Options:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(side=tk.LEFT)
        self.equal_axes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(axes_frame, text="Equal X and Y Axes", variable=self.equal_axes_var, bg="#f0f4f8").pack(side=tk.LEFT, padx=5)

        # Spot Weight Comparison
        weight_frame = tk.Frame(self.root, bg="#f0f4f8", pady=5)
        weight_frame.pack(fill=tk.X, padx=10)
        tk.Label(weight_frame, text="Spot Weight Comparison:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(side=tk.LEFT)
        self.compare_weights_var = tk.BooleanVar(value=False)
        tk.Checkbutton(weight_frame, text="Compare Spot Weights", variable=self.compare_weights_var, bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
        tk.Label(weight_frame, text="Tolerance (%):", bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
        self.tolerance_entry = tk.Entry(weight_frame, width=5)
        self.tolerance_entry.insert(0, "1")
        self.tolerance_entry.pack(side=tk.LEFT)

        # Weight Decimal Places
        decimal_frame = tk.Frame(self.root, bg="#f0f4f8", pady=5)
        decimal_frame.pack(fill=tk.X, padx=10)
        tk.Label(decimal_frame, text="Weight Decimal Places:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(side=tk.LEFT)
        self.decimal_places_var = tk.IntVar(value=2)
        decimal_menu = ttk.Combobox(decimal_frame, textvariable=self.decimal_places_var, values=[0, 1, 2, 3, 4], width=5)
        decimal_menu.pack(side=tk.LEFT, padx=5)

        # Spot 显示选项
        display_frame = tk.Frame(self.root, bg="#f0f4f8", pady=5)
        display_frame.pack(fill=tk.X, padx=10)
        tk.Label(display_frame, text="Display Options:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(side=tk.LEFT)
        self.show_spot_numbers_var = tk.BooleanVar(value=False)
        tk.Checkbutton(display_frame, text="Show Spot Numbers", variable=self.show_spot_numbers_var, bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
        self.show_spot_weights_var = tk.BooleanVar(value=True)
        tk.Checkbutton(display_frame, text="Show Spot Weights", variable=self.show_spot_weights_var, bg="#f0f4f8").pack(side=tk.LEFT, padx=5)

        # 输出信息区域
        output_frame = tk.Frame(self.root, bg="#f0f4f8")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(output_frame, text="Operation Output:", font=("Arial", 10, "bold"), bg="#f0f4f8").pack(anchor="w")
        self.output_text_frame = tk.Frame(output_frame, bg="#ffffff", bd=1, relief=tk.SUNKEN)
        self.output_text_frame.pack(fill=tk.X)
        self.output_text = tk.Text(self.output_text_frame, height=10, width=50, font=("Arial", 9), bg="#ffffff")
        scrollbar_output = tk.Scrollbar(self.output_text_frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar_output.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar_output.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.insert(tk.END, "No operations performed yet.\n")
        self.output_text.config(state=tk.DISABLED)
        self.bind_mouse_wheel(self.output_text)

        # 操作按钮（居中显示）
        button_frame = tk.Frame(self.root, bg="#f0f4f8")
        button_frame.pack(expand=True, pady=20)
        tk.Button(button_frame, text="Plot Spot Distribution", command=self.select_beams, bg="#4682b4", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Convert to CSV", command=self.convert_to_csv, bg="#4682b4", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Merge Repaint Spots", command=self.merge_repaint_spots, bg="#4682b4", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Split Repaint Spots", command=self.split_repaint_spots, bg="#4682b4", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

    def bind_mouse_wheel(self, widget):
        """绑定鼠标滚轮到指定控件"""
        if sys.platform == "win32":
            widget.bind("<MouseWheel>", lambda event: widget.yview_scroll(-1 * int(event.delta / 120), "units"))
        else:  # macOS 和 Linux
            widget.bind("<Button-4>", lambda event: widget.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda event: widget.yview_scroll(1, "units"))

    def redirect_output(self, func, *args, **kwargs):
        """重定向 print 输出到 output_text"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)  # 清空现有内容
        stdout_backup = sys.stdout
        output = io.StringIO()
        sys.stdout = output
        try:
            func(*args, **kwargs)
        finally:
            sys.stdout = stdout_backup
            self.output_text.insert(tk.END, output.getvalue())
            self.output_text.config(state=tk.DISABLED)
            output.close()

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select RT Ion Plan DICOM or CSV Files",
            filetypes=[("DICOM files", "*.dcm"), ("CSV files", "*.csv")],
            parent=self.root
        )
        if files:
            self.files = list(files)
            self.file_types = [os.path.splitext(f)[1][1:].lower() for f in files]
            self.file_count_label.config(text=f"Selected files: {len(self.files)}")
            self.file_list_text.config(state=tk.NORMAL)
            self.file_list_text.delete(1.0, tk.END)
            self.file_list_text.insert(tk.END, "\n".join([os.path.basename(f) for f in self.files]))
            self.file_list_text.config(state=tk.DISABLED)
            self.beam_data = []
        else:
            messagebox.showwarning("Warning", "No files selected!")
            self.file_list_text.config(state=tk.NORMAL)
            self.file_list_text.delete(1.0, tk.END)
            self.file_list_text.insert(tk.END, "No files loaded yet.")
            self.file_list_text.config(state=tk.DISABLED)

    def select_beams(self):
        if not self.files:
            messagebox.showerror("Error", "Please select DICOM or CSV files first!")
            return

        self.beam_data = []
        for file_path, file_type in zip(self.files, self.file_types):
            if file_type == "dcm":
                beam_info = extract_spot_data(file_path)
            elif file_type in ("csv", "txt"):
                beam_info = parse_csv_data(file_path)
            else:
                continue

            if beam_info:
                self.beam_data.append(beam_info)

        if not self.beam_data:
            messagebox.showerror("Error", "No valid beam data found in selected files!")
            return

        self.beam_selection_window()

    def beam_selection_window(self):
        beam_window = tk.Toplevel(self.root)
        beam_window.title("Select Beams")
        beam_window.geometry("400x300")
        beam_window.configure(bg="#f0f4f8")

        canvas = tk.Canvas(beam_window, bg="#f0f4f8")
        scrollbar = tk.Scrollbar(beam_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f4f8")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        tk.Label(scrollable_frame, text="Select a Beam for All Files", font=("Arial", 12, "bold"), bg="#f0f4f8").grid(row=0, column=0, pady=10, padx=10, sticky="w")

        min_beam_count = min(len(beams) for _, beams in self.beam_data)
        self.beam_var = tk.StringVar(value="0")
        tk.Label(scrollable_frame, text="Select Beam for All Plans", bg="#f0f4f8").grid(row=1, column=0, pady=5, padx=10, sticky="w")
        beam_menu = ttk.Combobox(scrollable_frame, textvariable=self.beam_var, values=[f"Beam {i}" for i in range(min_beam_count)])
        beam_menu.grid(row=2, column=0, pady=5, padx=10, sticky="ew")

        for idx, (filename, beams) in enumerate(self.beam_data, start=3):
            tk.Label(scrollable_frame, text=f"File: {filename} (Beams: {len(beams)})", bg="#f0f4f8").grid(row=idx, column=0, pady=5, padx=10, sticky="w")

        confirm_button = tk.Button(scrollable_frame, text="Confirm", command=lambda: self.plot_selected_beams(beam_window), bg="#4682b4", fg="white")
        confirm_button.grid(row=idx + 1, column=0, pady=20, padx=10, sticky="s")

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        beam_window.grid_rowconfigure(0, weight=1)
        beam_window.grid_columnconfigure(0, weight=1)

        self.bind_mouse_wheel(canvas)

        beam_window.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def plot_selected_beams(self, beam_window):
        def plot_wrapper():
            selected_layer_data = []
            max_layers = 0
            beam_idx = int(self.beam_var.get().split()[-1])
            for filename, beams in self.beam_data:
                if beam_idx < len(beams):
                    layers = beams[beam_idx]
                    selected_layer_data.append((layers, filename))
                    max_layers = max(max_layers, len(layers))

            spot_size_multiplier = self.spot_size_slider.get()
            plot_type = self.plot_type_var.get()
            equal_axes = self.equal_axes_var.get()
            compare_weights = self.compare_weights_var.get()
            decimal_places = self.decimal_places_var.get()
            show_spot_numbers = self.show_spot_numbers_var.get()
            show_spot_weights = self.show_spot_weights_var.get()
            try:
                tolerance = float(self.tolerance_entry.get()) / 100
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid tolerance percentage!")
                return

            plot_spot_distribution(selected_layer_data, max_layers, spot_size_multiplier, plot_type, equal_axes, compare_weights, tolerance, decimal_places, show_spot_numbers, show_spot_weights)

        beam_window.destroy()
        self.redirect_output(plot_wrapper)

    def convert_to_csv(self):
        self.redirect_output(convert_to_csv, self.files, self.file_types)

    def merge_repaint_spots(self):
        if not self.files:
            messagebox.showerror("Error", "Please select DICOM files first!")
            return

        valid_files = []
        for file_path, file_type in zip(self.files, self.file_types):
            if file_type != "dcm":
                messagebox.showinfo("Info", f"Skipping {os.path.basename(file_path)}: Only DICOM files (.dcm) are supported for merging repaint spots.")
                continue

            try:
                ds = pydicom.dcmread(file_path)
                if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
                    messagebox.showwarning("Invalid File", f"{os.path.basename(file_path)} is not an RT Ion Plan file (SOPClassUID ≠ '1.2.840.10008.5.1.4.1.1.481.8').\nPlease select valid RT Ion Plan DICOM files.")
                    self.files = []
                    self.file_types = []
                    self.file_count_label.config(text="Selected files: 0")
                    self.file_list_text.config(state=tk.NORMAL)
                    self.file_list_text.delete(1.0, tk.END)
                    self.file_list_text.insert(tk.END, "No files loaded yet.")
                    self.file_list_text.config(state=tk.DISABLED)
                    return
                valid_files.append(file_path)
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to read {os.path.basename(file_path)}: {e}")
                continue

        if not valid_files:
            messagebox.showerror("Error", "No valid RT Ion Plan DICOM files selected!")
            return

        def merge_wrapper():
            for file_path in valid_files:
                result = merge_repaint_spots(file_path, show=False, quick=True)
                if result:
                    filename, output_file, stats = result
                    messagebox.showinfo("Success", f"Merged repaint spots for {filename}\nSaved as: {output_file}\n"
                                                   f"MU: {stats['MU']}, Beams: {stats['beams']}, Layers: {stats['layers']}\n"
                                                   f"Spots Before: {stats['spots_before']}, Spots After: {stats['spots_after']}")
                else:
                    messagebox.showwarning("Warning", f"Failed to merge repaint spots for {os.path.basename(file_path)}.")

        self.redirect_output(merge_wrapper)

    def split_repaint_spots(self):
        if not self.files:
            messagebox.showerror("Error", "Please select DICOM files first!")
            return

        valid_files = []
        for file_path, file_type in zip(self.files, self.file_types):
            if file_type != "dcm":
                messagebox.showinfo("Info", f"Skipping {os.path.basename(file_path)}: Only DICOM files (.dcm) are supported for splitting repaint spots.")
                continue

            try:
                ds = pydicom.dcmread(file_path)
                if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
                    messagebox.showwarning("Invalid File", f"{os.path.basename(file_path)} is not an RT Ion Plan file (SOPClassUID ≠ '1.2.840.10008.5.1.4.1.1.481.8').\nPlease select valid RT Ion Plan DICOM files.")
                    self.files = []
                    self.file_types = []
                    self.file_count_label.config(text="Selected files: 0")
                    self.file_list_text.config(state=tk.NORMAL)
                    self.file_list_text.delete(1.0, tk.END)
                    self.file_list_text.insert(tk.END, "No files loaded yet.")
                    self.file_list_text.config(state=tk.DISABLED)
                    return
                valid_files.append(file_path)
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to read {os.path.basename(file_path)}: {e}")
                continue

        if not valid_files:
            messagebox.showerror("Error", "No valid RT Ion Plan DICOM files selected!")
            return

        def split_wrapper():
            for file_path in valid_files:
                result = split_repaint_spots(file_path, show=False, quick=True)
                if result:
                    filename, output_file = result
                    messagebox.showinfo("Success", f"Split repaint spots for {filename}\nSaved as: {output_file}")
                else:
                    messagebox.showwarning("Warning", f"Failed to split repaint spots for {os.path.basename(file_path)}.")

        self.redirect_output(split_wrapper)

    def show_about(self):
        messagebox.showinfo("About", "RT Ion Plan Analysis\nVersion 1.0\nDeveloped by Chunbo Liu, PhD\n© 2025")

if __name__ == "__main__":
    root = tk.Tk()
    app = RTIonSpotComparer(root)
    root.mainloop()