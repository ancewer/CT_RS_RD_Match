import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from functions import extract_spot_data, plot_spot_distribution

class RTIonSpotComparer:
    def __init__(self, root):
        self.root = root
        self.root.title("RT Ion Plan Spot Distribution Comparer")
        self.root.geometry("400x550")  # 增加高度以容纳新控件

        self.dicom_files = []
        self.beam_data = []

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select RT Ion Plan DICOM Files", font=("Arial", 12)).pack(pady=10)
        tk.Button(self.root, text="Browse Files", command=self.browse_files).pack(pady=5)
        self.file_count_label = tk.Label(self.root, text="Selected files: 0", font=("Arial", 10))
        self.file_count_label.pack(pady=5)

        tk.Label(self.root, text="Spot Size Multiplier", font=("Arial", 10)).pack(pady=5)
        self.spot_size_slider = tk.Scale(self.root, from_=1, to=200, orient=tk.HORIZONTAL,
                                         resolution=10, length=200)
        self.spot_size_slider.set(50)
        self.spot_size_slider.pack(pady=5)

        tk.Label(self.root, text="Plot Type", font=("Arial", 10)).pack(pady=5)
        self.plot_type_var = tk.StringVar(value="line")
        tk.Radiobutton(self.root, text="Scatter Plot", variable=self.plot_type_var, value="scatter").pack(pady=2)
        tk.Radiobutton(self.root, text="Line Plot", variable=self.plot_type_var, value="line").pack(pady=2)

        tk.Label(self.root, text="Axes Options", font=("Arial", 10)).pack(pady=5)
        self.equal_axes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Equal X and Y Axes", variable=self.equal_axes_var).pack(pady=5)

        # 添加比较 Spot Weight 选项
        tk.Label(self.root, text="Spot Weight Comparison", font=("Arial", 10)).pack(pady=5)
        self.compare_weights_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Compare Spot Weights", variable=self.compare_weights_var).pack(pady=2)
        tk.Label(self.root, text="Tolerance (%):").pack(pady=2)
        self.tolerance_entry = tk.Entry(self.root, width=10)
        self.tolerance_entry.insert(0, "1")  # 默认 1%
        self.tolerance_entry.pack(pady=2)

        tk.Button(self.root, text="Plot Spot Distribution", command=self.select_beams).pack(pady=20)

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select RT Ion Plan DICOM Files",
            filetypes=[("DICOM files", "*.dcm")],
            parent=self.root
        )
        if files:
            self.dicom_files = list(files)
            self.file_count_label.config(text=f"Selected files: {len(self.dicom_files)}")
            self.beam_data = []
        else:
            messagebox.showwarning("No Selection", "No files selected!")

    def select_beams(self):
        if not self.dicom_files:
            messagebox.showerror("Error", "Please select DICOM files first!")
            return

        self.beam_data = []
        for dicom_file in self.dicom_files:
            beam_info = extract_spot_data(dicom_file)
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

        canvas = tk.Canvas(beam_window)
        scrollbar = tk.Scrollbar(beam_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        tk.Label(scrollable_frame, text="Select a Beam for All Files", font=("Arial", 12)).grid(row=0, column=0, pady=10, padx=10, sticky="w")

        min_beam_count = min(len(beams) for _, beams in self.beam_data)
        self.beam_var = tk.StringVar(value="0")
        tk.Label(scrollable_frame, text="Select Beam for All Plans").grid(row=1, column=0, pady=5, padx=10, sticky="w")
        beam_menu = ttk.Combobox(scrollable_frame, textvariable=self.beam_var,
                                 values=[f"Beam {i}" for i in range(min_beam_count)])
        beam_menu.grid(row=2, column=0, pady=5, padx=10, sticky="ew")

        for idx, (filename, beams) in enumerate(self.beam_data, start=3):
            tk.Label(scrollable_frame, text=f"File: {filename} (Beams: {len(beams)})").grid(row=idx, column=0, pady=5, padx=10, sticky="w")

        confirm_button = tk.Button(scrollable_frame, text="Confirm", command=lambda: self.plot_selected_beams(beam_window))
        confirm_button.grid(row=idx + 1, column=0, pady=20, padx=10, sticky="s")

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        beam_window.grid_rowconfigure(0, weight=1)
        beam_window.grid_columnconfigure(0, weight=1)

        beam_window.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def plot_selected_beams(self, beam_window):
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
        try:
            tolerance = float(self.tolerance_entry.get()) / 100  # 转换为小数
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid tolerance percentage!")
            return

        beam_window.destroy()
        plot_spot_distribution(selected_layer_data, max_layers, spot_size_multiplier, plot_type, equal_axes, compare_weights, tolerance)

if __name__ == "__main__":
    root = tk.Tk()
    app = RTIonSpotComparer(root)
    root.mainloop()