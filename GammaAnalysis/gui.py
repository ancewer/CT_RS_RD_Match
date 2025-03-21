import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from functions import GammaCalc
import pydicom

class GammaCalcApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Gamma Pass Rate Calculator")
        self.root.geometry("500x700")

        self.dicom_files = []
        self.create_widgets()

    def create_widgets(self):
        # 文件选择部分
        tk.Label(self.root, text="Select DICOM Files (First file as Reference)", font=("Arial", 12)).pack(pady=10)
        tk.Button(self.root, text="Browse Files", command=self.browse_files).pack(pady=5)
        self.file_listbox = tk.Listbox(self.root, height=5, width=50)
        self.file_listbox.pack(pady=5)

        # Gamma 参数设置
        tk.Label(self.root, text="Gamma Calculation Options", font=("Arial", 12)).pack(pady=10)

        # Dose Percent Threshold
        tk.Label(self.root, text="Dose Percent Threshold (%)").pack(pady=5)
        self.dose_percent_var = tk.DoubleVar(value=2.0)
        tk.Entry(self.root, textvariable=self.dose_percent_var, width=10).pack(pady=2)

        # Distance MM Threshold
        tk.Label(self.root, text="Distance MM Threshold (mm)").pack(pady=5)
        self.distance_mm_var = tk.DoubleVar(value=2.0)
        tk.Entry(self.root, textvariable=self.distance_mm_var, width=10).pack(pady=2)

        # Lower Percent Dose Cutoff
        tk.Label(self.root, text="Lower Percent Dose Cutoff (%)").pack(pady=5)
        self.lower_cutoff_var = tk.DoubleVar(value=10.0)
        tk.Entry(self.root, textvariable=self.lower_cutoff_var, width=10).pack(pady=2)

        # Local Gamma
        tk.Label(self.root, text="Local Gamma").pack(pady=5)
        self.local_gamma_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Enable Local Gamma", variable=self.local_gamma_var).pack(pady=2)

        # Max Gamma
        tk.Label(self.root, text="Max Gamma").pack(pady=5)
        self.max_gamma_var = tk.DoubleVar(value=2.0)
        tk.Entry(self.root, textvariable=self.max_gamma_var, width=10).pack(pady=2)

        # 计算按钮
        tk.Button(self.root, text="Calculate Gamma Pass Rates", command=self.calculate_gamma).pack(pady=20)

        # 结果显示
        self.result_text = tk.Text(self.root, height=5, width=50)
        self.result_text.pack(pady=10)

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select DICOM Files",
            filetypes=[("DICOM files", "*.dcm")],
            parent=self.root
        )
        if files:
            self.dicom_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for file in self.dicom_files:
                self.file_listbox.insert(tk.END, file)
        else:
            messagebox.showwarning("No Selection", "No files selected!")

    def calculate_gamma(self):
        if len(self.dicom_files) < 2:
            messagebox.showerror("Error", "Please select at least two DICOM files (first as reference)!")
            return

        try:
            # 获取参数
            dose_percent = self.dose_percent_var.get()
            distance_mm = self.distance_mm_var.get()
            lower_cutoff = self.lower_cutoff_var.get()
            local_gamma = self.local_gamma_var.get()
            max_gamma = self.max_gamma_var.get()

            # 读取文件
            ref_file = self.dicom_files[0]
            evl_files = self.dicom_files[1:]

            ref_ds = pydicom.read_file(ref_file)
            axe_ref, dose_ref = pymedphys.dicom.zyx_and_dose_from_dataset(ref_ds)

            # 清空结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Gamma Pass Rates:\n")

            # 对每个评估文件计算 Gamma
            for evl_file in evl_files:
                evl_ds = pydicom.read_file(evl_file)
                axe_evl, dose_evl = pymedphys.dicom.zyx_and_dose_from_dataset(evl_ds)

                pass_ratio = GammaCalc(axe_ref, dose_ref, axe_evl, dose_evl,
                                       dose_percent, distance_mm, lower_cutoff, local_gamma, max_gamma)

                result = f"{evl_file}:\n  Pass Ratio: {pass_ratio:.1f}%\n"
                self.result_text.insert(tk.END, result)

        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")


if __name__ == "__main__":
    import pymedphys  # 确保在主程序中导入

    root = tk.Tk()
    app = GammaCalcApp(root)
    root.mainloop()