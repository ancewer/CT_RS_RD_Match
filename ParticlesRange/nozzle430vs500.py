import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def mat_to_npy(mat_file_path, npy_file_path=None, variable_name=None):
    """
    Convert MATLAB .mat file to Python .npy file.

    Parameters:
        mat_file_path (str): Path to input .mat file.
        npy_file_path (str, optional): Path to save .npy file.
        variable_name (str, optional): Name of variable to convert.

    Raises:
        ValueError: If no valid variables in MAT file.
        KeyError: If specified variable not found in MAT file.
    """
    mat_data = loadmat(mat_file_path)

    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    if not mat_data:
        raise ValueError("no valid variables in MAT file")

    if variable_name is None:
        variable_name = list(mat_data.keys())[0]
        print(f"default {variable_name}")

    if variable_name not in mat_data:
        raise KeyError(f"{variable_name} was not found, only has {list(mat_data.keys())}")

    data = mat_data[variable_name]
    data = np.asarray(data)

    if npy_file_path is None:
        npy_file_path = mat_file_path.replace('.mat', '.npy')

    np.save(npy_file_path, data)
    print(f"{variable_name} was saved as {npy_file_path}")


def calculate_r80(depth, dose):
    """
    Calculate R80: the depth where dose falls to 80% of the maximum on the distal side.

    Parameters:
    - depth (array): Depth values in cm.
    - dose (array): Dose values in Gy.

    Returns:
    - r80 (float): Depth (cm) at R80.
    - dose_r80 (float): Dose (Gy) at R80.
    """
    # Ensure that dose is a 1D array (not a 2D array)
    dose = np.ravel(dose)

    # Find the maximum dose and its depth (Bragg peak)
    max_dose_idx = np.argmax(dose)
    max_dose = dose[max_dose_idx]
    max_depth = depth[max_dose_idx]

    # Target dose is 80% of maximum
    target_dose = 0.8 * max_dose

    # Search on the distal side (depths > max_depth)
    distal_indices = np.where(depth > max_depth)[0]
    for i in distal_indices:
        # Ensure dose[i] is a scalar (single value)
        if dose[i] <= target_dose:
            # Interpolate between this point and the previous point
            x0, x1 = depth[i - 1], depth[i]
            y0, y1 = dose[i - 1], dose[i]
            # Linear interpolation: r80 = x0 + (x1 - x0) * (target_dose - y0) / (y1 - y0)
            r80 = x0 + (x1 - x0) * (target_dose - y0) / (y1 - y0)
            dose_r80 = target_dose
            return r80, dose_r80

    raise ValueError("R80 not found within the data range.")

# mat_to_npy('data/BPLibrary_50to280_420.mat', npy_file_path='data/BPLibrary_50to280_420.npy', variable_name=None)
# mat_to_npy('data/BPLibrary_50to280_500.mat', npy_file_path='data/BPLibrary_50to280_500.npy', variable_name=None)

# 加载两个 .mat 文件
data_420 = scipy.io.loadmat('data/BPLibrary_50to280_420.mat')
data_500 = scipy.io.loadmat('data/BPLibrary_50to280_500.mat')
# TODO: 通过比较，目前认为这两个数据没什么区别，0,1,2 -> ic diameter:[81.6 120 200]
#  对于不同测量直径结果也没什么区别，对不同nozzle距离也没什么区别，20250417，等待后续发现


# 假设 'MCdose' 是包含深度和 IDD 剂量数据的数组
# 深度在第一列，IDD 剂量在接下来的列中
depth_420 = data_420['MCdose'][:, 0][:,0]  # 第一列是深度
dose_420 = data_420['MCdose'][:, 1:]  # 其他列是不同能量的剂量

depth_500 = data_500['MCdose'][:, 0][:,0]  # 第一列是深度
dose_500 = data_500['MCdose'][:, 1:]  # 其他列是不同能量的剂量

# 假设要比较的能量索引（例如第3列对应某一能量）
energy_index = 500  # 选择要比较的能量列索引，按实际需要调整

# 获取指定能量的 IDD 曲线
# 0,1,2 -> ic diameter:[81.6 120 200]
idd_420 = dose_420[:, energy_index][:,0]
idd_500 = dose_500[:, energy_index][:,2]

# 绘制 IDD 曲线
plt.figure(figsize=(8, 6))
plt.plot(depth_420, idd_420, '-r', label='BPLibrary_420')
plt.plot(depth_500, idd_500, '-b', label='BPLibrary_500')
r80, dose_r80 = calculate_r80(depth_420, idd_420)
print(f"R80: {r80:.3f} cm, Dose at R80: {dose_r80:.2e} Gy")

# Set plot limits
xlim = (0, r80 + 5)  # Extend x-axis to R80 + 5 cm
plt.xlim(xlim)

plt.xlabel('Depth (mm)')
plt.ylabel('IDD (Gy)')
plt.legend()
plt.title('Comparison of IDD Curves for Same Energy')
plt.grid(True)
plt.show()
