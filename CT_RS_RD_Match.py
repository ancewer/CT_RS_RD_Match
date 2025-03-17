import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端
import os
import glob
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.ndimage import map_coordinates
from matplotlib.widgets import Slider

def load_ct_images(ct_folder):
    """读取CT DICOM序列，并按照InstanceNumber排序"""
    ct_files = sorted(
        [os.path.join(ct_folder, f) for f in os.listdir(ct_folder) if f.endswith(".dcm") and pydicom.dcmread(os.path.join(ct_folder, f), stop_before_pixels=True).Modality=='CT'],
        key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)
    )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ct_files)
    ct_image = reader.Execute()
    ct_array = sitk.GetArrayFromImage(ct_image)
    ct_origin = np.array(ct_image.GetOrigin())
    ct_spacing = np.array(ct_image.GetSpacing())
    ct_direction = np.array(ct_image.GetDirection()).reshape(3, 3)
    return ct_array, ct_origin, ct_spacing, ct_direction

def load_rt_structure(rt_file):
    """读取RT Structure文件，并提取ROI轮廓"""
    ds = pydicom.dcmread(rt_file)
    structures = {roi.ROIName: roi.ROINumber for roi in ds.StructureSetROISequence}
    contours = {}
    for roi in ds.ROIContourSequence:
        roi_name = next((name for name, num in structures.items() if num == roi.ReferencedROINumber), None)
        if roi_name and hasattr(roi, "ContourSequence"):
            contours[roi_name] = [np.array(contour.ContourData).reshape(-1, 3) for contour in roi.ContourSequence]
    return contours

def convert_contours_to_mask(ct_array, ct_origin, ct_spacing, contours, roi_name):
    """将RT Structure轮廓转换为二进制掩码"""
    if roi_name not in contours:
        raise ValueError(f"ROI '{roi_name}' not found in RT Structure.")
    mask = np.zeros_like(ct_array, dtype=np.uint8)
    for points in contours[roi_name]:
        voxel_indices = np.round((points - ct_origin) / ct_spacing).astype(int)
        for slice_idx in np.unique(voxel_indices[:, 2]):
            slice_points = voxel_indices[voxel_indices[:, 2] == slice_idx]
            if len(slice_points) < 3:
                continue
            rr, cc = polygon(slice_points[:, 1], slice_points[:, 0], shape=mask.shape[1:])
            mask[slice_idx, rr, cc] = 1
    mask = np.flip(mask, axis=0)
    return mask

def load_dose(dose_file):
    """读取 RT Dose 文件，并返回剂量矩阵"""
    ds = pydicom.dcmread(dose_file)
    dose_array = ds.pixel_array * ds.DoseGridScaling
    dose_origin = np.array(ds.ImagePositionPatient, dtype=np.float64)
    dose_spacing = np.array(list(ds.PixelSpacing) + [ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]],
                            dtype=np.float64)
    # 剂量网格orgin的z开始点与ct相反
    dose_origin[2] = dose_origin[2] + (dose_spacing[2] * (dose_array.shape[0] - 1))
    dose_array=np.flip(dose_array, axis=0)
    return dose_array, dose_origin, dose_spacing

def load_dose_itk(dose_file):
    """
    使用 SimpleITK 读取 DICOM 剂量文件，并返回剂量矩阵及其空间信息
    :param dose_file: DICOM 剂量文件路径
    :return: dose_array (numpy array), dose_origin, dose_spacing
    """
    # **读取 DICOM 剂量文件**
    dose_image = sitk.ReadImage(dose_file)

    # **转换为 NumPy 数组**
    dose_array = sitk.GetArrayFromImage(dose_image)

    # **提取 DICOM 坐标信息**
    dose_origin = np.array(dose_image.GetOrigin())  # (X, Y, Z) 物理原点
    dose_spacing = np.array(dose_image.GetSpacing())  # (X, Y, Z) 体素间距
    dose_direction = np.array(dose_image.GetDirection()).reshape(3, 3)  # 方向矩阵
    return dose_array, dose_origin, dose_spacing, dose_direction

def resample_to_dose_grid(source_array, source_origin, source_spacing, dose_array, dose_origin, dose_spacing, is_mask):
    # 强制让z坐标为递增（否则map_coordinate处理不了）
    # print(source_origin, dose_origin)
    source_origin = source_origin.copy()
    dose_origin = dose_origin.copy()
    source_origin[-1] = -source_origin[-1]
    dose_origin[-1] = -dose_origin[-1]
    # print(source_origin, dose_origin)

    """将 CT 或掩码数据重采样到与 Dose 相同的坐标系"""
    dose_shape = dose_array.shape
    z =  np.arange(dose_shape[0]) * dose_spacing[2] + dose_origin[2]
    y = np.arange(dose_shape[1]) * dose_spacing[1] + dose_origin[1]
    x = np.arange(dose_shape[2]) * dose_spacing[0] + dose_origin[0]

    z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing="ij")

    # 将剂量坐标转换到源数据的索引坐标
    source_coords = np.array([
        (z_grid - source_origin[2]) / source_spacing[2],
        (y_grid - source_origin[1]) / source_spacing[1],
        (x_grid - source_origin[0]) / source_spacing[0]
    ]).reshape(3, -1)  # Shape: (3, N)

    # 检查是否超出 source_array 范围
    source_shape = source_array.shape
    coord_mins = np.min(source_coords, axis=1)
    coord_maxs = np.max(source_coords, axis=1)
    print(f"✅Source coords min: {coord_mins}, max: {coord_maxs}")
    print(f"✅Source array shape: {source_shape}")

    # **选择插值模式**
    # 格外注意：map_coordinates不能处理坐标索引为负值的情况
    if is_mask:
        # 对于掩码，使用最近邻插值，防止 0/1 变成小数
        resampled_array = map_coordinates(source_array, source_coords, order=0, mode='grid-constant', cval=0).reshape(
            dose_shape)
    else:
        # CT/Dose 数据使用线性插值
        resampled_array = map_coordinates(source_array, source_coords, order=1, mode='nearest').reshape(dose_shape)
    x=0
    return resampled_array

def plot_ct_contour_dose_interactive_best1(ct_array, mask, dose_array, window_level=0, window_width=1000):
    # attention: matplotlib imshow function alpha_array works for python 3.10, failed for python 3.12
    ct_array = np.flip(ct_array, axis=1)
    mask = np.flip(mask, axis=1)
    dose_array = np.flip(dose_array, axis=1)

    # 计算 CT 的灰度范围
    vmin = window_level - window_width / 2
    vmax = window_level + window_width / 2
    """交互式绘制 CT + ROI 轮廓 + 剂量叠加图（支持滚动翻层）"""
    if ct_array.shape[0] != mask.shape[0] or ct_array.shape[0] != dose_array.shape[0]:
        raise ValueError("ct_array, mask, and dose_array must have the same number of slices.")

    # 找到 ROI 覆盖最多的切片
    default_slice = np.argmax(np.sum(mask, axis=(1, 2)))

    # 创建 Figure 和 Axes
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)  # 预留空间给 Slider

    # 显示默认 CT 切片
    # ct_min, ct_max = np.min(ct_array), np.max(ct_array)
    # print(ct_min, ct_max)
    img_ct = ax.imshow(ct_array[default_slice], cmap='gray', origin='lower', aspect='auto', zorder=0)
    # print(np.min(ct_array[default_slice]), np.max(ct_array[default_slice]))
    img_ct.set_clim(vmin=vmin, vmax=vmax)

    # 绘制初始 ROI 轮廓
    mask_slice = mask[default_slice]
    contour_obj = ax.contour(mask_slice, colors='r', linewidths=1, zorder=1)

    # 绘制初始剂量分布（使用 imshow）
    dose_slice = dose_array[default_slice]
    # print(f"Dose slice min: {np.min(dose_slice)}, max: {np.max(dose_slice)}")  # 调试剂量值范围
    alpha_array = np.where(dose_slice > 0, 0.5, 0)  # 剂量 > 0 时 alpha=0.3，否则 0
    # print(f"Alpha array min: {np.min(alpha_array)}, max: {np.max(alpha_array)}")  # 调试透明度范围
    dose_img = ax.imshow(dose_slice, alpha=alpha_array, cmap='jet', origin='lower', aspect='auto', zorder=2)
    dose_img.set_clim(vmin=np.min(dose_slice), vmax=np.max(dose_slice))

    # 添加 colorbar
    cbar = plt.colorbar(dose_img, ax=ax, label='Dose (Gy)')

    ax.set_title(f'CT + Contour + Dose (Slice {default_slice})')
    ax.set_xlabel("X-axis (pixels)")
    ax.set_ylabel("Y-axis (pixels)")

    # 添加滑动条 (Slider)
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])  # 滑动条位置
    slider = Slider(ax_slider, 'Slice', 0, ct_array.shape[0] - 1,valinit=default_slice, valfmt='%d', valstep=1)

    # 添加窗宽调节滑动条
    ax_width_slider = plt.axes([0.2, 0.07, 0.6, 0.03])  # 窗宽滑块位置
    width_slider = Slider(ax_width_slider, 'Window Width', 100, 2000, valinit=window_width, valstep=10)

    # 添加窗位调节滑动条
    ax_level_slider = plt.axes([0.2, 0.12, 0.6, 0.03])  # 窗位滑块位置
    level_slider = Slider(ax_level_slider, 'Window Level', -1000, 1000, valinit=window_level, valstep=10)

    def update(val):
        """更新切片图像、ROI 轮廓和剂量分布"""
        nonlocal contour_obj,dose_img

        slice_idx = int(slider.val)
        new_window_width = int(width_slider.val)
        new_window_level = int(level_slider.val)

        # 更新 CT 图像
        img_ct.set_data(ct_array[slice_idx])
        img_ct.set_clim(vmin=new_window_level - new_window_width / 2,
                        vmax=new_window_level + new_window_width / 2)
        # print(np.min(ct_array[slice_idx]), np.max(ct_array[slice_idx]))

        # 移除旧的 ROI 轮廓
        contour_obj.remove()
        new_mask_slice = mask[slice_idx]
        new_contour_obj = ax.contour(new_mask_slice, colors='r', linewidths=1)

        # 更新剂量分布，这样虽然可以更新剂量，但是不能设置alpha_array
        # new_dose_slice = dose_array[slice_idx]
        # dose_img.set_data(new_dose_slice)
        # 移除旧的剂量图像并重新绘制，牺牲性能，但是能满足透明度设置要求
        dose_img.remove()
        new_dose_slice = dose_array[slice_idx]
        new_alpha_array = np.where(new_dose_slice > 0, 0.5, 0)  # 重新计算 alpha
        dose_img = ax.imshow(new_dose_slice, alpha=new_alpha_array, cmap='jet', origin='lower', aspect='auto')
        dose_min = np.min(new_dose_slice)
        dose_max = np.max(new_dose_slice)
        if dose_max == dose_min:
            dose_max = dose_min + 1e-6
        dose_img.set_clim(vmin=dose_min, vmax=dose_max)

        # 更新 colorbar
        cbar.update_normal(dose_img)  # 刷新 colorbar

        # 更新 contour objects
        contour_obj = new_contour_obj

        #更新标题
        # ax.set_title(f'CT + Contour + Dose (Slice {slice_idx})')
        # 更新标题
        ax.set_title(f'CT + Contour + Dose (Slice {slice_idx}, WW={new_window_width}, WL={new_window_level})')

        fig.canvas.draw_idle()

    # 绑定滑动条事件
    slider.on_changed(update)
    width_slider.on_changed(update)
    level_slider.on_changed(update)

    plt.show()


def plot_ct_contour_dose_interactive_best2(ct_array, mask, dose_array, window_width=1000, window_level=0):
    """交互式绘制 CT + ROI 轮廓 + 剂量叠加图（支持滚动翻层）

    参数：
        ct_array: CT 数据数组
        mask: ROI 掩膜数组（将上下颠倒）
        dose_array: 剂量分布数组
        window_width: CT 图像的窗口宽度 (默认 1000)
        window_level: CT 图像的窗口水平 (默认 0)
    """
    if ct_array.shape[0] != mask.shape[0] or ct_array.shape[0] != dose_array.shape[0]:
        raise ValueError("ct_array, mask, and dose_array must have the same number of slices.")

    # 计算 CT 的灰度范围
    vmin = window_level - window_width / 2
    vmax = window_level + window_width / 2

    # 找到 ROI 覆盖最多的切片
    default_slice = np.argmax(np.sum(mask, axis=(1, 2)))

    # 创建 Figure 和 Axes
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)  # 预留空间给 Slider

    # 显示默认 CT 切片（灰度图）
    img_ct = ax.imshow(ct_array[default_slice], cmap='gray', origin='lower', aspect='auto')
    img_ct.set_clim(vmin=vmin, vmax=vmax)  # 使用窗口设置灰度范围

    # 绘制初始 ROI 轮廓（mask 上下颠倒）
    mask_slice = np.flip(mask[default_slice], axis=0)  # 翻转 y 轴
    contour_obj = ax.contour(mask_slice, colors='r', linewidths=1)

    # 绘制初始剂量分布（分层 contourf，动态 alpha）
    dose_slice = dose_array[default_slice]
    dose_min = np.min(dose_slice[dose_slice > 0]) if np.any(dose_slice > 0) else 0
    dose_max = np.max(dose_slice)
    if dose_max == dose_min:
        dose_max = dose_min + 1e-6
    # 定义剂量级别和对应的 alpha 值
    levels = np.linspace(dose_min, dose_max, 5)  # 5 个级别，可以调整
    alphas = [0.0, 0.1, 0.3, 0.5, 0.7]  # 对应透明度，从低到高
    dose_contour_objs = []  # 存储多个 contourf 对象
    for i in range(len(levels) - 1):
        masked_dose = np.ma.masked_outside(dose_slice, levels[i], levels[i + 1])
        contour_obj_layer = ax.contourf(masked_dose, levels=[levels[i], levels[i + 1]],
                                        alpha=alphas[i], cmap='jet')
        dose_contour_objs.append(contour_obj_layer)

    # 添加 colorbar（绑定到最后一个 contourf 对象）
    cbar = plt.colorbar(dose_contour_objs[-1], ax=ax, label='Dose (Gy)')

    ax.set_title(f'CT + Contour + Dose (Slice {default_slice})')
    ax.set_xlabel("X-axis (pixels)")
    ax.set_ylabel("Y-axis (pixels)")

    # 添加滑动条 (Slider)
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])  # 滑动条位置
    slider = Slider(ax_slider, 'Slice', 0, ct_array.shape[0] - 1,
                    valinit=default_slice, valfmt='%d', valstep=1)

    def update(val):
        """更新切片图像、ROI 轮廓和剂量分布"""
        nonlocal contour_obj
        nonlocal cbar

        slice_idx = int(slider.val)

        # 更新 CT 图像（灰度图）
        img_ct.set_data(ct_array[slice_idx])
        img_ct.set_clim(vmin=vmin, vmax=vmax)  # 保持窗口设置

        # 移除旧的 ROI 轮廓
        contour_obj.remove()
        new_mask_slice = np.flip(mask[slice_idx], axis=0)  # 翻转 y 轴
        new_contour_obj = ax.contour(new_mask_slice, colors='r', linewidths=1)

        # 移除旧的剂量分布层
        for obj in dose_contour_objs:
            obj.remove()
        dose_contour_objs.clear()
        new_dose_slice = dose_array[slice_idx]
        dose_min = np.min(new_dose_slice[new_dose_slice > 0]) if np.any(new_dose_slice > 0) else 0
        dose_max = np.max(new_dose_slice)
        if dose_max == dose_min:
            dose_max = dose_min + 1e-6
        # 重新绘制分层 contourf
        levels = np.linspace(dose_min, dose_max, 5)
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7]
        for i in range(len(levels) - 1):
            masked_dose = np.ma.masked_outside(new_dose_slice, levels[i], levels[i + 1])
            contour_obj_layer = ax.contourf(masked_dose, levels=[levels[i], levels[i + 1]],
                                            alpha=alphas[i], cmap='jet')
            dose_contour_objs.append(contour_obj_layer)

        # 移除旧的 colorbar
        cbar.remove()
        # 创建新的 colorbar
        cbar = plt.colorbar(dose_contour_objs[-1], ax=ax, label='Dose (Gy)')

        # 更新 contour objects
        contour_obj = new_contour_obj

        # 更新标题
        ax.set_title(f'CT + Contour + Dose (Slice {slice_idx})')

        fig.canvas.draw_idle()

    # 绑定滑动条事件
    slider.on_changed(update)

    plt.show()

def plot_ct_contour_dose_interactive_best3(ct_array, mask, dose_array):
    """交互式绘制 CT + ROI 轮廓 + 剂量叠加图（支持滚动翻层）"""
    if ct_array.shape[0] != mask.shape[0] or ct_array.shape[0] != dose_array.shape[0]:
        raise ValueError("ct_array, mask, and dose_array must have the same number of slices.")

    # 找到 ROI 覆盖最多的切片
    default_slice = np.argmax(np.sum(mask, axis=(1, 2)))

    # 创建 Figure 和 Axes
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)  # 预留空间给 Slider

    # 显示默认 CT 切片
    img_ct = ax.imshow(ct_array[default_slice], cmap='gray', origin='lower', aspect='auto')
    img_ct.set_clim(vmin=np.min(ct_array), vmax=np.max(ct_array))  # 防止 CT 图像消失

    # 绘制初始 ROI 轮廓
    mask_slice = mask[default_slice]
    contour_obj = ax.contour(mask_slice, colors='r', linewidths=1)

    # 绘制初始剂量分布
    dose_slice = dose_array[default_slice]
    dose_min = np.min(dose_slice)
    dose_max = np.max(dose_slice)
    dose_contour_obj = ax.contourf(dose_slice, alpha=0.3, cmap='jet', vmin=dose_min, vmax=dose_max)
    # 添加 colorbar
    cbar = plt.colorbar(dose_contour_obj, ax=ax, label='Dose (Gy)')

    ax.set_title(f'CT + Contour + Dose (Slice {default_slice})')
    ax.set_xlabel("X-axis (pixels)")
    ax.set_ylabel("Y-axis (pixels)")

    # 添加滑动条 (Slider)
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])  # 滑动条位置
    slider = Slider(ax_slider, 'Slice', 0, ct_array.shape[0] - 1,
                    valinit=default_slice, valfmt='%d', valstep=1)

    def update(val):
        """更新切片图像、ROI 轮廓和剂量分布"""
        nonlocal contour_obj, dose_contour_obj, cbar
        slice_idx = int(slider.val)

        # 更新 CT 图像
        img_ct.set_data(ct_array[slice_idx])

        # 移除旧的 ROI 轮廓
        contour_obj.remove()
        new_mask_slice = mask[slice_idx]
        new_contour_obj = ax.contour(new_mask_slice, colors='r', linewidths=1)

        # 移除旧的剂量分布
        dose_contour_obj.remove()
        new_dose_slice = dose_array[slice_idx]
        dose_min = np.min(new_dose_slice)
        dose_max = np.max(new_dose_slice)
        # 防止 dose_min 和 dose_max 相等
        if dose_max == dose_min:
            dose_max = dose_min + 1e-6
        new_dose_contour_obj = ax.contourf(new_dose_slice, alpha=0.5, cmap='jet',
                                          vmin=dose_min, vmax=dose_max)

        # 更新 contour objects
        contour_obj = new_contour_obj
        dose_contour_obj = new_dose_contour_obj

        # 更新 colorbar 的范围
        cbar.mappable.set_clim(vmin=dose_min, vmax=dose_max)
        cbar.update_ticks()

        # 更新标题
        ax.set_title(f'CT + Contour + Dose (Slice {slice_idx})')

        fig.canvas.draw_idle()

    # 绑定滑动条事件
    slider.on_changed(update)

    plt.show()

def save_as_mhd(image_array, origin, spacing, filename):
    """
    使用 SimpleITK 保存 NumPy 数组为 .mhd/.raw 格式
    参数:
        image_array: 需要保存的 NumPy 3D 数组
        origin: 图像的坐标原点 (list or tuple)
        spacing: 图像的像素间距 (list or tuple)
        filename: 保存的文件名（.mhd）
    """

    # **1️⃣ 转换 NumPy 数组为 SimpleITK 图像**
    image = sitk.GetImageFromArray(image_array)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)

    # **2️⃣ 保存为 .mhd**
    sitk.WriteImage(image, filename)
    print(f"✅ 已保存: {filename}")

def resample_image_to_target(source_mhd, target_mhd, output_mhd, is_mask=False):
    """
    使用 SimpleITK 对 `source_mhd` 进行重采样，使其匹配 `target_mhd` 的 spacing、origin 和 size。
    参数:
        source_mhd: 源 .mhd 文件路径
        target_mhd: 目标 .mhd 文件路径
        output_mhd: 重采样后保存的 .mhd 文件路径
        is_mask: 是否为二值掩码，决定插值方式 (默认 False)
    """

    # **1️⃣ 读取源和目标图像**
    source_image = sitk.ReadImage(source_mhd)
    target_image = sitk.ReadImage(target_mhd)

    # **2️⃣ 设定重采样参数**
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_image.GetSize())         # 目标 size
    resampler.SetOutputSpacing(target_image.GetSpacing())  # 目标 spacing
    resampler.SetOutputOrigin(target_image.GetOrigin())    # 目标 origin
    resampler.SetOutputDirection(target_image.GetDirection())  # 目标方向
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetOutputPixelType(source_image.GetPixelIDValue())

    # **3️⃣ 选择插值方式**
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 近邻插值，保证二值性
    else:
        resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值，适用于 CT/Dose

    # **4️⃣ 进行重采样**
    resampled_image = resampler.Execute(source_image)

    # **5️⃣ 保存重采样后的 .mhd**
    sitk.WriteImage(resampled_image, output_mhd)
    print(f"✅ 重采样完成: {output_mhd}")

def compute_roi_volume(mask, spacing):
    """
    计算 ROI 体积（单位：mm³）
    参数:
        mask: ROI 掩码 (3D NumPy 数组, 0/1)
        spacing: 体素间距 (dx, dy, dz)，单位 mm
    返回:
        体积 (mm³) 和 体积 (cm³)
    """
    # **计算每个体素的体积**
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³

    # **统计 ROI 内的体素数**
    num_voxels = np.sum(mask)

    # **计算 ROI 体积**
    roi_volume_mm3 = num_voxels * voxel_volume  # mm³
    roi_volume_cm3 = roi_volume_mm3 / 1000  # cm³

    return roi_volume_cm3

def compute_physical_range(origin, spacing, shape, z_decreasing=True):
    """
    计算数据在物理空间中的范围 (X_min, X_max, Y_min, Y_max, Z_min, Z_max)
    注意：Z 方向通常是递减的
    """
    x_min, y_min, z_max = origin  # Z_max 直接是 origin[2]
    x_max = x_min + (shape[0] - 1) * spacing[0]
    y_max = y_min + (shape[1] - 1) * spacing[1]
    # **修正 Z 方向范围**
    if z_decreasing:
        z_min = z_max - (shape[2] - 1) * spacing[2]  # 方向递减
    else:
        z_min = z_max + (shape[2] - 1) * spacing[2]  # 方向递增（非常见情况）
    return (x_min, x_max, y_min, y_max, z_min, z_max)

def main(ct_folder, rt_structure_file, dose_file, roi_name, tmp_folder, write_mhd):
    """主函数：加载 CT、RT Structure、RT Dose 并进行可视化"""
    # 1️⃣ 读取 CT 影像
    ct_array, ct_origin, ct_spacing, _ = load_ct_images(ct_folder)
    # 2️⃣ 读取 RT Structure 并转换为掩码
    contours = load_rt_structure(rt_structure_file)
    mask = convert_contours_to_mask(ct_array, ct_origin, ct_spacing, contours, roi_name)

    # 3️⃣ 读取 RT Dose
    dose_array, dose_origin, dose_spacing = load_dose(dose_file)

    if write_mhd:
        save_as_mhd(ct_array, ct_origin, ct_spacing, os.path.join(tmp_folder,"ct_array.mhd"))
        save_as_mhd(dose_array, dose_origin, dose_spacing, os.path.join(tmp_folder, "dose_array.mhd"))
        save_as_mhd(mask, ct_origin, ct_spacing, os.path.join(tmp_folder, "mask_array.mhd"))

    # resample_image_to_target(os.path.join(tmp_folder,'ct_array.mhd'), os.path.join(tmp_folder,'dose_array.mhd'), os.path.join(tmp_folder,'ct_resampled.mhd'), is_mask=False)
    # 4️⃣ 重新采样 CT 和掩码到剂量坐标系
    print(f"✅ct_origin:{ct_origin},ct_spacing:{ct_spacing},ct_shape:{ct_array.shape[::-1]}")
    print(f"✅dose_origin:{dose_origin},dose_spacing:{dose_spacing},dose_shape:{dose_array.shape[::-1]}")
    ct_range = compute_physical_range(ct_origin,ct_spacing,ct_array.shape[::-1])
    dose_range = compute_physical_range(dose_origin,dose_spacing,dose_array.shape[::-1])
    print(f"✅CT 物理范围 (mm): X:[{ct_range[0]}, {ct_range[1]}], "f"Y:[{ct_range[2]}, {ct_range[3]}], Z:[{ct_range[4]}, {ct_range[5]}]")
    print(f"✅Dose 物理范围 (mm): X:[{dose_range[0]}, {dose_range[1]}], "f"Y:[{dose_range[2]}, {dose_range[3]}], Z:[{dose_range[4]}, {dose_range[5]}]")

    ct_resampled = resample_to_dose_grid(ct_array, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing,is_mask=False)
    print(f"✅Before Resampling, Mask Sum: {np.sum(mask)}")
    mask_resampled = resample_to_dose_grid(mask, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing, is_mask=True)
    print(f"✅After Resampling, Mask Sum: {np.sum(mask_resampled)}")  # 这里不应该是 0
    volume = compute_roi_volume(mask, ct_spacing)
    print(f"✅Before Resampling, {roi_name} volume: {volume}cc")
    volume = compute_roi_volume(mask_resampled, dose_spacing)
    print(f"✅After Resampling, {roi_name} volume: {volume}cc")
    if write_mhd:
        save_as_mhd(ct_resampled, dose_origin, dose_spacing, os.path.join(tmp_folder, "ct_resampled.mhd"))
        save_as_mhd(mask_resampled, dose_origin, dose_spacing, os.path.join(tmp_folder, "mask_resampled.mhd"))
    # 5️⃣ 进行可视化
    plot_ct_contour_dose_interactive_best1(ct_resampled, mask_resampled, dose_array)

# **示例调用**
if __name__ == "__main__":
    tmp_folder = 'Results'
    # 运行主函数
    id = '4862399'
    ct_folder = r"C:\Users\m313763\Desktop\Data\\" + id +  r"\DcmFiles"
    rs_file = glob.glob(r"C:\Users\m313763\Desktop\Data\\"+ id +r"\DcmFiles\RS."+ id +"*.dcm")[0]
    rd_file = r"C:\Users\m313763\Desktop\Data\\"+id+r"\RD."+id+".opt_d.dcm"
    target = "CtvLung"
    main(ct_folder, rs_file, rd_file, target, tmp_folder, write_mhd=True)
