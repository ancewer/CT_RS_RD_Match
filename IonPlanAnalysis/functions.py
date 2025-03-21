import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter import messagebox
from datetime import datetime
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import os, copy


def extract_spot_data(dicom_file):
    """从 RT Ion Plan DICOM 文件中按 Beam 提取 spot 数据，处理单个 spot 的 weights"""
    try:
        ds = pydicom.dcmread(dicom_file)
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
            return None

        filename = os.path.basename(dicom_file)
        beams = []
        for ii, ion_beam in enumerate(ds.IonBeamSequence):
            if hasattr(ion_beam, 'TreatmentDeliveryType') and ion_beam.TreatmentDeliveryType != 'TREATMENT':
                continue
            scale_to_mu = float(ds.FractionGroupSequence[0].ReferencedBeamSequence[ii].BeamMeterset) / float(ion_beam.FinalCumulativeMetersetWeight)
            layers = []
            for i in range(0, len(ion_beam.IonControlPointSequence), 2):
                control_point = ion_beam.IonControlPointSequence[i]
                if hasattr(control_point, 'ScanSpotPositionMap'):
                    spots_x = []
                    spots_y = []
                    spots_weight = []
                    positions = control_point.ScanSpotPositionMap
                    weights = control_point.ScanSpotMetersetWeights
                    spot_count = len(positions) // 2
                    if spot_count == 1 and not isinstance(weights, (list, tuple, np.ndarray)):
                        spots_x.append(positions[0])
                        spots_y.append(positions[1])
                        spots_weight.append(weights)
                    else:
                        for j in range(0, len(positions), 2):
                            spots_x.append(positions[j])
                            spots_y.append(positions[j + 1])
                            spots_weight.append(weights[j // 2])
                    energy = control_point.NominalBeamEnergy if hasattr(control_point, 'NominalBeamEnergy') else "Unknown"
                    layers.append(((np.array(spots_x), np.array(spots_y), np.array(spots_weight)*scale_to_mu), energy, spot_count))
            if layers:
                beams.append(layers)
        return filename, beams
    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        return None

def plot_spot_distribution(layer_data, max_layers, spot_size_multiplier, plot_type, equal_axes, compare_weights, tolerance, decimal_places=2, show_spot_numbers=False, show_spot_weights=True):
    """在一个窗口内并列显示多个 Plan 的分布，滑动条绑定键盘左右键，支持显示 Spot 编号和权重"""
    num_plans = len(layer_data)
    fig, axes = plt.subplots(1, num_plans, figsize=(6 * num_plans, 6), sharey=True)
    plt.subplots_adjust(bottom=0.25)

    if num_plans == 1:
        axes = [axes]

    colors = plt.cm.get_cmap('tab10', num_plans)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']
    single_spot_marker = '*'
    energy_values = [[] for _ in range(num_plans)]

    for idx, (layers, label) in enumerate(layer_data):
        ax = axes[idx]
        if layers:
            (spots_x, spots_y, spots_weight), energy, spot_count = layers[0]
            total_mu = np.sum(spots_weight)
            weight_format = f".{decimal_places}f"
            if plot_type == "scatter":
                if spot_count == 1:
                    ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                               color=colors(idx), marker=single_spot_marker)
                    if show_spot_numbers:
                        ax.text(spots_x[0] + 0.5, spots_y[0] - 0.5, "1", fontsize=8, color='red')
                    if show_spot_weights:
                        ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:{weight_format}}",
                                fontsize=8, color='black')
                else:
                    ax.scatter(spots_x, spots_y, s=np.array(spots_weight) * spot_size_multiplier, alpha=0.8,
                               color=colors(idx), marker=markers[idx % len(markers)])
                    for i, (x, y, w) in enumerate(zip(spots_x, spots_y, spots_weight)):
                        if show_spot_numbers:
                            ax.text(x + 0.5, y - 0.5, f"{i + 1}", fontsize=8, color='red')
                        if show_spot_weights:
                            ax.text(x + 0.5, y + 0.5, f"{w:{weight_format}}", fontsize=8, color='black')
            else:  # line
                if spot_count == 1:
                    ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                               color=colors(idx), marker=single_spot_marker)
                    if show_spot_numbers:
                        ax.text(spots_x[0] + 0.5, spots_y[0] - 0.5, "1", fontsize=8, color='red')
                    if show_spot_weights:
                        ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:{weight_format}}",
                                fontsize=8, color='black')
                else:
                    ax.plot(spots_x, spots_y, color=colors(idx), linestyle='--', linewidth=2,
                            marker=markers[idx % len(markers)])
                    for i, (x, y, w) in enumerate(zip(spots_x, spots_y, spots_weight)):
                        if show_spot_numbers:
                            ax.text(x + 0.5, y - 0.5, f"{i + 1}", fontsize=8, color='red')
                        if show_spot_weights:
                            ax.text(x + 0.5, y + 0.5, f"{w:{weight_format}}", fontsize=8, color='black')
            energy_values[idx].append(energy)
        energy_str = f"{energy_values[idx][0]} MeV" if energy_values[idx][0] != "Unknown" else "Unknown"
        ax.set_title(f"{label}\nLayer 0 (Energy: {energy_str}, Total MU: {total_mu:.2f})", fontsize=10)
        ax.set_xlabel("X Position (mm)")
        if idx == 0:
            ax.set_ylabel("Y Position (mm)")
        ax.grid(True)
        ax.minorticks_on()
        ax.tick_params(which='minor', length=4, color='gray')
        if equal_axes:
            ax.set_aspect('equal')

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Layer', 0, max_layers - 1, valinit=0, valstep=1)

    def update(val):
        layer_idx = int(slider.val)
        for idx, (layers, label) in enumerate(layer_data):
            ax = axes[idx]
            ax.clear()
            energy_values[idx].clear()
            if layer_idx < len(layers):
                (spots_x, spots_y, spots_weight), energy, spot_count = layers[layer_idx]
                total_mu = np.sum(spots_weight)
                weight_format = f".{decimal_places}f"
                if plot_type == "scatter":
                    if spot_count == 1:
                        ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                                   color=colors(idx), marker=single_spot_marker)
                        if show_spot_numbers:
                            ax.text(spots_x[0] + 0.5, spots_y[0] - 0.5, "1", fontsize=8, color='red')
                        if show_spot_weights:
                            ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:{weight_format}}",
                                    fontsize=8, color='black')
                    else:
                        ax.scatter(spots_x, spots_y, s=np.array(spots_weight) * spot_size_multiplier, alpha=0.8,
                                   color=colors(idx), marker=markers[idx % len(markers)])
                        for i, (x, y, w) in enumerate(zip(spots_x, spots_y, spots_weight)):
                            if show_spot_numbers:
                                ax.text(x + 0.5, y - 0.5, f"{i + 1}", fontsize=8, color='red')
                            if show_spot_weights:
                                ax.text(x + 0.5, y + 0.5, f"{w:{weight_format}}", fontsize=8, color='black')
                else:  # line
                    if spot_count == 1:
                        ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                                   color=colors(idx), marker=single_spot_marker)
                        if show_spot_numbers:
                            ax.text(spots_x[0] + 0.5, spots_y[0] - 0.5, "1", fontsize=8, color='red')
                        if show_spot_weights:
                            ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:{weight_format}}",
                                    fontsize=8, color='black')
                    else:
                        ax.plot(spots_x, spots_y, color=colors(idx), linestyle='--', linewidth=2,
                                marker=markers[idx % len(markers)])
                        for i, (x, y, w) in enumerate(zip(spots_x, spots_y, spots_weight)):
                            if show_spot_numbers:
                                ax.text(x + 0.5, y - 0.5, f"{i + 1}", fontsize=8, color='red')
                            if show_spot_weights:
                                ax.text(x + 0.5, y + 0.5, f"{w:{weight_format}}", fontsize=8, color='black')
                energy_values[idx].append(energy)
            energy_str = f"{energy_values[idx][0]} MeV" if energy_values[idx][0] != "Unknown" else "Unknown"
            ax.set_title(f"{label}\nLayer {layer_idx} (Energy: {energy_str}, Total MU: {total_mu:.2f})", fontsize=10)
            ax.set_xlabel("X Position (mm)")
            if idx == 0:
                ax.set_ylabel("Y Position (mm)")
            ax.grid(True)
            ax.minorticks_on()
            ax.tick_params(which='minor', length=4, color='gray')
            if equal_axes:
                ax.set_aspect('equal')

        if compare_weights and len(layer_data) > 1:
            ref_spots_x, ref_spots_y, ref_spots_weight = None, None, None
            ref_label = layer_data[0][1]
            for idx, (layers, label) in enumerate(layer_data):
                if layer_idx < len(layers):
                    (spots_x, spots_y, spots_weight), _, _ = layers[layer_idx]
                    if idx == 0:
                        ref_spots_x, ref_spots_y, ref_spots_weight = spots_x, spots_y, spots_weight
                    else:
                        for i, (x, y, w) in enumerate(zip(spots_x, spots_y, spots_weight)):
                            distances = np.sqrt((ref_spots_x - x)**2 + (ref_spots_y - y)**2)
                            closest_idx = np.argmin(distances)
                            min_distance = distances[closest_idx]
                            if min_distance < 0.1:
                                ref_w = ref_spots_weight[closest_idx]
                                if ref_w != 0:
                                    diff_percent = abs(ref_w - w) / ref_w
                                    if diff_percent > tolerance:
                                        messagebox.showwarning("Weight Difference Alert",
                                                               f"Layer {layer_idx}, Spot at ({x:.2f}, {y:.2f}):\n"
                                                               f"{ref_label}: {ref_w:.2f}\n"
                                                               f"{label}: {w:.2f}\n"
                                                               f"Difference: {diff_percent*100:.2f}% > {tolerance*100}%")
                            else:
                                messagebox.showwarning("Weight Comparison Error",
                                                       f"Layer {layer_idx}: No matching spot found for ({x:.2f}, {y:.2f}) in {label}!")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key_press(event):
        current_val = slider.val
        if event.key == 'left' and current_val > 0:
            slider.set_val(current_val - 1)
        elif event.key == 'right' and current_val < max_layers - 1:
            slider.set_val(current_val + 1)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    def save_plot(event):
        if event.button == 3:
            layer_idx = int(slider.val)
            filename = f"spot_distribution_all_plans_layer_{layer_idx}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot as {filename}")

    fig.canvas.mpl_connect('button_press_event', save_plot)
    plt.show()

def convert_to_csv(files, file_types):
    """将 DICOM 文件转换为 CSV 格式"""
    if not files:
        messagebox.showerror("Error", "Please select DICOM or CSV files first!")
        return

    all_csv = all(file_type == "csv" for file_type in file_types)
    if all_csv:
        messagebox.showinfo("Info", "All imported files are already in CSV format, no conversion needed.")
        return

    for file_path, file_type in zip(files, file_types):
        if file_type == "csv":
            messagebox.showinfo("Info", f"{os.path.basename(file_path)} is already a CSV file, skipping conversion.")
            continue

        beam_info = extract_spot_data(file_path)
        if not beam_info:
            messagebox.showwarning("Warning", f"Failed to process {file_path}.")
            continue

        filename, beams = beam_info
        output_dir = os.path.dirname(file_path)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_spot_data.csv")

        with open(output_file, 'w') as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{filename},{current_time}\n")
            f.write("beam_idx,energy,current_layer,num_of_spots,weight_layer,(x y),weight\n")

            for beam_idx, layers in enumerate(beams):
                total_layers = len(layers)
                for layer_idx, ((spots_x, spots_y, spots_weight), energy, spot_count) in enumerate(layers):
                    total_weight = np.sum(spots_weight)
                    layer_info = f"{beam_idx},{energy},{layer_idx + 1}/{total_layers},{spot_count},{total_weight:.4f}"
                    for x, y, w in zip(spots_x, spots_y, spots_weight):
                        spot_info = f"({x:.2f} {y:.2f}),{w:.6f}"
                        f.write(f"{layer_info},{spot_info}\n")
                    if spot_count == 0:
                        f.write(f"{layer_info},,\n")

        messagebox.showinfo("Success", f"Converted {filename} to {output_file} successfully.")

def parse_csv_data(csv_file):
    """Parse CSV file into a format compatible with extract_spot_data, including Beam identifier"""
    try:
        filename = os.path.basename(csv_file)
        beams = []
        current_beam_idx = -1
        current_layer = None
        spots_x = []
        spots_y = []
        spots_weight = []
        energy = None
        spot_count = 0

        with open(csv_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"Error: {csv_file} has fewer than 2 lines")
                return None

            for i, line in enumerate(lines[2:], start=2):
                line = line.strip()
                if not line:
                    print(f"Warning: Skipping empty line {i} in {csv_file}")
                    continue

                parts = line.split(',')
                if len(parts) != 7:
                    print(f"Error: Line {i} in {csv_file} has {len(parts)} fields (expected 7): {line}")
                    continue

                try:
                    new_beam_idx = int(parts[0])
                    new_energy = float(parts[1])
                    layer_info = parts[2].split('/')
                    if len(layer_info) != 2:
                        print(f"Error: Invalid layer format in line {i} of {csv_file}: {line}")
                        continue
                    layer_idx = int(layer_info[0]) - 1
                    new_spot_count = int(parts[3])

                    if current_beam_idx != -1 and (
                            new_beam_idx != current_beam_idx or layer_idx != current_layer):
                        if len(spots_x) != spot_count:
                            print(
                                f"Warning: Spot count mismatch in Beam {current_beam_idx}, Layer {current_layer + 1}: expected {spot_count}, got {len(spots_x)}")
                        layer_data = (
                            (np.array(spots_x), np.array(spots_y), np.array(spots_weight)), energy, len(spots_x))
                        while current_beam_idx >= len(beams):
                            beams.append([])
                        beams[current_beam_idx].append(layer_data)
                        spots_x, spots_y, spots_weight = [], [], []

                    current_beam_idx = new_beam_idx
                    current_layer = layer_idx
                    energy = new_energy
                    spot_count = new_spot_count

                    xy = parts[5].strip('()').split()
                    if len(xy) != 2:
                        print(f"Error: Invalid (x y) format in line {i} of {csv_file}: {parts[5]}")
                        continue
                    x, y = float(xy[0]), float(xy[1])
                    w = float(parts[6])
                    spots_x.append(x)
                    spots_y.append(y)
                    spots_weight.append(w)

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line {i} in {csv_file}: {line} - {e}")
                    continue

            if spots_x:
                if len(spots_x) != spot_count:
                    print(
                        f"Warning: Spot count mismatch in Beam {current_beam_idx}, Layer {current_layer + 1}: expected {spot_count}, got {len(spots_x)}")
                layer_data = ((np.array(spots_x), np.array(spots_y), np.array(spots_weight)), energy, len(spots_x))
                while current_beam_idx >= len(beams):
                    beams.append([])
                beams[current_beam_idx].append(layer_data)

        if not beams:
            print(f"Error: No valid beams found in {csv_file}")
            return None
        return filename, beams
    except Exception as e:
        print(f"Error parsing {csv_file}: {e}")
        return None

def merge_same_energy_layers_and_sum_weights(dicom_file, output_dir=None):
    """
    从 RT Ion Plan DICOM 文件中合并相同能量层的 IonControlPointSequence，
    相同坐标的权重相加，并保存为新 DICOM 文件。

    参数:
        dicom_file (str): 输入 DICOM 文件路径
        output_dir (str, optional): 输出目录，默认为输入文件所在目录

    返回:
        tuple: (filename, beams, output_file)，合并后的数据和新文件路径
    """
    try:
        # 读取原始 DICOM 文件
        ds = pydicom.dcmread(dicom_file)
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
            return None

        filename = os.path.basename(dicom_file)
        beams = []

        # 处理每个 Beam
        for ii, ion_beam in enumerate(ds.IonBeamSequence):
            if hasattr(ion_beam, 'TreatmentDeliveryType') and ion_beam.TreatmentDeliveryType != 'TREATMENT':
                continue
            scale_to_mu = float(ds.FractionGroupSequence[0].ReferencedBeamSequence[ii].BeamMeterset) / float(
                ion_beam.FinalCumulativeMetersetWeight)

            # 按能量分组控制点
            energy_dict = {}
            for i in range(0, len(ion_beam.IonControlPointSequence), 2):
                control_point = ion_beam.IonControlPointSequence[i]
                if not hasattr(control_point, 'ScanSpotPositionMap'):
                    continue
                energy = control_point.NominalBeamEnergy if hasattr(control_point, 'NominalBeamEnergy') else "Unknown"
                if energy not in energy_dict:
                    energy_dict[energy] = []
                energy_dict[energy].append(control_point)

            # 合并相同能量层
            layers = []
            for energy, control_points in energy_dict.items():
                spot_dict = {}  # key: (x, y), value: weight
                for control_point in control_points:
                    positions = control_point.ScanSpotPositionMap
                    weights = control_point.ScanSpotMetersetWeights
                    spot_count = len(positions) // 2
                    if spot_count == 1 and not isinstance(weights, (list, tuple, np.ndarray)):
                        x, y = positions[0], positions[1]
                        w = weights * scale_to_mu
                        spot_dict[(x, y)] = spot_dict.get((x, y), 0) + w
                    else:
                        for j in range(0, len(positions), 2):
                            x, y = positions[j], positions[j + 1]
                            w = weights[j // 2] * scale_to_mu
                            spot_dict[(x, y)] = spot_dict.get((x, y), 0) + w

                spots_x = np.array([pos[0] for pos in spot_dict.keys()])
                spots_y = np.array([pos[1] for pos in spot_dict.keys()])
                spots_weight = np.array([w for w in spot_dict.values()])
                spot_count = len(spots_x)
                layers.append(((spots_x, spots_y, spots_weight), energy, spot_count))

            if layers:
                beams.append(layers)

        # 创建新的 DICOM 数据集
        new_ds = ds.copy()  # 复制原始数据集
        new_ion_beam_seq = Sequence()

        for ii, ion_beam in enumerate(ds.IonBeamSequence):
            if hasattr(ion_beam, 'TreatmentDeliveryType') and ion_beam.TreatmentDeliveryType != 'TREATMENT':
                new_ion_beam_seq.append(ion_beam)
                continue

            # 创建新的 IonBeam 数据集
            new_ion_beam = ion_beam.copy()
            new_control_point_seq = Sequence()
            layers = beams[len(new_ion_beam_seq)]  # 当前 Beam 的合并层

            for (spots_x, spots_y, spots_weight), energy, spot_count in layers:
                # 创建新的控制点
                control_point = Dataset()
                control_point.NominalBeamEnergy = energy if energy != "Unknown" else None
                control_point.ScanSpotPositionMap = [float(x) for pair in zip(spots_x, spots_y) for x in pair]
                # 权重需要反向缩放回原始单位
                weights = spots_weight / scale_to_mu
                control_point.ScanSpotMetersetWeights = weights.tolist()
                control_point.NumberOfScanSpotPositions = spot_count
                # 复制其他必要的属性（示例）
                if hasattr(ion_beam.IonControlPointSequence[0], 'ScanSpotTuneID'):
                    control_point.ScanSpotTuneID = ion_beam.IonControlPointSequence[0].ScanSpotTuneID
                if hasattr(ion_beam.IonControlPointSequence[0], 'ScanningSpotSize'):
                    control_point.ScanningSpotSize = ion_beam.IonControlPointSequence[0].ScanningSpotSize

                # 添加第二个控制点（如果需要保持格式一致）
                second_control_point = Dataset()
                second_control_point.NominalBeamEnergy = energy if energy != "Unknown" else None
                second_control_point.ScanSpotPositionMap = []
                second_control_point.ScanSpotMetersetWeights = []
                second_control_point.NumberOfScanSpotPositions = 0

                new_control_point_seq.append(control_point)
                new_control_point_seq.append(second_control_point)

            new_ion_beam.IonControlPointSequence = new_control_point_seq
            new_ion_beam.NumberOfControlPoints = len(new_control_point_seq)
            new_ion_beam_seq.append(new_ion_beam)

        new_ds.IonBeamSequence = new_ion_beam_seq

        # 保存新文件
        if output_dir is None:
            output_dir = os.path.dirname(dicom_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_merged_split.dcm")
        new_ds.save_as(output_file)
        print(f"Saved merged DICOM file as {output_file}")

        return filename, beams, output_file

    except Exception as e:
        print(f"Error processing or saving {dicom_file}: {e}")
        return None


def merge_same_energy_layers_concat_spots(dicom_file, output_dir=None):
    """
    从 RT Ion Plan DICOM 文件中合并相同能量层的 IonControlPointSequence，
    坐标和权重按顺序连接，不累加权重，并保存为新 DICOM 文件。

    参数:
        dicom_file (str): 输入 DICOM 文件路径
        output_dir (str, optional): 输出目录，默认为输入文件所在目录

    返回:
        tuple: (filename, beams, output_file)，合并后的数据和新文件路径
    """
    try:
        # 读取原始 DICOM 文件
        ds = pydicom.dcmread(dicom_file)
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
            return None

        filename = os.path.basename(dicom_file)
        beams = []

        # 处理每个 Beam
        for ii, ion_beam in enumerate(ds.IonBeamSequence):
            if hasattr(ion_beam, 'TreatmentDeliveryType') and ion_beam.TreatmentDeliveryType != 'TREATMENT':
                continue
            scale_to_mu = float(ds.FractionGroupSequence[0].ReferencedBeamSequence[ii].BeamMeterset) / float(
                ion_beam.FinalCumulativeMetersetWeight)

            # 按能量分组控制点
            energy_dict = {}
            for i in range(0, len(ion_beam.IonControlPointSequence), 2):
                control_point = ion_beam.IonControlPointSequence[i]
                if not hasattr(control_point, 'ScanSpotPositionMap'):
                    continue
                energy = control_point.NominalBeamEnergy if hasattr(control_point, 'NominalBeamEnergy') else "Unknown"
                if energy not in energy_dict:
                    energy_dict[energy] = []
                energy_dict[energy].append(control_point)

            # 合并相同能量层，连接 Spots
            layers = []
            for energy, control_points in energy_dict.items():
                all_spots_x = []
                all_spots_y = []
                all_spots_weight = []

                for control_point in control_points:
                    positions = control_point.ScanSpotPositionMap
                    weights = control_point.ScanSpotMetersetWeights
                    spot_count = len(positions) // 2
                    if spot_count == 1 and not isinstance(weights, (list, tuple, np.ndarray)):
                        all_spots_x.append(positions[0])
                        all_spots_y.append(positions[1])
                        all_spots_weight.append(weights * scale_to_mu)
                    else:
                        for j in range(0, len(positions), 2):
                            all_spots_x.append(positions[j])
                            all_spots_y.append(positions[j + 1])
                            all_spots_weight.append(weights[j // 2] * scale_to_mu)

                # 转换为 numpy arrays
                spots_x = np.array(all_spots_x)
                spots_y = np.array(all_spots_y)
                spots_weight = np.array(all_spots_weight)
                spot_count = len(spots_x)
                layers.append(((spots_x, spots_y, spots_weight), energy, spot_count))

            if layers:
                beams.append(layers)

        # 创建新的 DICOM 数据集
        new_ds = ds.copy()
        new_ion_beam_seq = Sequence()

        for ii, ion_beam in enumerate(ds.IonBeamSequence):
            if hasattr(ion_beam, 'TreatmentDeliveryType') and ion_beam.TreatmentDeliveryType != 'TREATMENT':
                new_ion_beam_seq.append(ion_beam)
                continue

            # 创建新的 IonBeam 数据集
            new_ion_beam = ion_beam.copy()
            new_control_point_seq = Sequence()
            layers = beams[len(new_ion_beam_seq)]  # 当前 Beam 的合并层

            for (spots_x, spots_y, spots_weight), energy, spot_count in layers:
                # 创建新的控制点
                control_point = Dataset()
                control_point.NominalBeamEnergy = energy if energy != "Unknown" else None
                control_point.ScanSpotPositionMap = [float(x) for pair in zip(spots_x, spots_y) for x in pair]
                # 权重反向缩放回原始单位
                weights = spots_weight / scale_to_mu
                control_point.ScanSpotMetersetWeights = weights.tolist()
                control_point.NumberOfScanSpotPositions = spot_count
                # 复制其他必要的属性
                if hasattr(ion_beam.IonControlPointSequence[0], 'ScanSpotTuneID'):
                    control_point.ScanSpotTuneID = ion_beam.IonControlPointSequence[0].ScanSpotTuneID
                if hasattr(ion_beam.IonControlPointSequence[0], 'ScanningSpotSize'):
                    control_point.ScanningSpotSize = ion_beam.IonControlPointSequence[0].ScanningSpotSize

                # 添加第二个控制点（保持格式一致）
                second_control_point = Dataset()
                second_control_point.NominalBeamEnergy = energy if energy != "Unknown" else None
                second_control_point.ScanSpotPositionMap = []
                second_control_point.ScanSpotMetersetWeights = []
                second_control_point.NumberOfScanSpotPositions = 0

                new_control_point_seq.append(control_point)
                new_control_point_seq.append(second_control_point)

            new_ion_beam.IonControlPointSequence = new_control_point_seq
            new_ion_beam.NumberOfControlPoints = len(new_control_point_seq)
            new_ion_beam_seq.append(new_ion_beam)

        new_ds.IonBeamSequence = new_ion_beam_seq

        # 保存新文件
        if output_dir is None:
            output_dir = os.path.dirname(dicom_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_concat.dcm")
        new_ds.save_as(output_file)
        print(f"Saved concatenated DICOM file as {output_file}")

        return filename, beams, output_file

    except Exception as e:
        print(f"Error processing or saving {dicom_file}: {e}")
        return None


def find_repeat_items(a, quick=False):
    """找到列表中的重复项，返回字典，键为项，值为索引列表"""
    index_all = {}
    for i in range(len(a)):
        target = a[i]
        index_ = []
        for index, value in enumerate(a):
            if value == target:
                index_.append(index)
            elif not quick:
                if sum(abs(np.array(value) - np.array(target))) < 0.1:
                    index_.append(index)
        if isinstance(target, list):
            index_all['_'.join([str(x) for x in target])] = index_
        else:
            index_all[target] = index_
    return index_all


def merge_repaint_spots(dicom_file, output_dir=None, show=False, quick=True):
    """
    合并 RT Ion Plan DICOM 文件中重复的 Spot，权重相加，生成新的 DICOM 文件。

    参数:
        dicom_file (str): 输入 DICOM 文件路径
        output_dir (str, optional): 输出目录，默认为输入文件所在目录
        show (bool): 是否显示原始和合并后的 Spot 分布图
        quick (bool): 是否快速合并（跳过容差检查）

    返回:
        tuple: (filename, output_file, stats)，文件名、新文件路径和统计信息
    """
    try:
        # 读取 DICOM 文件
        ds = pydicom.dcmread(dicom_file)
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
            return None

        filename = os.path.basename(dicom_file)
        numofspots_before = 0
        numofspots_after = 0

        # 处理每个 Beam
        for i in range(len(ds.IonBeamSequence)):
            if ds.IonBeamSequence[i].TreatmentDeliveryType == 'TREATMENT':
                for j in range(len(ds.IonBeamSequence[i].IonControlPointSequence)):
                    if j % 2 == 0:
                        scanmap1 = ds.IonBeamSequence[i].IonControlPointSequence[j].ScanSpotPositionMap
                        scanweight1 = ds.IonBeamSequence[i].IonControlPointSequence[j].ScanSpotMetersetWeights
                        if isinstance(scanweight1, list):
                            numofspots_before += len(scanweight1)
                            scanmap1_list = [[scanmap1[ii * 2], scanmap1[ii * 2 + 1]] for ii in range(len(scanweight1))]
                            index_all = find_repeat_items(scanmap1_list, quick)
                            scanweight2 = copy.deepcopy(scanweight1)
                            for key, value in index_all.items():
                                if len(value) > 1:
                                    scanweight2[value[0]] = sum([scanweight2[ii] for ii in value])
                                    for iii in value[1:]:
                                        scanweight2[iii] = 0
                            scanmap1_list = [scanmap1_list[ii] for ii in range(len(scanweight1)) if scanweight2[ii] > 0]
                            scanweight2 = [xx for xx in scanweight2 if xx > 0]
                            scanmap2 = np.array(scanmap1_list).flatten()
                            numofspots_after += len(scanweight2)

                            # 更新控制点数据
                            ds.IonBeamSequence[i].IonControlPointSequence[j].ScanSpotPositionMap = list(scanmap2)
                            ds.IonBeamSequence[i].IonControlPointSequence[j].ScanSpotMetersetWeights = scanweight2
                            ds.IonBeamSequence[i].IonControlPointSequence[j].NumberOfScanSpotPositions = len(
                                scanweight2)
                            ds.IonBeamSequence[i].IonControlPointSequence[j].NumberOfPaintings = 1

                            ds.IonBeamSequence[i].IonControlPointSequence[j + 1].ScanSpotPositionMap = list(scanmap2)
                            ds.IonBeamSequence[i].IonControlPointSequence[j + 1].ScanSpotMetersetWeights = len(
                                scanweight2) * [0]
                            ds.IonBeamSequence[i].IonControlPointSequence[j + 1].NumberOfScanSpotPositions = len(
                                scanweight2)
                            ds.IonBeamSequence[i].IonControlPointSequence[j + 1].NumberOfPaintings = 1

                            if show:
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                                ax1.plot(scanmap1[::2], scanmap1[1::2], '--b')
                                ax1.scatter(scanmap1[::2], scanmap1[1::2], s=[x * 1e4 for x in scanweight1])
                                ax1.set_title('Original plan')

                                ax2.scatter(scanmap2[::2], scanmap2[1::2], c='b', s=[x * 1e4 for x in scanweight2])
                                ax2.plot(scanmap2[::2], scanmap2[1::2], '--r')
                                ax2.set_title('MergeRepaint plan')
                                ax2.set_xlim(ax1.get_xlim())
                                plt.show()
                        else:
                            numofspots_before += 1
                            numofspots_after += 1

        # 保存新文件
        if output_dir is None:
            output_dir = os.path.dirname(dicom_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_MergeRepaint.dcm")
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.save_as(output_file)

        # 统计信息
        numofbeams = len([x for x in ds.IonBeamSequence if x.TreatmentDeliveryType == "TREATMENT"])
        mus = sum(
            [x.BeamMeterset for x in ds.FractionGroupSequence[0].ReferencedBeamSequence if hasattr(x, "BeamMeterset")])
        numoflayers = sum(
            [len(x.IonControlPointSequence) / 2 for x in ds.IonBeamSequence if x.TreatmentDeliveryType == "TREATMENT"])
        stats = {
            'MU': mus,
            'beams': numofbeams,
            'layers': numoflayers,
            'spots_before': numofspots_before,
            'spots_after': numofspots_after
        }

        print(
            f"Processed {filename}: MU={mus}, Beams={numofbeams}, Layers={numoflayers}, Spots Before={numofspots_before}, Spots After={numofspots_after}")
        return filename, output_file, stats

    except Exception as e:
        print(f"Error processing {dicom_file}: {e}")
        return None


def find_repeat_items_split(a, quick=False):
    """找到列表中的重复项，返回索引矩阵，列数为最大重复次数"""
    index_all = {}
    max_len = 0
    for i in range(len(a)):
        target = a[i]
        index_ = []
        for index, value in enumerate(a):
            if value == target:
                index_.append(index)
            elif not quick:
                if sum(abs(np.array(value) - np.array(target))) < 0.1:
                    index_.append(index)
        if isinstance(target, list):
            index_all['_'.join([str(x) for x in target])] = index_
        else:
            index_all[target] = index_
        max_len = max(max_len, len(index_))
    matrix_index = []
    for key, value in index_all.items():
        matrix_index.append(value + [-1] * (max_len - len(value)))
    return np.array(matrix_index)


def split_repaint_spots(dicom_file, output_dir=None, show=False, quick=True):
    """
    将 RT Ion Plan DICOM 文件中的重复 Spot 分割为多个控制点，生成新的 DICOM 文件。

    参数:
        dicom_file (str): 输入 DICOM 文件路径
        output_dir (str, optional): 输出目录，默认为输入文件所在目录
        show (bool): 是否显示原始和分割后的 Spot 分布图
        quick (bool): 是否快速分割（跳过容差检查）

    返回:
        tuple: (filename, output_file)，文件名和新文件路径
    """
    try:
        # 读取 DICOM 文件
        ds = pydicom.dcmread(dicom_file)
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
            return None

        filename = os.path.basename(dicom_file)
        ds_new = copy.deepcopy(ds)

        # 处理每个 Beam
        for i in range(len(ds.IonBeamSequence)):
            NumberOfControlPoints = 0
            if ds.IonBeamSequence[i].TreatmentDeliveryType == 'TREATMENT':
                CumulativeMetersetWeight = 0
                IonCPSeq1 = ds.IonBeamSequence[i].IonControlPointSequence
                IonCPSeq2 = []
                for j in range(len(IonCPSeq1)):
                    if j % 2 == 0:
                        scanmap1 = IonCPSeq1[j].ScanSpotPositionMap
                        scanweight1 = IonCPSeq1[j].ScanSpotMetersetWeights
                        ICP_j = IonCPSeq1[j]
                        ICP_j1 = IonCPSeq1[j + 1]
                        if isinstance(scanweight1, list):
                            scanmap1_2d = [[scanmap1[ii * 2], scanmap1[ii * 2 + 1]] for ii in range(len(scanweight1))]
                            matrix_index = find_repeat_items_split(scanmap1_2d, quick)
                            print(
                                f"Beam {i}, Layer {int(j / 2)}, Number of Paintings: {IonCPSeq1[j].NumberOfPaintings} =? {matrix_index.shape[1]}")
                            for k in range(matrix_index.shape[1]):
                                new_index = [ii for ii in matrix_index[:, k] if ii != -1]
                                new_map = list(np.array([scanmap1_2d[ii] for ii in new_index]).flatten())
                                new_wts = [scanweight1[ii] for ii in new_index]
                                if j == 0 and k != 0:
                                    ICP_j_new = copy.deepcopy(ICP_j1)
                                    ICP_j_new.ScanSpotPositionMap = new_map
                                    ICP_j_new.ScanSpotMetersetWeights = new_wts
                                    ICP_j_new.NumberOfScanSpotPositions = len(new_wts)
                                    ICP_j_new.NumberOfPaintings = 1
                                    ICP_j_new.CumulativeMetersetWeight = CumulativeMetersetWeight
                                    NumberOfControlPoints += 1
                                    ICP_j_new.ControlPointIndex = NumberOfControlPoints - 1
                                    IonCPSeq2.append(copy.deepcopy(ICP_j_new))
                                else:
                                    ICP_j.ScanSpotPositionMap = new_map
                                    ICP_j.ScanSpotMetersetWeights = new_wts
                                    ICP_j.NumberOfScanSpotPositions = len(new_wts)
                                    ICP_j.NumberOfPaintings = 1
                                    ICP_j.CumulativeMetersetWeight = CumulativeMetersetWeight
                                    NumberOfControlPoints += 1
                                    ICP_j.ControlPointIndex = NumberOfControlPoints - 1
                                    IonCPSeq2.append(copy.deepcopy(ICP_j))

                                ICP_j1.ScanSpotPositionMap = new_map
                                ICP_j1.ScanSpotMetersetWeights = len(new_wts) * [0]
                                ICP_j1.NumberOfScanSpotPositions = len(new_wts)
                                ICP_j1.NumberOfPaintings = 1
                                CumulativeMetersetWeight += sum(new_wts)
                                ICP_j1.CumulativeMetersetWeight = CumulativeMetersetWeight
                                NumberOfControlPoints += 1
                                ICP_j1.ControlPointIndex = NumberOfControlPoints - 1
                                IonCPSeq2.append(copy.deepcopy(ICP_j1))

                                if show:
                                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                                    ax[0].plot(scanmap1[::2], scanmap1[1::2], '--b')
                                    ax[0].scatter(scanmap1[::2], scanmap1[1::2], s=[x * 1e4 for x in scanweight1])
                                    ax[0].set_title('Original plan')
                                    maps_2d = np.array(scanmap1).reshape((int(len(scanmap1) / 2), 2))
                                    for ii in range(maps_2d.shape[0]):
                                        ax[0].annotate(ii, (maps_2d[ii][0] + ii * 0.5, maps_2d[ii][1]))

                                    ax[1].scatter(new_map[::2], new_map[1::2], c='b', s=[x * 1e4 for x in new_wts])
                                    ax[1].plot(new_map[::2], new_map[1::2], '--r')
                                    ax[1].set_title('SplitRepaint plan')
                                    ax[1].set_xlim(ax[0].get_xlim())
                                    maps_2d = np.array(new_map).reshape((int(len(new_map) / 2), 2))
                                    for ii in range(maps_2d.shape[0]):
                                        ax[1].annotate(new_index[ii],
                                                       (maps_2d[ii][0] + ii * 0.1, maps_2d[ii][1] + ii * 0.1))
                                    plt.show()
                        else:
                            NumberOfControlPoints += 1
                            ICP_j.ControlPointIndex = NumberOfControlPoints - 1
                            IonCPSeq2.append(copy.deepcopy(ICP_j))
                            NumberOfControlPoints += 1
                            ICP_j1.ControlPointIndex = NumberOfControlPoints - 1
                            IonCPSeq2.append(copy.deepcopy(ICP_j1))

                ds_new.IonBeamSequence[i].NumberOfControlPoints = NumberOfControlPoints
                ds_new.IonBeamSequence[i].IonControlPointSequence = IonCPSeq2
                print(
                    f"CumulativeMetersetWeight {ds.IonBeamSequence[i].FinalCumulativeMetersetWeight} =? {CumulativeMetersetWeight}")

        # 保存新文件
        if output_dir is None:
            output_dir = os.path.dirname(dicom_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_SplitRepaint.dcm")
        ds_new.SOPInstanceUID = pydicom.uid.generate_uid()
        ds_new.save_as(output_file)
        print(f"Saved split repaint DICOM file as {output_file}")

        return filename, output_file

    except Exception as e:
        print(f"Error processing {dicom_file}: {e}")
        return None


# 示例用法
if __name__ == "__main__":
    dicom_file = r"*.dcm"  # 替换为实际文件路径
    result = merge_same_energy_layers_and_sum_weights(dicom_file)
    if result:
        filename, beams, output_file = result
        print(f"Processed file: {filename}")
        print(f"Saved as: {output_file}")
        for beam_idx, layers in enumerate(beams):
            print(f"Beam {beam_idx}:")
            for layer_idx, ((spots_x, spots_y, spots_weight), energy, spot_count) in enumerate(layers):
                print(f"  Layer {layer_idx} (Energy: {energy}, Spots: {spot_count}):")
                for x, y, w in zip(spots_x, spots_y, spots_weight):
                    print(f"    Spot at ({x:.2f}, {y:.2f}): Weight = {w:.3f}")