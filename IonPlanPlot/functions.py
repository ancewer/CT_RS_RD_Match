import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import tkinter as tk
from tkinter import messagebox

def extract_spot_data(dicom_file):
    """从 RT Ion Plan DICOM 文件中按 Beam 提取 spot 数据，处理单个 spot 的 weights"""
    try:
        ds = pydicom.dcmread(dicom_file)
        if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.481.8':
            return None

        filename = os.path.basename(dicom_file)
        beams = []
        for ion_beam in ds.IonBeamSequence:
            if hasattr(ion_beam, 'TreatmentDeliveryType') and ion_beam.TreatmentDeliveryType != 'TREATMENT':
                continue
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
                    layers.append(((np.array(spots_x), np.array(spots_y), np.array(spots_weight)), energy, spot_count))
            if layers:
                beams.append(layers)
        return filename, beams
    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        return None

def plot_spot_distribution(layer_data, max_layers, spot_size_multiplier, plot_type, equal_axes, compare_weights, tolerance):
    """在一个窗口内并列显示多个 Plan 的分布，滑动条绑定键盘左右键"""
    num_plans = len(layer_data)
    fig, axes = plt.subplots(1, num_plans, figsize=(6 * num_plans, 6), sharey=True)
    plt.subplots_adjust(bottom=0.25)

    if num_plans == 1:
        axes = [axes]

    # 初始绘制第 0 层
    colors = plt.cm.get_cmap('tab10', num_plans)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']
    single_spot_marker = '*'
    energy_values = [[] for _ in range(num_plans)]

    for idx, (layers, label) in enumerate(layer_data):
        ax = axes[idx]
        if layers:
            (spots_x, spots_y, spots_weight), energy, spot_count = layers[0]
            total_mu = np.sum(spots_weight)
            if plot_type == "scatter":
                if spot_count == 1:
                    ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                               color=colors(idx), marker=single_spot_marker)
                    ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:.2f}",
                            fontsize=8, color='black')
                else:
                    ax.scatter(spots_x, spots_y, s=np.array(spots_weight) * spot_size_multiplier, alpha=0.8,
                               color=colors(idx), marker=markers[idx % len(markers)])
                    for x, y, w in zip(spots_x, spots_y, spots_weight):
                        ax.text(x + 0.5, y + 0.5, f"{w:.2f}", fontsize=8, color='black')
            else:  # line
                if spot_count == 1:
                    ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                               color=colors(idx), marker=single_spot_marker)
                    ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:.2f}",
                            fontsize=8, color='black')
                else:
                    ax.plot(spots_x, spots_y, color=colors(idx), linewidth=2,
                            marker=markers[idx % len(markers)])
                    for x, y, w in zip(spots_x, spots_y, spots_weight):
                        ax.text(x + 0.5, y + 0.5, f"{w:.2f}", fontsize=8, color='black')
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

    # 添加共同滑动条
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
                if plot_type == "scatter":
                    if spot_count == 1:
                        ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                                   color=colors(idx), marker=single_spot_marker)
                        ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:.2f}",
                                fontsize=8, color='black')
                    else:
                        ax.scatter(spots_x, spots_y, s=np.array(spots_weight) * spot_size_multiplier, alpha=0.8,
                                   color=colors(idx), marker=markers[idx % len(markers)])
                        for x, y, w in zip(spots_x, spots_y, spots_weight):
                            ax.text(x + 0.5, y + 0.5, f"{w:.2f}", fontsize=8, color='black')
                else:  # line
                    if spot_count == 1:
                        ax.scatter(spots_x, spots_y, s=spot_size_multiplier * 100, alpha=0.8,
                                   color=colors(idx), marker=single_spot_marker)
                        ax.text(spots_x[0] + 0.5, spots_y[0] + 0.5, f"{spots_weight[0]:.2f}",
                                fontsize=8, color='black')
                    else:
                        ax.plot(spots_x, spots_y, color=colors(idx), linewidth=2,
                                marker=markers[idx % len(markers)])
                        for x, y, w in zip(spots_x, spots_y, spots_weight):
                            ax.text(x + 0.5, y + 0.5, f"{w:.2f}", fontsize=8, color='black')
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

        # Spot Weight 比较逻辑（按坐标匹配）
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

    # 绑定键盘左右键
    def on_key_press(event):
        current_val = slider.val
        if event.key == 'left' and current_val > 0:
            slider.set_val(current_val - 1)
        elif event.key == 'right' and current_val < max_layers - 1:
            slider.set_val(current_val + 1)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # 保存功能
    def save_plot(event):
        if event.button == 3:
            layer_idx = int(slider.val)
            filename = f"spot_distribution_all_plans_layer_{layer_idx}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot as {filename}")

    fig.canvas.mpl_connect('button_press_event', save_plot)
    # plt.tight_layout()
    plt.show()
