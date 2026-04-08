import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import plasma, viridis
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, FormatStrFormatter
from matplotlib.colors import to_rgb, LogNorm
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np
import os
import sys
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
import json5
import ProcessingLibrary
import copy

script_dir = os.path.dirname(os.path.abspath(__file__))

def set_matplotlibstyle_Theresia(textSize, textSizeLegend):
    plt.rcParams.update(
        {
            "xtick.major.size": 5,
            "xtick.major.width": 1,
            "xtick.major.pad": 7,
            "ytick.major.size": 5,
            "ytick.major.width": 1,
            "ytick.major.pad": 7,
            "xtick.minor.size": 2,
            "xtick.minor.width": 0.5,
            "xtick.minor.pad": 5,
            "ytick.minor.size": 2,
            "ytick.minor.width": 0.5,
            "ytick.minor.pad": 5,
            "font.size": 28,
            "legend.fontsize": 24,
            "legend.framealpha": None,
        }
    )

    plt.rcParams["text.usetex"] = False

def set_matplotlibstyle_Fernando(textSize, textSizeLegend):
    plt.rcParams.update({
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "xtick.major.pad": 3,
        "ytick.major.size": 3,
        "ytick.major.width": 0.8,
        "ytick.major.pad": 3,

        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.6,
        "xtick.minor.pad": 2,
        "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.6,
        "ytick.minor.pad": 2,

        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,

        "legend.fontsize": 7,
        "legend.framealpha": 0,
        "axes.linewidth": 1,

        "lines.linewidth": 1,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.8,
        "lines.markeredgecolor": "#13073A",
    })

    plt.rcParams["text.usetex"] = False

def log_fmt(val, pos):
    if val <= 0:
        return ""
    
    exp = np.log10(val)
    
    if not np.isclose(exp, np.round(exp)):
        return ""
    
    exp = int(np.round(exp))
    
    if (exp + 12) % 2 == 0:
        return rf"$10^{{{exp}}}$"
    return ""

def make_log_formatter(exponents_to_show):
    def log_fmt(val, pos):
        if val <= 0:
            return ""
        
        exp = np.log10(val)
        
        if not np.isclose(exp, np.round(exp)):
            return ""
        
        exp = int(np.round(exp))
        
        if exp in exponents_to_show:
            return rf"$10^{{{exp}}}$"
        return ""
    
    return log_fmt

def custom_log_formatter(exponents_to_show, number):
    def log_fmt(val, pos):
        if val <= 0:
            return ""
        
        exp = np.log10(val)
        
        if not np.isclose(exp, np.round(exp)):
            return ""
        
        exp = int(np.round(exp))
        
        if exp in exponents_to_show:
            return rf"${number}0^{{{exp}}}$"
        
        return ""
    
    return log_fmt

def opaque_equivalent(color, alpha, bg=(1,1,1)):
    c = np.array(to_rgb(color))
    bg = np.array(bg)
    return alpha * c + (1 - alpha) * bg

def Vd_color(Vd, min_Vd=0.1,max_Vd=1.5):
    normalized_vd = (Vd - min_Vd) / (max_Vd - min_Vd)
    return plasma(normalized_vd)

def WL_color(WL, min_WL=0.5, max_WL=15):
    normalized_wl = (WL - min_WL) / (max_WL - min_WL)
    return viridis(normalized_wl)

def Area_color(Area, min_Area=0.5, max_Area=150):
    normalized_area = (Area - min_Area) / (max_Area - min_Area)
    return viridis(normalized_area)

def truncate_colormap(cmap, minval=0.3, maxval=1.0, n=100):
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name})',
        cmap(np.linspace(minval, maxval, n))
    )

def safe_json_load(x):
    if isinstance(x, str):
        try:
            x = x.replace('nan', 'null')  # JSON-safe
            out = json5.loads(x)
            return [np.nan if v is None else v for v in out]
        except Exception:
            return x   # or return None
    return x

if __name__ == "__main__":
    # Plot settings
    flength = 6.0
    textSize = 28
    textSizeLegend = 24
    set_matplotlibstyle_Fernando(textSize, textSizeLegend)
    inputdir = os.getcwd()
    
    #data_folder = os.path.join('/','home','delossa','Nextcloud','DataProcessingLab','data')
    data_folder = Path(__file__).resolve().parent.parent / "DataProcessingLab" / "data"

    color_up = "#AA3939"
    color_down = "#887CAF"

    color_stress = "#8C2D04"   # dark burnt orange
    color_relax  = "#3F007D"   # deep purple
    color_precondition = "#006400"   # dark green

    # Background (lighter + transparent)
    cmap_stress = truncate_colormap(plt.cm.Oranges, 0.2, 1)
    cmap_relax  = truncate_colormap(plt.cm.Purples, 0.1, 1)
    cmap_precondition = truncate_colormap(plt.cm.Greens, 0.4, 1)
    norm = LogNorm(vmin=1, vmax=1e5)

    # Figures

    ########## Performance and variability ##########

    if 0: # IdVg curves for encapsulated devices

        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated',"IdVg_TUWien_planar_hbn-encapsulated.csv"))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1)]
        for c in ['Id','Vg','Ig']:
            df[c] = df[c].map(json5.loads)
        
        # metrics
        SS = df[(df['Vd'] == 1.5)]['SS'].iloc[0]
        Vzero_current = df[(df['Vd'] == 1.5)]['Vzero_current'].iloc[0]
        IonIoff = df[(df['Vd'] == 1.5)]['Ion/Ioff'].iloc[0]
        Imax = df[(df['Vd'] == 1.5)]['Imax'].iloc[0]
        Igate_A = df[(df['Vd'] == 1.5)]['gate_leakage'].iloc[0]
        width = df[(df['Vd'] == 1.5)]['width'].iloc[0]

        fig, ax = plt.subplots(figsize=(2.7, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.6
        right_in  = 0.1
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height
        )
        
        # Plot IdVg curves for different Vds values
        vds_values = df['Vd'].unique()
        #colors = viridis(np.linspace(0, 1, len(vds_values)))
        
        for idx, vds in enumerate(sorted(vds_values)):
            df_vds = df[df['Vd'] == vds]
            vg = np.array(df_vds['Vg'].values[0])
            id_vals = np.array(df_vds['Id'].values[0])
            ig_vals = np.array(df_vds['Ig'].values[0])
            ax.plot(vg, id_vals/width*1e6, '-', 
                   markersize=6, label=rf'$V_{{D}}$ = {vds} V', color=Vd_color(vds))
            ax.plot(vg, ig_vals/width*1e6, '--', 
                   markersize=6, label=rf'$V_{{D}}$ = {vds} V', color=Vd_color(vds))
            
        ax.set_yscale('log')

        ax.relim()
        ax.autoscale()

        ymin, ymax = ax.get_ylim()
        ymin = 10**np.floor(np.log10(ymin))
        ymax = 10**np.ceil(np.log10(ymax))

        ax.set_ylim(ymin, ymax)

        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax.yaxis.set_major_formatter(FuncFormatter(make_log_formatter([-12, -10, -8, -6, -4, -2, 0])))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))

        ax.tick_params(axis='y', which='minor')

        # Add legend with smaller font size
        # ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-2]], 
        #      [rf'$V_{{D}}$ = {sorted(vds_values)[0]} V', rf'$V_{{D}}$ = {sorted(vds_values)[-1]} V'],
        #      fontsize=22, loc='lower right', framealpha=0.0)

        # Add an arrow pointing to higher drain voltage curves
        # Add vertical arrow pointing to higher drain voltage curves
        arrow_x = 4
        arrow_y_min = 30e-5
        arrow_y_max = 30e-2
        ax.annotate('', xy=(arrow_x, arrow_y_min), xytext=(arrow_x, arrow_y_max), arrowprops=dict(arrowstyle='<|-', color='black'))
        ax.text(arrow_x, arrow_y_min - 10e-5, 
               rf'$V_\mathsf{{D}}$ = {np.min(vds_values):.1f} V', fontsize=6,
               verticalalignment='top', ha='center')
        ax.text(arrow_x, arrow_y_max-20e-2, 
               rf'$V_\mathsf{{D}}$ = {np.max(vds_values):.1f} V',
               fontsize=6,
               verticalalignment='bottom', ha='center')
        
        ax.text(4,1e-8,r'$I_\mathsf{G}$')
        # Set axis labels with units
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]')
        ax.set_ylabel(r'$I_\mathsf{D}/W$ [$\mu$A/$\mu$m]')
        
        # Add device info text
        device_text = rf'$W/L = \frac{{{df["width"].iloc[0]:.0f}\,\mu\mathrm{{m}}}}{{{df["length"].iloc[0]:.0f}\,\mu\mathrm{{m}}}}$'
        ax.text(0.05, 0.95, device_text, transform=ax.transAxes, verticalalignment='top')
               #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Id_SS = 10**(1/SS*(vg - Vzero_current))
        # ax.plot(vg,Id_SS/width*1e6,'--', color='k',alpha=0.5)

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        
        # Create text with the metrics (Ion/Ioff,SS,Imax)
        metrics_text = rf'$I_\mathsf{{on}}/I_\mathsf{{off}}$ $>$ $10^{int(np.floor(np.log10(IonIoff)))}$; ' + rf'$SS$ = {SS:.0f} mV/dec' + '\n' + rf'$I_{{max}}$ = {Imax/width*1e6:.2f} $\mu$A/$\mu$m; ' + rf'$I_\mathsf{{G}}/A$ $<$ $10^{{{int(np.ceil(np.log10(Igate_A)))}}}$ A/cm$^2$'
        
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=5.5,
               bbox=dict(boxstyle='round', facecolor='white', alpha=1))
        
        # ax.text(0.77, 0.07, rf'@ V$_\mathsf{{D}}$ = {vds} V', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontsize=6)

        plt.savefig(os.path.join(inputdir,'figures','IdVg_encapsulated_1.pdf'), bbox_inches=None, transparent=True)
        plt.close()

    if 0: # IdVg curves for different devices with W/L normalization

        df = pd.read_csv("data/IdVg_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['Vd'] == 1.0) & (df['sample']==1)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get unique devices and assign symbols
        devices = df['dut'].unique()
        arrays = df['array'].unique()
        symbols = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x']
        colors = viridis(np.linspace(0, 1, len(devices)))
        device_colors = {device: colors[idx % len(colors)] for idx, device in enumerate(devices)}

        for idx, device in enumerate(sorted(devices)):
            df_device = df[df['dut'] == device]
            Ith = df_device['Ith'].iloc[0]
            width = df_device['width'].iloc[0]
            area = df_device['area'].iloc[0]
            length = df_device['length'].iloc[0]
            array = df_device['array'].iloc[0]
            vg = np.array(df_device['Vg'].values[0])
            id_vals = np.array(df_device['Id'].values[0])
            ax.plot(vg, id_vals/width*1e6*length, '-', linewidth=2.5, 
                   markersize=6, label=rf'$W/L$ = {width}/{length}', color=device_colors[device])
        ax.axhline(Ith*1e6,linestyle = '--',color='k')
        ax.set_yscale("log")

        # Set axis labels
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D} \cdot L/W$ [$\mu$A]', fontsize=textSize)
        
        # Add more ticks
        ax.tick_params(axis='both', which='major', labelsize=textSizeLegend)
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add legend
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.0)
        
        plt.savefig(script_dir+"/figures/IdVg_duts_WL.pdf", bbox_inches="tight", transparent=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))

        for idx, device in enumerate(sorted(devices)):
            df_device = df[df['dut'] == device]
            width = df_device['width'].values[0]
            area = df_device['area'].values[0]
            length = df_device['length'].values[0]
            array = df_device['array'].iloc[0]
            vg = np.array(df_device['Vg'].values[0])
            id_vals = np.array(df_device['Id'].values[0])
            ax.plot(vg, id_vals*1e6, '-', linewidth=2.5, 
                   markersize=6, label=rf'$W/L$ = {width}/{length}', color=device_colors[device])
            ax.axhline(Ith*width/length*1e6, linestyle='--',color=device_colors[device])
        ax.set_yscale("log")

        # Set axis labels
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D}$ [$\mu$A]', fontsize=textSize)
        
        # Add device info
        device_text = r'$T$ = 300 K' +'\n' + rf'$V_\mathsf{{D}}$ = {df["Vd"].iloc[0]} V'
        ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        # Add more ticks
        ax.tick_params(axis='both', which='major', labelsize=textSizeLegend)
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.0)
        
        plt.savefig(script_dir+"/figures/IdVg_duts.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Plot inverter VTC and schematic
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','inv-transfer_TUWien_planar_hbn-encapsulated.csv'))
        df = df[(df['temp'] == '300K') & (df['sample'] == 4) & (df['Vdd'] == 3.0)]
        for c in ['Voutput','Vinput','Voutput_fit','Vinput_fit','dVoutdVin','dVoutdVin_fit']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(1,1,figsize=(2.1, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.6
        right_in  = 0.1
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            wspace = 0.0/fig_width,
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height,
        )

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        removed_keys = [
            ('TUWien_planar_hbn-encapsulated', 'sinINV3A', 4),
            ('TUWien_planar_hbn-encapsulated', 'INV2A9t1', 4),
        ]
        selected_keys = [
            ('TUWien_planar_hbn-encapsulated', 'INV4A1t1', 4),
        ]
        #all_keys = [k for k in df_groups.keys() if k not in removed_keys]
        all_keys = [k for k in df_groups.keys() if k in selected_keys]
        n_keys = len(all_keys)
        df_filtered = df[
            ~df.set_index(['batch','dut','sample']).index.isin(removed_keys)
            #df.set_index(['batch','dut','sample']).index.isin(selected_keys)
        ]
        color_key = {key: viridis(i / n_keys) for i, key in enumerate(all_keys)}
        for key in all_keys:
            df_key = df_filtered[df_filtered.set_index(['batch','dut','sample']).index == key]
            width_driver = df_key['width_driver'].iloc[0]
            width_load = df_key['width_load'].iloc[0]
            k = width_driver / width_load
            Vin = df_key['Vinput'].values[0]
            Vout = df_key['Voutput'].values[0]
            Vdd = df_key['Vdd'].values[0]
            Vm = df_key['Vm'].values[0]
            gain = df_key['gain'].values[0]
            intersection = Vdd/2 + Vm*gain
            ax.plot(Vin, Vout, '-', color=plasma(width_load/45), label=f'{width_driver}/{width_load}')
            ax.hlines(y=np.max(Vout), linestyle='-', color='k', alpha = 0.3, xmin=-1.25, xmax=-0.25)
            ax.hlines(y=np.min(Vout), linestyle='-', color='k', alpha = 0.3, xmin=-1.25, xmax=3)
            ax.hlines(y=Vdd/2, linestyle='-', color='k', alpha = 0.3, xmin=-1.25, xmax=Vm)
            ax.vlines(x=Vm, linestyle='-', color='k', alpha = 0.3, ymin=-0.25, ymax=Vdd/2)
            ax.vlines(x=Vdd, linestyle='-', color='k', alpha = 0.3, ymin=-0.25, ymax=np.min(Vout))
            ax.vlines(x=np.min(Vout), linestyle='-', color='k', alpha = 0.3, ymin=-0.25, ymax=np.max(Vout))
            ax.scatter(Vm, Vdd/2, color='k', marker='x', s=10, alpha=1)
            ax.plot(Vin, -gain*np.array(Vin) + intersection, '--', color='k', alpha = 1)
            ax.text(Vm, Vdd, r'Gain', fontsize=5, verticalalignment='bottom', horizontalalignment='left')
            # pick a reference point (center is usually best)
            # x0 = np.mean(Vm)
            # y0 = -gain * x0 + intersection
            # # define horizontal step
            # dx = 0.005 * (np.max(Vin) - np.min(Vin))  # adjustable
            # # slope gives vertical step
            # dy = -gain * dx
            # # triangle
            # ax.plot([x0, x0 + dx], [y0, y0], color='k')          # Δx
            # ax.plot([x0 + dx, x0 + dx], [y0, y0 + dy], color='k') # Δy

        ax.text(0.75, 0.95, r'Depletion-load' + '\n' + r'Inverter', fontsize=6, verticalalignment='top', horizontalalignment='center', transform=ax.transAxes)

        # ax.text(0.825, 0.7, f'Circuit \n Schematic', fontsize=6, verticalalignment='top', horizontalalignment='center', transform=ax.transAxes)

        ax.set_xlabel(r'Input Voltage, $V_\mathsf{in}$ [V]', fontsize=8)
        ax.set_ylabel(r'Output Voltage, $V_\mathsf{out}$ [V]', fontsize=8)
        ax.set_yticks([np.min(Vout),Vdd/2,Vdd])
        ax.set_yticklabels([r'$V_\mathsf{out,low}$', r'$V_\mathsf{DD}/2$', r'$V_\mathsf{out,high}$' + '\n' + r'$\left(= V_\mathsf{DD}\right)$'], fontsize=6)
        ax.set_xticks([np.min(Vout),Vm, Vdd])
        ax.set_xticklabels([r'$V_\mathsf{out,low}$   ', r'$V_\mathsf{M}$', r'$V_\mathsf{out,high}$'], fontsize=6)
        ax.set_xlim(-1, 4)
        ax.set_ylim(-0.25, 3.25)
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=6, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        plt.savefig(script_dir+"/figures/inverter_VTC.svg", bbox_inches=None, transparent=True)
        plt.close()

    if 0: # Plot inverter gain for different Vdd
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','inv-transfer_TUWien_planar_hbn-encapsulated.csv'))
        df = df[(df['dut'] == 'INV4A1t1') & (df['temp'] == '300K') & (df['sample'] == 4)]
        for c in ['Voutput','Vinput','Voutput_fit','Vinput_fit','dVoutdVin','dVoutdVin_fit']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(1,1,figsize=(2.7, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.6
        right_in  = 0.1
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            wspace = 0.0/fig_width,
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height,
        )

        # Inset
        axins = ax.inset_axes([0.475, 0.3, 0.48, 0.48])
        axins.tick_params(axis='x',
            # which='both',
            # bottom=False,
            # labelbottom=False,
            labelsize=5
        )
        axins.tick_params(axis='y', labelsize=5)

        Vdd_array = df['Vdd'].sort_values().unique()
        Vdd_array_gain = [3.5, 4, 4.5, 5.0] # [1, 1.5, 2.0, 2.5, 3.0] #
        colors = plasma(np.linspace(0.1, 0.9, len(Vdd_array)))
        Vdd_colors = {Vdd: colors[i] for i, Vdd in enumerate(Vdd_array)}
        for Vdd in Vdd_array[::-1]:
            df_Vdd = df[df['Vdd'] == Vdd]
            Vin = df_Vdd['Vinput'].values[0]
            Vout = df_Vdd['Voutput'].values[0]
            Vin_fit = df_Vdd['Vinput_fit'].values[0]
            Vout_fit = df_Vdd['Voutput_fit'].values[0]
            dVoutdVin = df_Vdd['dVoutdVin'].values[0]
            dVoutdVin_fit = df_Vdd['dVoutdVin_fit'].values[0]
            ax.plot(Vin, Vout, '.', color=Vdd_colors[Vdd], label=f'{Vdd:.1f} V', markeredgewidth=0.00, markeredgecolor="#13073A", markerfacecolor=Vdd_colors[Vdd], alpha = 0.7)
            ax.plot(Vin_fit, Vout_fit, '-', color=Vdd_colors[Vdd], alpha = 1)
            if Vdd in Vdd_array_gain:
                axins.plot(Vin, dVoutdVin, '.', color=Vdd_colors[Vdd], markeredgewidth=0.1, markeredgecolor="#13073A", markerfacecolor=Vdd_colors[Vdd], alpha = 0.5)
                axins.plot(Vin_fit, dVoutdVin_fit, '-', color=Vdd_colors[Vdd], alpha = 1, label=f'{Vdd:.1f} V')
                ymin, ymax = axins.get_ylim()
                axins.vlines(x=Vin_fit[np.argmax(dVoutdVin_fit)], ymin=ymin, ymax=np.max(dVoutdVin_fit), color=Vdd_colors[Vdd], linestyle='--', alpha=0.7)
        
        x1, x2 = 0.525, 0.675
        axins.set_xlim(x1, x2)
        axins.set_ylim(0, 140)
        #axins.set_xticks([0.5, 0.55, 0.6, 0.65])
        #axins.set_xticklabels([0.50, 0.55, 0.60, 0.65])
        ax.axvspan(x1, x2, color='gray', alpha=0.15)
        con1 = ConnectionPatch(
            xyA=(x1, ax.get_ylim()[0]), coordsA=ax.transData,
            xyB=(0, 0), coordsB=axins.transAxes,
            color="0.5"
        )

        axins.legend(fontsize=4, loc='upper right', framealpha=0.0, title=r'$V_\mathsf{DD}$', title_fontsize=4, handlelength=0.6, bbox_to_anchor=(1.00, 1.00))

        con2 = ConnectionPatch(
            xyA=(x2, ax.get_ylim()[0]), coordsA=ax.transData,
            xyB=(1, 0), coordsB=axins.transAxes,
            color="0.5"
        )
        axins.set_title(r'Gain, $\left|\mathrm{d}V_\mathsf{out}/\mathrm{d}V_\mathsf{in}\right|$', fontsize=6, pad=2)

        ax.add_artist(con1)
        ax.add_artist(con2)
        ax.set_xlabel(r'Input Voltage, $V_\mathsf{in}$ [V]', fontsize=8)
        ax.set_ylabel(r'Output Voltage, $V_\mathsf{out}$ [V]', fontsize=8)
        ax.set_xlim(-1.25, 4.25)
        ax.set_ylim(-0.25, 5.75)
        #ax.set_title('Voltage Transfer Characteristic', fontsize=8)
        #ax.legend(fontsize=6, loc='upper right', framealpha=0.0, title=r'$V_\mathsf{DD}$', title_fontsize=6)
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=6, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        # img_inverter = mpimg.imread('figures/inverter.png')
        # imagebox = OffsetImage(img_inverter, zoom=0.05)
        # ab = AnnotationBbox(
        #     imagebox,
        #     (0.8, 0.8),               # position
        #     xycoords='axes fraction',
        #     frameon=False
        # )
        # ax.add_artist(ab)
        # Vdd arrow
        min_Vdd = np.min(Vdd_array)
        max_Vdd = np.max(Vdd_array)
        ax.annotate('', xy=(-0.5, 0.25), xytext=(-0.5, 5.25),
            arrowprops=dict(arrowstyle='<|-', color='black',shrinkA=0, shrinkB=0))
        ax.text(-0.5, 0.25, rf'$V_\mathsf{{DD}}$ = {min_Vdd:.1f} V', fontsize=5, verticalalignment='top', horizontalalignment='center')
        ax.text(-0.5, 5.25, rf'$V_\mathsf{{DD}}$ = {max_Vdd:.1f} V', fontsize=5, verticalalignment='bottom', horizontalalignment='center')
        plt.savefig(script_dir+"/figures/inverter_gain.png", bbox_inches=None, dpi=600, transparent=True)
        plt.close()
    
    ########## Hysteresis plots ##########
    if 0: # Hysteresis IdVg example
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4) & (df['nom_freq'] == 0.001) & (df['precondition'] == False)]
        for c in ['Id','Vg']:
            if c in df.columns:
                df[c] = df[c].map(json5.loads)

        width = df['width'].iloc[0]
        Vmax = df['Vmax'].iloc[0]
        Vmin = df['Vmin'].iloc[0]

        fig, ax = plt.subplots(figsize=(2.3, 1.75),constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.5
        right_in  = 0.05
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height
        )

        # Get unique sweep frequencies
        freqs = df['nom_freq'].unique()
        colors = plasma(np.linspace(0, 1, len(freqs)))
        freq_to_color = {freq: colors[idx] for idx, freq in enumerate(sorted(freqs))}

        # Inset
        axins = ax.inset_axes([0.35, 0.15, 0.6, 0.3])
        axins.tick_params(axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
        Vg = df['Vg'].values[0]
        Id = df['Id'].values[0]
        LenSweepData = len(Vg)
        UpDownSplit = int(np.floor(LenSweepData/2))
        Id_UpDown=np.array([Id[0:UpDownSplit+1],
                np.flip(Id[UpDownSplit:LenSweepData])])
        Vg_UpDown=np.array([Vg[0:UpDownSplit+1],
                            np.flip(Vg[UpDownSplit:LenSweepData])])
        Vth_up = df['Vth_up'].iloc[0]
        Vth_down = df['Vth_down'].iloc[0]
        ax.plot(Vg_UpDown[0], Id_UpDown[0]/width*1e6, '-', 
                label=f'up sweep', color = color_up)
        ax.plot(Vg_UpDown[1], Id_UpDown[1]/width*1e6, '-', 
                label=f'down sweep', color = color_down)
        ax.scatter([Vth_up], [df['Ith'].iloc[0]/width*1e6], color=color_up, marker='+')
        ax.scatter([Vth_down], [df['Ith'].iloc[0]/width*1e6], color=color_down, marker='+')
        axins.plot(Vg_UpDown[0], Id_UpDown[0]/width*1e6, '-', color = color_up)
        axins.plot(Vg_UpDown[1], Id_UpDown[1]/width*1e6, '-', color = color_down)

        ax.set_yscale("log")
        ax.set_ylim(8e-7, 2e-1)
        ax.set_xlabel(r'Gate Voltage, $V_\mathsf{G}$ [V]')
        ax.set_ylabel(r'Drain Current, $I_\mathsf{D}$ [$\mu$A/$\mu$m]')
        axins.set_yscale("log")
        axins.set_xticks([])
        axins.set_yticks([])
        x1, x2 = 2.85, 3.3
        y1, y2 = 4e-3, 7e-3
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

        # Add device info
        device_text = (
            r"$T$ = 300 K\n"
            + rf"$W/L$ = $\frac{{{df['width'].iloc[0]:.0f}\,\mu m}}{{{df['length'].iloc[0]:.0f}\,\mu m}}$\n"
            + rf"$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]} V\n"
            + rf"$r_\mathsf{{sweep}}$ = {df['nom_freq'].iloc[0]*(Vmax - Vmin):.2f} V/s"
        )
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #        fontsize=6, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        # Legend with min and max frequencies
        sorted_freqs = sorted(freqs)
        ax.legend(fontsize=6, loc='upper left',
             handlelength=1.5)

        #ax.set_title(r'Hysteresis $I_\mathsf{D}-V_\mathsf{G}$ curve')
        ax.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.5,zorder=0)
        ax.text(-0.5,df['Ith'].iloc[0]/width*0.9e6, r'$I_\mathsf{th}$-criterion', fontsize=6, va='top', ha='left')
        axins.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.2,zorder=0)
        current_max = np.max(np.array(Id)/width*1e6)
        current_decade = current_max / 10
        vg_max = Vg[np.argmax(np.array(Id)/width*1e6)]
        x1 = df['Vth_down'].iloc[0]
        x2 = df['Vth_up'].iloc[0]
        y  = df['Ith'].iloc[0]/width*1e6
        xm = 0.5 * (x1 + x2)
        axins.annotate(
            '',
            xy=(xm, y),
            xytext=(x1, y),
            arrowprops=dict(arrowstyle='<|-', color='black', shrinkA=0, shrinkB=6)
        )
        axins.annotate(
            '',
            xy=(xm, y),
            xytext=(x2, y),
            arrowprops=dict(arrowstyle='<|-', color='black', shrinkA=0, shrinkB=6)
        )
        axins.text(
            xm, y*0.975,
            r'$V_\mathsf{H}$',
            ha='center',
            va='center',
            fontsize=6,
        )
        #axins.text(0.5*(df['Vth_down'].iloc[0] + df['Vth_up'].iloc[0]), df['Ith'].iloc[0]/width*1e6, r'$V_\mathsf{H}$', fontsize=6, va='bottom',ha='center')
        axins.scatter([Vth_up], [df['Ith'].iloc[0]/width*1e6], color=color_up, marker='+')
        axins.scatter([Vth_down], [df['Ith'].iloc[0]/width*1e6], color=color_down, marker='+')
        axins.text(df['Vth_up'].iloc[0], df['Ith'].iloc[0]/width*1e6, r'$V_\mathsf{th,up}$', fontsize=6, va='bottom', ha='right', color=color_up)
        axins.text(df['Vth_down'].iloc[0]-0.02, df['Ith'].iloc[0]/width*0.96e6, r'$V_\mathsf{th,down}$', fontsize=6, va='top', ha='left', color=color_down)
        plt.savefig(script_dir+"/figures/hysteresis_IdVg_example.pdf", bbox_inches=None)
        plt.close()

    if 0: # Id vs t Hysteresis
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4) & (df['precondition'] == False)]
        for c in ['Id','Vg','time']:
            if c in df.columns:
                df[c] = df[c].map(json5.loads)
        
        width = df['width'].iloc[0]
        Vmin = df['Vmin'].iloc[0]
        Vmax = df['Vmax'].iloc[0]

        fig, ax = plt.subplots(figsize=(2.3, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.2
        right_in  = 0.3
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height
        )

        # Get unique sweep frequencies
        freqs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        # freqs = np.unique(df['nom_freq'])
        df['nom_freq'] = np.around(df['nom_freq'], decimals=4)
        colors = plasma(np.linspace(0, 1, len(freqs)))
        freq_to_color = {freq: colors[idx] for idx, freq in enumerate(sorted(freqs))}
        
        # Add rectangles to indicate sweep direction
        ax.add_patch(Rectangle((0, 8e-7), 0.5, 1e-0-8e-7, facecolor=color_up, alpha=0.1))
        ax.add_patch(Rectangle((0.5, 8e-7), 0.5, 1e-0-8e-7, facecolor=color_down, alpha=0.1))
        ax.text(0.25, 0.975, r'up sweep', fontsize=6, va='top', ha='center', color=color_up, transform=ax.transAxes)
        ax.text(0.75, 0.975, r'down sweep', fontsize=6, va='top', ha='center', color=color_down, transform=ax.transAxes)

        # Inset
        axins = ax.inset_axes([0.30, 0.35, 0.4, 0.3])
        axins.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,labelleft=False)
        ax.patch.set_zorder(0)
        axins.set_zorder(100)
        axins.set_facecolor(opaque_equivalent(color=color_down, alpha=0.1))
        axins.patch.set_alpha(1)
        axins.patch.set_visible(True)

        ax2 = ax.twinx()
        for freq in sorted(freqs):
            df_freq = df[df['nom_freq'] == freq]
            Vg = df_freq['Vg'].values[0]
            Id = df_freq['Id'].values[0]
            time = df_freq['time'].values[0]
            sweep_time = np.max(time) - np.min(time)
            ax.plot((time-np.min(time))/sweep_time, Id/width*1e6, '-', color =freq_to_color[freq])
            axins.plot((time-np.min(time))/sweep_time, Id/width*1e6, '-', color =freq_to_color[freq])

        # Sweep of the gate voltage
        idx_max = np.argmax(Vg)
        x = (time - np.min(time)) / sweep_time
        ax2.step(x[:idx_max+1], Vg[:idx_max+1], where='mid', color=color_up)
        ax2.step(x[idx_max:], Vg[idx_max:], where='mid', color=color_down)

        ax.set_yscale("log")
        ax.set_ylim(8e-7, 2e-1)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r'Normalized time, $t/t_\mathsf{sw}$')
        #ax.set_ylabel(r'Drain current, $I_\mathsf{D}$ [$\mu$A/$\mu$m]')
        # ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(axis='y', which='minor', left=False)
        ax.set_yticks([df['Ith'].iloc[0]/width*1e6])
        ax.set_yticklabels([r'$I_\mathsf{th}$'], fontsize=6)
        ax2.set_ylim(df['Vmin'].iloc[0]-0.1, df['Vmax'].iloc[0]*4)
        #ax2.set_ylabel(r'Gate voltage, $V_\mathsf{G}$ [V]')
        # ax2.tick_params(axis='y', which='both', right=False, labelright=False)
        ax2.set_yticks([Vmin, Vmax])
        ax2.set_yticklabels([r'$V_\mathsf{min}$', r'$V_\mathsf{max}$'], fontsize=6)
        axins.set_yscale("log")
        axins.set_xticks([])
        axins.set_yticks([])
        x1, x2 = 0.67, 0.835
        y1, y2 = 3e-3, 7e-3
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5",alpha=0.7)

        ax.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.5)
        axins.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.2)
        ax2.axhline(Vmax, linestyle='--',xmin=0.5, xmax=1, color = 'k', alpha=0.5)
        current_max = np.max(np.array(Id)/width*1e6)
        current_decade = current_max / 10
        vg_max = Vg[np.argmax(np.array(Id)/width*1e6)]
        axins.annotate('', xy=(0.715, 4e-3), xytext=(0.79, 4e-3), arrowprops=dict(arrowstyle='-|>', color='black', shrinkA=0, shrinkB=0))
        # axins.text(0.755, 4.5e-3, r'$r_\mathsf{sweep}$', fontsize=7, va='bottom',ha='center')
        # Text sweep rate limits
        axins.text(0.72, 4e-3, r'$t_\mathsf{sw}^{\max}$', fontsize=5, va='center', ha='right')
        axins.text(0.8, 4e-3, r'$t_\mathsf{sw}^{\min}$', fontsize=5, va='center', ha='left')
        # Axis 2 texts
        # ax2.text(0.1, 3.5, r'$V_\mathsf{G}$- up sweep', fontsize=6, va='top', ha='left', color=color_up,rotation=30)
        # ax2.text(0.9, 3.5, r'$V_\mathsf{G}$- down sweep', fontsize=6, va='top', ha='right', color=color_down,rotation=-30)
        ax2.text(0.35, 4.25, r'$V_\mathsf{G}$', fontsize=6, va='bottom', ha='right')
        ax.text(0.34, 1.5e-2, r'$\log(I_\mathsf{D})$', fontsize=6, va='bottom', ha='right')
        # ax2.text(0.5, 6, r'$V_\mathsf{max}$', fontsize=6, va='bottom', ha='center')
        # ax2.text(1, 0, r'$V_\mathsf{min}$', fontsize=6, va='bottom', ha='center')
        plt.savefig(script_dir+"/figures/hysteresis_time_example.pdf", bbox_inches=None)
        plt.close()

    if 0: # Hysteresis Vth and DeltaVth vs frequency (one figure version)
        ###############################################################################################
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4)]

        width = df['width'].iloc[0]
        Vmax = df['Vmax'].iloc[0]
        Vmin = df['Vmin'].iloc[0]

        # Create figure with two stacked axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.3, 1.75), sharex=True, constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.43
        right_in  = 0.05
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height,
            hspace = 0.08
        )

        # =========================
        # Top plot: Vth
        # =========================
        ax1.plot(df['freq'], df['Vth_up'], '^',
                markeredgecolor="#13073A",
                markerfacecolor=color_up, label=r'$V_{th,\mathrm{up}}$')

        ax1.plot(df['freq'], df['Vth_down'], 'v',
                markeredgecolor="#13073A",
                markerfacecolor=color_down, label=r'$V_{th,\mathrm{down}}$')

        ax1.plot(df['freq'], df['Vth_up_fit'], '-',
                color=color_up, alpha=0.7)

        ax1.plot(df['freq'], df['Vth_down_fit'], '-',
                color=color_down, alpha=0.7)

        ax1.set_ylabel(r'$V_\mathsf{th}$ [V]', fontsize=8)
        ax1.set_xscale('log')
        ax1.legend(fontsize=6, loc='best')

        # Remove x labels on top plot
        ax1.tick_params(labelbottom=False)
        # Elipse to indicate the region of long term degradation
        # ellipse = Ellipse(
        #     (0.18, 0.55),
        #     width=0.3,
        #     height=1.2/aspect,
        #     transform=ax1.transAxes,
        #     edgecolor='k',
        #     facecolor='none',
        #     linewidth=1,
        #     angle= 50,
        #     alpha=0.7
        # )
        # ax1.add_patch(ellipse)
        # point = plt.ginput(2)
        # x_click, y_click = point[0]
        # x_click2, y_click2 = point[1]
        # print(f"You clicked at: ({x_click:.2f}, {y_click:.2f})")
        # print(f"You clicked at: ({x_click2:.2f}, {y_click2:.2f})")
        ax1.annotate('Long-term shift', xy=(0.01, 2.86), xytext=(0.06, 3.01),
            arrowprops=dict(arrowstyle='-|>', color='black'),
            fontsize=5, va='center', ha='center')
        # =========================
        # Bottom plot: Delta Vth
        # =========================
        ax2.plot(df['freq'], df['DeltaVth'], 'o',
                markeredgecolor="#13073A",
                markerfacecolor="#2E8B57", label=r'$\Delta V_{th}$')

        ax2.plot(df['freq'], df['DeltaVth_fit'], '-',
                color="#2E8B57", alpha=0.7)

        ax2.axhline(df['Ith'].iloc[0] / width * 1e6,
                    linestyle='--', color='k')

        ax2.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=7.5)
        ax2.set_xlabel(r'Frequency, $1/t_\mathsf{sw}$ [Hz]', fontsize=8)
        ax2.set_xscale('log')
        ax2.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax2.xaxis.set_major_formatter(FuncFormatter(make_log_formatter([-2,0,2])))
        ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))

        # =========================
        # Layout adjustments
        # =========================
        plt.subplots_adjust(hspace=0.08)
        fig.align_ylabels()

        # Save figure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(script_dir, "figures", "hysteresis_Vth_DeltaVth_vs_freq.pdf"), bbox_inches=None)

        plt.close()

    if 0: # Plot hysteresis DeltaVth vs freq different Vd constant range
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut']=='2A9t1') & (df['precondition'] == False)]

        fig, ax = plt.subplots(figsize=(3.3, 2.25), constrained_layout=False)
        plt.subplots_adjust(left=0.18, right=0.95, top=0.98, bottom=0.18)
        
        Vmax = df['Vmax'].iloc[0]
        Vmin = df['Vmin'].iloc[0]

        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        # Map colors to vacuum conditions
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                Vd = subset['Vd'].iloc[0]
                ax.plot(subset['freq']*(Vmax-Vmin), subset['DeltaVth'],
                    marker=marker, linestyle=' ',
                    markeredgecolor="#13073A",
                    markerfacecolor=Vd_color(Vd), label=rf'{Vd:.1f} V')

                ax.plot(subset['freq']*(Vmax-Vmin), subset['DeltaVth_fit'], linestyle='-', markersize=8,color=Vd_color(Vd))
        
        ax.axhline(0, linestyle='--', color = 'k')
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc='upper center', framealpha=0, title = r'$V_D$')
        
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K\n$V_{{g,range}}$ = [{subset["Vmin"].iloc[0]:.0f}, {subset["Vmax"].iloc[0]:.0f}] V' 
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=22, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        ax.set_xlabel(r'Sweep rate, $r_\mathsf{sw}$ [V/s]', fontsize=8)
        ax.set_ylabel(r'Hysteresis Width, $V_\mathsf{H}$ [V]', fontsize=8)
        ax.set_xscale('log')
        ax.set_ylim(-0.2, 0.3)
        #ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        #plt.title('Same voltage range')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_differentVd_constant_range.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot hysteresis DeltaVth different Vd adjusted ranges
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut']=='1A15t1') & (df['precondition'] == False) & (df['sample'].isin([5,6,7,8,9]))]

        fig, ax = plt.subplots(figsize=(3.3, 2.25), constrained_layout=False)
        plt.subplots_adjust(left=0.18, right=0.95, top=0.98, bottom=0.18)

        Vmin = df['Vmin'].iloc[0]
        Vmax = df['Vmax'].iloc[0]

        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        # Map colors to vacuum conditions
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        dut_labels = []
        dut_handles = []
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                Vd = subset['Vd'].iloc[0]
                dut = key[1]
                line_dut, = ax.plot(subset['freq']*(Vmax - Vmin), subset['DeltaVth'],
                    marker=marker, linestyle=' ', markeredgecolor="#13073A",
                    markerfacecolor=Vd_color(Vd), label=rf'{Vd:.1f} V')
                dut_handles.append(line_dut)
                dut_labels.append(rf'$V_\text{{D}}$ = {Vd:.1f} V, $V_{{g,\text{{range}}}}$=[{subset["Vmin"].iloc[0]:.2f}, {subset["Vmax"].iloc[0]:.2f}] V' )

                ax.plot(subset['freq']*(Vmax - Vmin), subset['DeltaVth_fit'], linestyle='-',color=Vd_color(Vd))
        
        ax.axhline(0, linestyle='--', color = 'k')
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # First legend for Vd
        legend1 = ax.legend(
            handles=[Line2D([0], [0], color=h.get_markerfacecolor(),  linewidth=3) for h in by_label.values()],
            labels=by_label.keys(),
            fontsize=textSizeLegend,
            loc='best',
            framealpha=0.9,
            title=r'$V_\mathsf{D}$',
            handlelength=1.0
        )
        # ax.add_artist(legend1)
        
        # Second legend for DUTs
        ax.legend(dut_handles, dut_labels, fontsize=5, 
              loc='upper left', framealpha=0,bbox_to_anchor=(0.2, 1))
                
        ax.set_xlabel(r'Sweep rate, $r_\mathsf{sw}$ [V/s]', fontsize=8)
        ax.set_ylabel(r'Hysteresis Width, $V_\mathsf{H}$ [V]', fontsize=8)
        ax.set_xscale('log')
        ax.set_ylim(-0.2, 0.3)
        #ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        # device_text = r'$T$ = 300 K' + '\n' + rf'$E_{{od}} = {df["Eod"].iloc[0]:.2f}$ MV/cm'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=6, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_differentVd_adjusted_range.pdf", bbox_inches=None, transparent=True)
        plt.close()

    if 0: # Plot hysteresis DeltaVth vs freq different Vd (one figure version)
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut']=='2A9t1') & (df['precondition'] == False)]

        fig, ax = plt.subplots(1, 2, figsize=(4, 1.75), constrained_layout=False, sharey=True, sharex=True)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.5
        right_in  = 0.05
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height,
            wspace = 0.1
        )
        
        Vmax = df['Vmax'].iloc[0]
        Vmin = df['Vmin'].iloc[0]

        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))
        n_keys = len(all_keys)

        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']
        Vd_values = sorted(df['Vd'].unique())

        dut_handles = []
        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                Vd = subset['Vd'].iloc[0]
                line_dut, = ax[0].plot(subset['freq'], subset['DeltaVth'],
                    marker=marker, linestyle=' ',
                    markeredgecolor="#13073A",
                    markerfacecolor=Vd_color(Vd), label=rf'{Vd:.1f} V')
                
                dut_handles.append(line_dut)

                ax[0].plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-',color=Vd_color(Vd))
        
        ax[0].axhline(0, linestyle='--', color = 'k')
        
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K\n$V_{{g,range}}$ = [{subset["Vmin"].iloc[0]:.0f}, {subset["Vmax"].iloc[0]:.0f}] V' 
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=22, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        ax[0].set_ylabel(r'Hysteresis Width, $V_\mathsf{H}$ [V]', fontsize=8)
        ax[0].set_xscale('log')
        ax[0].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[0].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([-2,0,2])))
        ax[0].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        # ax[0].text(0.5, 0.85, rf'$V_\mathsf{{G,range}}$ = [{subset["Vmin"].iloc[0]:.0f}, {subset["Vmax"].iloc[0]:.0f}] V', transform=ax[0].transAxes, fontsize=5, va='top', ha='center')
        ax[0].legend(
            [tuple(dut_handles)],
            [rf'$V_\mathsf{{G,range}}$ = [{subset["Vmin"].iloc[0]:.0f}, {subset["Vmax"].iloc[0]:.0f}] V'],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            handlelength=3.5,
            fontsize=5,
            frameon=False,
            markerscale=0.5,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.85)
        )
        #ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)

        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut']=='1A15t1') & (df['precondition'] == False) & (df['sample'].isin([5,6,7,8,9]))]

        Vmin = df['Vmin'].iloc[0]
        Vmax = df['Vmax'].iloc[0]

        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        # Map colors to vacuum conditions
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        dut_labels = []
        dut_handles = []
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                Vd = subset['Vd'].iloc[0]
                dut = key[1]
                ax[1].plot(subset['freq'], subset['DeltaVth'],
                    marker=marker, linestyle=' ', markeredgecolor="#13073A",
                    markerfacecolor=Vd_color(Vd), label=rf'{Vd:.1f} V')
                line_dut = Line2D([0], [0], marker=marker, linestyle=' ', markeredgecolor="#13073A", markerfacecolor=Vd_color(Vd), markersize=2)
                dut_handles.append(line_dut)
                dut_labels.append(rf'$V_\mathsf{{G,range}}$=[{subset["Vmin"].iloc[0]:.2f}, {subset["Vmax"].iloc[0]:.2f}] V')

                ax[1].plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-',color=Vd_color(Vd))
        
        ax[1].axhline(0, linestyle='--', color = 'k')
        # Remove duplicate legend entries
        handles, labels = ax[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # First legend for Vd
        # legend1 = ax[1].legend(
        #     handles=[Line2D([0], [0], color=h.get_markerfacecolor(),  linewidth=3) for h in by_label.values()],
        #     labels=by_label.keys(),
        #     fontsize=5,
        #     loc='best',
        #     framealpha=0.9,
        #     title=r'$V_\mathsf{D}$',
        #     handlelength=1.0,
        #     title_fontsize=5
        # )
        # ax.add_artist(legend1)
        
        # Second legend for DUTs
        ax[1].legend(dut_handles, dut_labels, fontsize=5, 
              loc='upper left', framealpha=0, bbox_to_anchor=(0.2, 1))
                
        #ax[1].set_xlabel(r'Frequency, $1/t_\mathsf{sw}$ [Hz]', fontsize=8)
        fig.supxlabel(r'Frequency, $1/t_\mathsf{sw}$ [Hz]', fontsize=8)
        ax[1].set_xscale('log')
        ax[1].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[1].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([-2,0,2])))
        ax[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        ax[1].set_ylim(-0.2, 0.4)
        #ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        # device_text = r'$T$ = 300 K' + '\n' + rf'$E_{{od}} = {df["Eod"].iloc[0]:.2f}$ MV/cm'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=6, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        Vd_handles = []
        Vd_labels = []
        for Vd in Vd_values:
            handle = Line2D(
                [0], [0],
                color=Vd_color(Vd),
                linewidth=2,
                linestyle='-'
            )
            label = rf'{Vd:.1f} V'
            Vd_handles.append(handle)
            Vd_labels.append(label)

        Vd_leg = fig.legend(
            Vd_handles,
            Vd_labels,
            fontsize=5,
            loc='upper center',
            framealpha=1,
            title=r'$V_\mathsf{D}$',
            title_fontsize=6,
            bbox_to_anchor=(0.56, 1.00),
            labelspacing=0.2,
            handlelength=1,
            handletextpad=0.4,
            borderpad=0.4
        )
        Vd_leg.set_zorder(10)
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_differentVd.pdf", bbox_inches=None)
        plt.close()
        
    if 0: # Plot hysteresis DeltaVth vs freq duts variability
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS_Eod.csv'))
        df = df[(df['Vd']==1) & (df['precondition'] == False)]

        fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.6
        right_in  = 0.1
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height,
        )
        
        df_groups = {
            k: g for k, g in df.groupby(['batch', 'dut', 'sample'])
            if np.isclose(g['nom_freq'].min(), 1e-3)
        }
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        # Map colors to vacuum conditions
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        dut_labels = []
        dut_handles = []
        all_keys = sorted(all_keys, key=lambda x: df_groups[x]['width'].iloc[0] / df_groups[x]['length'].iloc[0])
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                Vd = subset['Vd'].iloc[0]
                width = subset['width'].iloc[0]
                length = subset['length'].iloc[0]
                Vmax = subset['Vmax'].iloc[0]
                Vmin = subset['Vmin'].iloc[0]
                dut = key[1]
                line_dut, = ax.plot(subset['freq'], subset['DeltaVth'],
                    marker=marker, linestyle=' ',
                    markeredgecolor="#13073A",
                    markerfacecolor=WL_color(width/length), label = (
                    rf"$W/L$ = {subset['width'].iloc[0]:.0f}/{subset['length'].iloc[0]:.0f}; "
                    + f"Array {subset['array'].iloc[0]}; "
                    + f"Meas {subset['sample'].iloc[0]}"
                ))

                ax.plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-', markersize=8,color=WL_color(width/length))
        
        ax.axhline(0, linestyle='--', color = 'k')
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # First legend for Vd
        legend1 = ax.legend(
            handles=handles,
            labels=by_label.keys(),
            fontsize=6,
            loc='best',
            framealpha=0,
            handlelength=1.0
        )
        ax.add_artist(legend1)
        
        # Second legend for DUTs
        ax.legend(dut_handles, dut_labels, fontsize=5, 
              loc='upper right', framealpha=1,bbox_to_anchor=(1.75, 1))
                
        ax.set_ylabel(r'Hysteresis Width, $V_\mathsf{H}$ [V]', fontsize=8)
        ax.set_xscale('log')
        #ax.set_xlim(1e-3,200)
        ax.set_ylim(-0.2, 0.4)
        #ax.grid(True, which='both', alpha=0.3)
        ax.set_xlabel(r'Frequency, $1/t_\mathsf{sw}$ [Hz]', fontsize=8)
        
        device_text = r'$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=6, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        #plt.title('Different devices and voltage ranges')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_duts.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot hysteresis DeltaVth vs freq comparison
        df = pd.read_csv(os.path.join(data_folder,'hbn-encapsulated_vs_non-encapsulated','hyst_hbn-encapsulated_vs_non-encapsulated.csv'))
        df = df[(df['Vd']==0.1) & (df['precondition'] == False)]

        fig, ax = plt.subplots(figsize=(3, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        left_in   = 0.5
        right_in  = 0.05
        top_in    = 0.1
        bottom_in = 0.4
        plt.subplots_adjust(
            left   = left_in / fig_width,
            right  = 1 - right_in / fig_width,
            bottom = bottom_in / fig_height,
            top    = 1 - top_in / fig_height,
        )
        
        df_dict = {
            k: g for k, g in df.groupby(['batch', 'dut', 'sample'])
            if np.isclose(g['nom_freq'].min(), 1e-1)
        }
        all_keys = sorted(set(list(df_dict.keys())))
        n_keys = len(all_keys)

        df_groups = pd.concat(df_dict.values())

        # Average DeltaVth for each batch and frequency
        df_avg = df_groups.groupby(['batch', 'nom_freq']).mean(numeric_only=True).reset_index()

        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '<', '>']
        batch_colors = {
            'TUWien_planar_hbn-encapsulated': '#2E8B57',  # SeaGreen
            'TUWien_planar_20nm': '#1E90FF',  # DodgerBlue
        }
        batch_labels = {
            'TUWien_planar_hbn-encapsulated': 'hBN-enc.',
            'TUWien_planar_20nm': 'non-enc.',
        }
        batch_linestyle = {
            'TUWien_planar_hbn-encapsulated': '-',
            'TUWien_planar_20nm': '-',
        }
        batch_handles = {
            'TUWien_planar_hbn-encapsulated': {},
            'TUWien_planar_20nm': {},
        }

        # Plot each unique (batch, dut, sample) with distinct color & marker
        # dut_labels = []
        # dut_handles = []
        # all_keys = sorted(all_keys, key=lambda x: df_groups[x]['width'].iloc[0] / df_groups[x]['length'].iloc[0])

        for idx, key in enumerate(all_keys):
            marker = markers[idx % len(markers)]
            if key in df_dict:
                subset = df_dict[key]
                Vd = subset['Vd'].iloc[0]
                batch = subset['batch'].iloc[0]
                tox = subset['tox'].iloc[0]*1e-8 # convert to cm
                width = str(subset['width'].iloc[0])
                length = str(subset['length'].iloc[0])
                Vmax = subset['Vmax'].iloc[0]
                Vmin = subset['Vmin'].iloc[0]
                dut = key[1]
                subset = subset.sort_values(by='freq') 
                if batch == 'TUWien_planar_hbn-encapsulated':
                    subset = subset.iloc[:-4]  
                freq = subset['freq']
                DeltaVth = subset['DeltaVth']/tox*1e-6 # Convert to MV/cm
                line_dut, = ax.plot(freq, DeltaVth,
                    marker=marker, linestyle=' ',
                    markeredgecolor="#13073A",
                    markerfacecolor=batch_colors[batch], alpha=0.7) # label=rf'$W/L$ = {subset['width'].iloc[0]:.0f}/{subset['length'].iloc[0]:.0f};') #+f' Array {subset["array"].iloc[0]}; '+ f'Meas {subset["sample"].iloc[0]}' )
                key_wl = width + '/' + length
                if key_wl not in batch_handles[batch]:
                    batch_handles[batch][key_wl] = []
                batch_handles[batch][key_wl].append(line_dut)
                # dut_handles.append(line_dut)
                # dut_labels.append(dut + rf'$V_{{g,range}}$ = [{subset["Vmin"].iloc[0]:.2f}, {subset["Vmax"].iloc[0]:.2f}] V' )

                # ax.plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-', markersize=8,color=batch_colors[batch])

        batch_elements = []
        for batch in df['batch'].unique():
            tox = df[df['batch'] == batch]['tox'].iloc[0]*1e-8 # convert to cm
            batch_elements.append(
                Line2D([0], [0],
                    marker=None,                    # choose your marker
                    linestyle='-',
                    linewidth=2,
                    color=batch_colors[batch],
                    label=batch_labels[batch])
            )
            freq_fit = np.logspace(np.log10(df_avg['nom_freq'].min()), np.log10(df_avg['nom_freq'].max()), 100)
            df_avg_batch = df_avg[df_avg['batch'] == batch].sort_values(by='nom_freq')
            if batch == 'TUWien_planar_hbn-encapsulated':
                df_avg_batch = df_avg_batch.iloc[:-4]  
            freq = df_avg_batch['nom_freq']
            DeltaVth = df_avg_batch['DeltaVth']/tox*1e-6
            DeltaVth_fit,_ = ProcessingLibrary.fit_data_freq(freq, DeltaVth, freq_fit=freq_fit, fit='spline')
            ax.plot(freq_fit, DeltaVth_fit, linestyle=batch_linestyle[batch], color=batch_colors[batch], alpha=1)
        
        ax.axhline(0, linestyle='--', color = 'k', alpha=0.5)
        # First legend for batch
        leg_batch = ax.legend(
            handles=batch_elements,
            fontsize=7,
            loc='upper right',
            frameon=False,
            handlelength=0.8
        )
        # ax.add_artist(legend1)
        
        # Second legend for DUTs
        # ax.legend(dut_handles, dut_labels, fontsize=5, 
        #       loc='upper right', framealpha=1,bbox_to_anchor=(1.75, 1))
                
        ax.set_ylabel(r'$V_\mathsf{H}/t_\mathsf{ox}$ [MV/cm]', fontsize=8)
        ax.set_xscale('log')
        #ax.set_xlim(1e-3,200)
        ax.set_ylim(-1, 1.5)
        #ax.grid(True, which='both', alpha=0.3)
        ax.set_xlabel(r'Frequency, $1/t_\mathsf{sw}$ [Hz]', fontsize=8)
        
        device_text = r'$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=6, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        legend_handles = []
        legend_labels = []
        for batch in batch_handles:
            for wl in batch_handles[batch]:
                handles_tuple = tuple(batch_handles[batch][wl])

                legend_handles.append(handles_tuple)
                width, length = wl.split('/')
                label = rf'$W/L = \frac{{{int(float(width))}\,\mu\mathrm{{m}}}}{{{int(float(length))}\,\mu\mathrm{{m}}}}$'
                legend_labels.append(label) 

        leg = ax.legend(
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            handlelength=3.5,
            fontsize=5,
            frameon=False,
            markerscale=0.5,
            loc='upper center',
            bbox_to_anchor=(0.475, 0.925)
        )

        ax.add_artist(leg_batch)
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_comparison.pdf", bbox_inches=None)
        plt.close()

    ######### BTI plots ###################
    if 0: # Plot BTI DeltaVth vs total time
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_all.csv'))
        # df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1)]
        df = df[~df['cycle'].isin([0])]
        for c in ['Id','Vg']:
            df[c] = df[c].map(safe_json_load)
        width = df['width'].iloc[0]

        groups = df.groupby(['batch', 'dut', 'sample','meas_type'])
        selected_keys = [
            ('TUWien_planar_hbn-encapsulated', '2A13t1', 1, 'OTF'),
            ('TUWien_planar_hbn-encapsulated', '2A1t1', 1, 'MSM')
        ]
        df_list = []
        for k in selected_keys:
            g = groups.get_group(k).copy()

            if k == ('TUWien_planar_hbn-encapsulated', '2A1t1', 1, 'MSM'):
                g = g[~g['cycle'].isin([4,5])]
            elif k == ('TUWien_planar_hbn-encapsulated', '2A13t1', 1, 'OTF'):
                g = g[~g['cycle'].isin([7,8])]

            df_list.append(g)
        df_selected = pd.concat(df_list)

        markers = ['o', '^']
        max_cycles = df_selected['cycle'].max()
        show_precondition = 0
        fig, ax = plt.subplots(1,max_cycles + show_precondition, figsize=(2.3, 1.75), sharey=True, constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        x_left_in   = 0.5
        x_right_in  = 0.05
        x_top_in    = 0.12
        x_bottom_in = 0.4
        # plt.subplots_adjust(wspace=0.00, bottom=0.2, top=0.90, left=0.12, right=0.98)
        plt.subplots_adjust(
            left   = x_left_in / fig_width,
            right  = 1 - x_right_in / fig_width,
            bottom = x_bottom_in / fig_height,
            top    = 1 - x_top_in / fig_height,
            wspace = 0.00
        )

        for key in selected_keys:
            df_dut = df_selected[(df_selected['batch'] == key[0]) & (df_selected['dut'] == key[1]) & (df_selected['sample'] == key[2]) & (df_selected['meas_type'] == key[3])]
            marker = markers[selected_keys.index(key)]
            cycles = sorted(df_dut['cycle'].unique())
            for j, cycle in enumerate(cycles):

                df_cycle = df_dut[df_dut['cycle'] == cycle]
                df_initial = df_cycle[df_cycle['initial'] == True]

                df_cycle_stress = df_cycle[
                    (df_cycle['initial'] != True) &
                    (df_cycle['extra'] != True) &
                    (df_cycle['end'] != True)
                ].copy()
                
                meas_type = df_cycle['meas_type'].iloc[0]
                if meas_type == 'OTF':
                    i = cycle - (1 - show_precondition)
                    tvar = 'tStress'
                    df_cycle_stress = df_cycle_stress.sort_values(by=tvar)
                    t = df_cycle_stress[tvar].values
                    Vth = df_cycle_stress['Vth'].values
                    Vth_initial = df_initial['Vth'].values[0]

                elif meas_type == 'MSM':
                    i = cycle*2 - (1 - show_precondition)
                    tvar = 'tRec'
                    df_cycle_stress = df_cycle_stress.sort_values(by=tvar)
                    t = df_cycle_stress[tvar].values
                    Vth = df_cycle_stress['Vth'].values + 1 # shift up for better visibility
                    Vth_initial = df_initial['Vth'].values[0] + 1


                if j == 0: # only label first cycle for each measurement type
                    if meas_type == 'OTF':
                        ax[i].text(t[len(t)//2], np.max(Vth)+0.1, f'{df_cycle["meas_type"].iloc[0]}', fontsize=6, verticalalignment='bottom',horizontalalignment='center')
                    else:
                        ax[i].text(t[len(t)//2], np.max(Vth)+0.1, f'{df_cycle["meas_type"].iloc[0]}', fontsize=6, verticalalignment='bottom',horizontalalignment='center')

                        ax[i].axhline(Vth_initial, linestyle='--', color=color_precondition, alpha=0.7)

                if cycle == 0:

                    ax[i].scatter(t, Vth, c=t, cmap=cmap_precondition, norm=norm, marker=marker, edgecolors='#13073A', linewidths=0.8)

                else:

                    ax[i].scatter(t, Vth, c=t, cmap=cmap_stress if (i - (1 - show_precondition)) % 2  == 1 else cmap_relax, norm=norm, marker=marker,edgecolors='#13073A', linewidths=0.8)

                    if meas_type == 'MSM':
                        ax[i].axhline(Vth_initial, linestyle='--', color=color_stress if (i - (1 - show_precondition)) % 2  == 1 else color_relax, alpha=1)
                        ax[i-1].axhline(Vth_initial, linestyle='--', color=color_stress, alpha=1)
                    elif meas_type == 'OTF' and (i - (1 - show_precondition)) % 2  == 1:
                        ax[i].axhline(Vth_initial, linestyle='--', color=color_stress, alpha=1)
                        ax[i+1].axhline(Vth_initial, linestyle='--', color=color_relax, alpha=1)


        stress_ind = 1
        relax_ind = 1
        for i in range(max_cycles + show_precondition):
            if i == 0 and show_precondition:
                ax[i].text(0.5, 0.96, f'Pre-conditioning', transform=ax[i].transAxes, fontsize=6, verticalalignment='top',horizontalalignment='center', rotation=90)

                # --- background rectangle ---
                ax[i].axvspan(t.min(), t.max(), color=color_precondition, alpha=0.05)
                ax[i].annotate(
                        '',
                        xy=(0.5, -0.05),
                        xytext=(1e4, -0.05),
                        xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<|-|>', color='k', linewidth=1, shrinkA=0, shrinkB=0),
                    )
                ax[i].text(
                    1e2,
                    -0.075,
                    rf'$t_\mathsf{{precond}}$',
                    transform=ax[i].get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=6
                )
            else:
                if (i - (1 - show_precondition)) % 2  == 1: # Stress
                    ax[i].text(0.5, 0.95, f'Stress\n #{stress_ind}', transform=ax[i].transAxes, fontsize=6, verticalalignment='top',horizontalalignment='center')
                    ax[i].annotate(
                        '',
                        xy=(0.5, -0.1),
                        xytext=(1e4, -0.1),
                        xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<|-|>', color='k', linewidth=1, shrinkA=0, shrinkB=0),
                    )
                    ax[i].text(
                        1e2,
                        -0.125,
                        rf'$t_\mathsf{{str,{stress_ind}}}$',
                        transform=ax[i].get_xaxis_transform(),
                        ha='center',
                        va='top',
                        fontsize=6
                    )
                    stress_ind += 1
                else: # Relax
                    ax[i].text(0.5, 0.9, f'Relax\n #{relax_ind}', transform=ax[i].transAxes, fontsize=6, verticalalignment='top',horizontalalignment='center')
                    ax[i].annotate(
                        '',
                        xy=(0.5, -0.05),
                        xytext=(1e4, -0.05),
                        xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<|-|>', color='k', linewidth=1, shrinkA=0, shrinkB=0),
                    )
                    ax[i].text(
                        1e2,
                        -0.075,
                        rf'$t_\mathsf{{relax,{relax_ind}}}$',
                        transform=ax[i].get_xaxis_transform(),
                        ha='center',
                        va='top',
                        fontsize=6
                    )
                    relax_ind += 1


                # --- background rectangle ---
                ax[i].axvspan(t.min(), t.max(), color=color_stress if (i - (1 - show_precondition)) % 2  == 1 else color_relax, alpha=0.05)

            ax[i].set_xscale('log')
            ax[i].set_ylim(-0.25, 4.5)
            ax[i].set_xlim(0.5, 1e4)

            # Hide right spine except last plot
            if i != max_cycles + show_precondition - 1:
                ax[i].spines['right'].set_visible(False)
                #ax[i].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([0,4])))
                ax[i].tick_params(axis='x', which='both', labelbottom=False)

            # Hide left spine except first plot
            if i != 0:
                #ax[i].spines['left'].set_visible(False)
                ax[i].spines['left'].set_linestyle('-')
                ax[i].spines['left'].set_alpha(0.5)
                ax[i].tick_params(axis='y',left=False,labelleft=False)
                ax[i].tick_params(axis='x', which='both', labelbottom=False)
                #ax[i].xaxis.set_major_formatter(FuncFormatter(custom_log_formatter([4], i+1)))

            # Optional: cleaner ticks
            #ax[i].tick_params(direction='in')
            ax[i].set_facecolor('none')
            #ax[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            # ax[i].tick_params(axis='y', which='both',labelleft=False)
            ax[i].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
            ax[i].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))

        # Set axis labels
        fig.text(0.55, 0.03, r'Time [a.u]', ha='center', fontsize=8)
        ax[0].set_ylabel(r'Threshold Voltage, $V_\mathsf{th}$ [a.u.]', y=0.45, fontsize=8)
        
        legend_elements = [
            Line2D([0], [0], marker='^',markerfacecolor='none', linestyle='none', label='MSM meas.'),
            Line2D([0], [0], marker='o',markerfacecolor='none', linestyle='none', label='OTF meas. '),
            #Line2D([0], [0], color='k', linestyle='--', label=r'$V_\mathsf{th}$ before stress'),
        ]

        fig.legend(
            handles=legend_elements,
            loc='upper center',
            ncol=4,
            fontsize=7,
            framealpha=0.0,
            columnspacing=1,   # ↓ space between columns (default ~2.0)
            handletextpad=0.4,
            bbox_to_anchor=(0.57, 1.04)
        )

        # Add device info
        #device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{stress}}$ = {df["VgStress"].iloc[0]:.2f} V'
        # ax[0].text(0.05, 0.95, device_text, transform=ax[0].transAxes,fontsize=6, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        # sorted_times = sorted(stress_times)
        # ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
        #      [f'$t_{{stress}}$ = {sorted_times[0]} s', f'$t_{{stress}}$ = {sorted_times[-1]} s'],
        #      fontsize=textSizeLegend, loc='lower right', framealpha=0.9, handlelength=1.5)
        
        plt.savefig(script_dir+"/figures/BTI_stress_time.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot BTI IdVg (joint plot with recovery)
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 5)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)

        width = df['width'].iloc[0]

        fig, ax = plt.subplots(1,2,figsize=(2.3, 1.75),sharey=False, constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        x_left_in   = 0.5
        x_right_in  = 0.05
        x_top_in    = 0.12
        x_bottom_in = 0.4
        #plt.subplots_adjust(wspace=0.00, bottom=0.2, top=0.9, left=0.21, right=0.98)
        plt.subplots_adjust(
            left   = x_left_in / fig_width,
            right  = 1 - x_right_in / fig_width,
            bottom = x_bottom_in / fig_height,
            top    = 1 - x_top_in / fig_height,
            wspace = 0.00
        )
        
        # Get unique stress times and assign colors
        stress_times = df[(df['initial'] != True) &
                (df['extra'] != True) &
                (df['end'] != True)]['tStress'].unique()
        colors = plasma(np.linspace(0, 1, len(stress_times)))
        
        for idx, stress_time in enumerate(sorted(stress_times)):
            df_stress = df[df['tStress'] == stress_time]
            vg = df_stress['Vg'].values[0]
            id_vals = df_stress['Id'].values[0]
            ax[0].plot(vg, np.array(id_vals)/width*1e6, '-',label=f'$t_{{stress}}$ = {stress_time} s', color=cmap_stress(idx / len(stress_times)))
            
        ax[0].axhline(df_stress['Ith'].iloc[0]/width*1e6, linestyle='--', color='k', alpha=0.5)
        ax[0].text(-1.2, df_stress['Ith'].iloc[0]/width*1e6, r'$I_\mathsf{th}$', fontsize=5, verticalalignment='center', horizontalalignment='right')

        # Set axis labels
        ax[0].spines['right'].set_linestyle('-')
        ax[0].spines['right'].set_alpha(0.5)
        ax[0].set_yscale('log')
        ax[0].set_ylim(4e-9, 1e-2)
        ax[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[0].yaxis.set_major_formatter(FuncFormatter(make_log_formatter([-7,-5,-3])))
        ax[0].yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        ax[0].set_xlim(-1.15, 2.7)
        ax[0].axvspan(-1.15, 2.7, color=color_stress, alpha=0.05)
        ax[0].text(0.05, 0.95, 'Stress #3', transform=ax[0].transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left')
        ax[0].set_ylabel(r'Drain Current, $I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=8)
        
       # annotate with the time of the first and last stress points
        first_time = sorted(stress_times)[0]
        last_time = sorted(stress_times)[-1]
        ax[0].annotate('', xy=(0.65, 4e-4), xytext=(1.55, 4e-5), arrowprops=dict(arrowstyle='<|-', color='black',shrinkA=0, shrinkB=0))
        ax[0].text(1.3, 1.5e-5, rf'$10^{{ {int(np.log10(last_time))} }}$ s', fontsize=5, va='bottom', ha='left')
        ax[0].text(0.65, 4e-4, rf'$t_\mathsf{{str}}$ = {first_time} s', fontsize=5, va='center', ha='right')

        # Plot BTI IdVg Recovery
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 6)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)

        width = df['width'].iloc[0]
        
        # Get unique stress times and assign colors
        stress_times = df[(df['initial'] != True) &
                (df['extra'] != True) &
                (df['end'] != True)]['tStress'].unique()
        colors = plasma(np.linspace(0, 1, len(stress_times)))
        
        for idx, stress_time in enumerate(sorted(stress_times)):
            df_stress = df[df['tStress'] == stress_time]
            vg = df_stress['Vg'].values[0]
            id_vals = df_stress['Id'].values[0]
            ax[1].plot(vg, np.array(id_vals)/width*1e6, '-',label=f'$t_{{stress}}$ = {stress_time} s', color=cmap_relax(idx / len(stress_times)))
            
        ax[1].axhline(df_stress['Ith'].iloc[0]/width*1e6, linestyle='--', color='k', alpha=0.5)

        # Set axis labels
        ax[1].spines['left'].set_linestyle('-')
        ax[1].spines['left'].set_alpha(0.5)
        ax[1].tick_params(axis='y', which='both', labelleft=False, left=False)
        ax[1].set_yscale('log')
        ax[1].set_ylim(4e-9, 1e-2)
        ax[1].set_xlim(-1.15, 2.7)
        ax[1].axvspan(-1.15, 2.7, color=color_relax, alpha=0.05)
        ax[1].text(0.05, 0.95, 'Relax #3', transform=ax[1].transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left')
        fig.text(0.6, 0.03, r'Gate Voltage, $V_\mathsf{G}$ [V]', ha='center', fontsize=8)
        # ax[1].set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=8)
        
        # annotate with the time of the first and last stress points
        first_time = sorted(stress_times)[0]
        last_time = sorted(stress_times)[-1]
        ax[1].annotate('', xy=(0.6, 5e-4), xytext=(1.55, 4e-5), arrowprops=dict(arrowstyle='-|>', color='black',shrinkA=0, shrinkB=0))
        ax[1].text(0.85, 4e-4, rf'$t_\mathsf{{relax}}$ = $10^{{ {int(np.log10(last_time))} }}$ s', fontsize=5, va='bottom', ha='right')
        ax[1].text(1.4, 2e-5, rf'{first_time} s', fontsize=5, va='center', ha='left')

        
        # Add device info
        # device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{stress}}$ = {df["VgStress"].iloc[0]:.2f} V'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #        fontsize=22, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        # sorted_times = sorted(stress_times)
        # ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
        #      [f'$t_{{stress}}$ = {sorted_times[0]} s', f'$t_{{stress}}$ = {sorted_times[-1]} s'],
        #      fontsize=textSizeLegend, loc='lower right', framealpha=0.9, handlelength=1.5)
        
        plt.savefig(script_dir+"/figures/BTI_IdVg_stressrelax.pdf", bbox_inches=None)
        plt.close()

    if 1: # Plot inverter BTI
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_inv.csv'))
        df = df[(df['dut'] == 'INV4A1t1') & (df['temp'] == '300K') & (df['sample'] == 4) & (df['cycle'] == 1)]
        for c in ['Vinput','Voutput']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(1,2,figsize=(2.3, 1.75),sharey=True, constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        x_left_in   = 0.5
        x_right_in  = 0.05
        x_top_in    = 0.05
        x_bottom_in = 0.4
        #plt.subplots_adjust(wspace=0.00, bottom=0.2, top=0.9, left=0.21, right=0.98)
        plt.subplots_adjust(
            left   = x_left_in / fig_width,
            right  = 1 - x_right_in / fig_width,
            bottom = x_bottom_in / fig_height,
            top    = 1 - x_top_in / fig_height,
            wspace = 0.00
        )
        
        axinset_0 = ax[0].inset_axes([0.6, 0.49, 0.4, 0.4])
        axinset_0.set_xlabel(r'$t_\mathsf{stress}$ [s]', fontsize=5, labelpad=0.75)
        # axinset_0.set_title(r'$V_\mathsf{M}$ [V]', fontsize=5, pad=1.2, x=1)
        axinset_0.set_xscale('log')
        axinset_0.tick_params(axis='both', labelsize=5)
        axinset_0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axinset_0.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        axinset_0.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        axinset_0.xaxis.set_major_formatter(FuncFormatter(make_log_formatter([2,3])))
        fig.text(0.59, 0.95, r'$V_\mathsf{M} [V]$', fontsize=6, va='top', ha='center')

        # Get unique stress times and assign colors
        stress_times = df[(df['initial'] != True) &
                (df['extra'] != True) &
                (df['end'] != True)]['tStress'].unique()
        
        for idx, stress_time in enumerate(sorted(stress_times)):
            df_stress = df[df['tStress'] == stress_time]
            Vin = df_stress['Vinput'].values[0]
            Vout = df_stress['Voutput'].values[0]
            Vm = df_stress['Vm'].values[0]
            ax[0].plot(Vin[5:], np.array(Vout[5:]), '-',label=f'$t_{{stress}}$ = {stress_time} s', color=cmap_stress(idx / len(stress_times)))
            axinset_0.plot(stress_time, Vm, 'o', color=color_stress)
        
        axinset_0.plot(df['tStress'], df['Vm_fit'], '-', color=color_stress, alpha=0.5)

        limits_inset = axinset_0.get_ylim()
            
        ax[0].axhline(df_stress['Vdd'].iloc[0]/2, linestyle='--', color='k', alpha=0.5)
        ax[0].text(-0.7, df_stress['Vdd'].iloc[0]/2-0.1, r'$V_\mathsf{dd}/2$', fontsize=5, verticalalignment='top', horizontalalignment='left')

        # Set axis labels
        ax[0].spines['right'].set_linestyle('-')
        ax[0].spines['right'].set_alpha(0.2)
        # ax[0].set_ylim(4e-9, 1e-2)
        ax[0].set_xlim(-0.75, 5.5)
        ax[0].set_xticks([0, 2, 4])
        ax[0].axvspan(-0.75, 5.5, color=color_stress, alpha=0.05)
        #ax[0].text(0.05, 0.95, 'Stress #4', transform=ax[0].transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left')
        ax[0].set_ylabel(r'Output Voltage, $V_\mathsf{out}$ [V]', fontsize=8)
        
       # annotate with the time of the first and last stress points
        first_time = sorted(stress_times)[0]
        last_time = sorted(stress_times)[-1]
        ax[0].annotate('', xy=(0.65, 1), xytext=(2.3, 1), arrowprops=dict(arrowstyle='<|-', color='black',shrinkA=0, shrinkB=0))
        ax[0].text(2.3, 1, rf'$t_\mathsf{{str}}$ = $10^{{ {int(np.log10(last_time))} }}$ s', fontsize=4.5, va='center', ha='left')
        ax[0].text(0.60, 1, rf'$10^{{ {int(np.log10(first_time))} }}$ s', fontsize=4.5, va='center', ha='right')

        # Plot BTI IdVg Recovery
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_inv.csv'))
        df = df[(df['dut'] == 'INV4A1t1') & (df['temp'] == '300K') & (df['sample'] == 4) & (df['cycle'] == 2)]
        for c in ['Vinput','Voutput']:
            df[c] = df[c].map(json5.loads)
        
        # Get unique stress times and assign colors
        stress_times = df[(df['initial'] != True) &
                (df['extra'] != True) &
                (df['end'] != True)]['tStress'].unique()
        
        ax_inset_1 = ax[1].inset_axes([0.0, 0.49, 0.4, 0.4])
        ax_inset_1.set_xlabel(r'$t_\mathsf{relax}$ [s]', fontsize=5, labelpad=0.75)
        ax_inset_1.set_xscale('log')
        ax_inset_1.yaxis.set_visible(False)
        ax_inset_1.tick_params(axis='both', labelsize=5)
        ax_inset_1.set_ylim(limits_inset)
        ax_inset_1.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax_inset_1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        ax_inset_1.xaxis.set_major_formatter(FuncFormatter(make_log_formatter([3,4])))
        
        for idx, stress_time in enumerate(sorted(stress_times)):
            df_stress = df[df['tStress'] == stress_time]
            Vin = df_stress['Vinput'].values[0]
            Vout = df_stress['Voutput'].values[0]
            Vm = df_stress['Vm'].values[0]
            ax[1].plot(Vin[5:], np.array(Vout[5:]), '-',label=f'$t_{{stress}}$ = {stress_time} s', color=cmap_relax(idx / len(stress_times)))
            ax_inset_1.plot(stress_time, Vm, 'o', color=color_relax)
        
        ax_inset_1.plot(df['tStress'], df['Vm_fit'], '-', color=color_relax, alpha=0.5)
        ax[1].axhline(df_stress['Vdd'].iloc[0]/2, linestyle='--', color='k', alpha=0.5)

        # Set axis labels
        ax[1].spines['left'].set_linestyle('-')
        ax[1].spines['left'].set_alpha(0.2)
        ax[1].tick_params(axis='y', which='both', labelleft=False, left=False)
        # ax[1].set_ylim(4e-9, 1e-2)
        ax[1].set_xlim(-2, 3.5)
        ax[1].set_xticks([-1,1,3])
        ax[1].axvspan(-2, 3.5, color=color_relax, alpha=0.05)
        #ax[1].text(0.05, 0.95, 'Relax #4', transform=ax[1].transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left')
        
        fig.supxlabel(r'Input Voltage, $V_\mathsf{in}$ [V]', ha='center', fontsize=8, x=0.57)
        
        # annotate with the time of the first and last stress points
        first_time = sorted(stress_times)[0]
        last_time = sorted(stress_times)[-1]
        ax[1].annotate('', xy=(0.54, 1), xytext=(1.6, 1), arrowprops=dict(arrowstyle='-|>', color='black',shrinkA=0, shrinkB=0))
        ax[1].text(0.54, 1, rf'$t_\mathsf{{relax}}$ = $10^{{ {int(np.log10(last_time))} }}$ s', fontsize=4.5, va='center', ha='right')
        ax[1].text(1.7, 1, rf'$10^{{ {int(np.log10(first_time))} }}$ s', fontsize=4.5, va='center', ha='left')
        
        plt.savefig(script_dir+"/figures/BTI_inv_stressrelax.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot BTI OTF DeltaVth vs different VgStress
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1)]
        
        fig, ax = plt.subplots(1,2,figsize=(2.3, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        xlabel_space = 0.4
        ylabel_space = 0.5
        right_space = 0.05
        top_space = 0.12
        plt.subplots_adjust(
            wspace = 0.0/fig_width,
            left   = ylabel_space / fig_width,
            right  = 1 - right_space / fig_width,
            bottom = xlabel_space / fig_height,
            top    = 1 - top_space / fig_height,
        )
        #plt.subplots_adjust(wspace=0.0,left=0.2, right=0.95, top=0.98, bottom=0.18)
        
        VgStress_array = [3.0, 4.0, 5.0, 6.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))

        for i,VgStress in enumerate(VgStress_array):
            subset = df[df['VgStress']==VgStress].sort_values(by='tStress')
            ax[0].plot(subset['tStress'], subset['Vth'] - subset['Vth_initial'], 'o', markeredgecolor="#13073A",markerfacecolor=colors[i], label=f'{VgStress:.0f} V' )
            ax[0].plot(subset['tStress'], subset['Vth_fit'] - subset['Vth_initial'].iloc[0], '-',color=colors[i], alpha=0.7)
        
        ax[0].text(0.05, 0.95, 'Stress', transform=ax[0].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='left')
        ax[0].legend(fontsize=5, loc='upper left', framealpha=0.0, title=r'$V_\mathsf{G,str}$', bbox_to_anchor=(0, 0.9),title_fontsize=6)
        ax[0].axhline(0, linestyle='--', color = 'k')
        # xlim = ax[0].get_xlim()
        # ylim = ax[0].get_ylim()
        ax[0].set_ylim(-0.2, 1.4)
        ax[0].set_xlim(0.3, 2e4)
        ax[0].set_xlabel(r'$t_{\mathsf{str}}$ [s]', fontsize=8)
        ax[0].set_ylabel(r'Threshold Shift, $\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
        ax[0].set_xscale('log')
        ax[0].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[0].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([1,3])))
        ax[0].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        ax[0].axvspan(0.1, 1e5, color=color_stress, alpha=0.05)

        
        # Add device info
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{G,rec}}$ = 0.0 V'
        # ax[0].text(0.05, 0.95, device_text, transform=ax[0].transAxes, 
        #        fontsize=5, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        # Recovery plot
        cycle_array = [2,4,6,8]
        for i,cycle in enumerate(cycle_array):
            subset = df[df['cycle']==cycle].sort_values(by='tStress')
            ax[1].plot(subset['tStress'], subset['Vth'] - df[df['cycle']==cycle-1]['Vth_initial'].iloc[0], 'o',markeredgecolor="#13073A", markerfacecolor=colors[i], label=f'{VgStress:.1f} V')
            ax[1].plot(subset['tStress'], subset['Vth_fit'] - df[df['cycle']==cycle-1]['Vth_initial'].iloc[0], '-',
                color=colors[i], alpha=0.7)
        
        ax[1].text(0.95, 0.95, 'Relax', transform=ax[1].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='right')
        ax[1].text(0.95, 0.87, r'$V_\mathsf{{G,relax}}$ = 0 V', transform=ax[1].transAxes, fontsize=5, verticalalignment='top',horizontalalignment='right')

        # ax[1].text(0.05, 0.25,r'$V_\mathsf{{G,rec}}$ = 0 V', transform=ax[1].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='left')
                   
        # Set axis labels
        ax[1].tick_params(axis='both', which='major', left=False, labelleft=False)
        ax[1].axhline(0, linestyle='--', color = 'k')
        ax[1].set_ylim(-0.2, 1.4)
        ax[1].set_xlim(0.3, 2e4)
        ax[1].set_xlabel(r'$t_{\mathsf{relax}}$ [s]', fontsize=8)
        ax[1].set_xscale('log')
        ax[1].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[1].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([1,3])))
        ax[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        ax[1].axvspan(0.1, 1e5, color=color_relax, alpha=0.05)
        

        plt.savefig(script_dir+"/figures/OTF_DeltaVth_strrec_differentVstr.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot BTI MSM DeltaVth all duts vs different VgStress
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_MSM.csv'))
        df = df[(df['VgRemain'] == 0.0) & (df['tStress']==100)]
        
        fig, ax = plt.subplots(figsize=(2.3, 1.75), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        x_bottom_in = 0.4
        x_left_in = 0.5
        x_right_in = 0.05
        x_top_in = 0.05
        plt.subplots_adjust(
            wspace = 0.0/fig_width,
            left   = x_left_in / fig_width,
            right  = 1 - x_right_in / fig_width,
            bottom = x_bottom_in / fig_height,
            top    = 1 - x_top_in / fig_height,
        )

        VgStress_array = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))
        str_colors = {Vg: colors[i] for i, Vg in enumerate(VgStress_array)}
        color_legend = [Line2D([0], [0], color=str_colors[Vg],lw=2,
           label=rf'{Vg:.1f} V') for Vg in VgStress_array]

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        removed_keys = [
            ('TUWien_planar_hbn-encapsulated', '2A1t1', 2),
            ('TUWien_planar_hbn-encapsulated', '1A11t1', 3),
            ('TUWien_planar_hbn-encapsulated', '1A11t1', 6),
            ('TUWien_planar_hbn-encapsulated', '1A13t1', 2),
            ('TUWien_planar_hbn-encapsulated', '1A13t1', 4),
        ]
        all_keys = [k for k in df_groups.keys() if k not in removed_keys]
        n_keys = len(all_keys)
        df_filtered = df[
            ~df.set_index(['batch','dut','sample']).index.isin(removed_keys)
        ]

        markers = ['v', '^', 's', 'D', 'p', '*', 'h']
        used_labels = set()
        #legend_meas = []
        wl_handles = {}
        for i,VgStress in enumerate(VgStress_array[::-1]):
            df_Vstr = df_filtered[(df_filtered['VgStress']==VgStress)]
            df_groups_Vstr = dict(tuple(df_Vstr.groupby(['batch','dut','sample'])))
            df_avg_Vstr = df_Vstr.groupby('tRec').mean(numeric_only=True).reset_index()
            for idx, key in enumerate(all_keys):
                marker = markers[idx % len(markers)]
                if key in df_groups_Vstr:
                    subset = df_groups_Vstr[key]
                    subset = subset.sort_values(by='tRec')
                    width = subset['width'].iloc[0]
                    length = subset['length'].iloc[0]
                    wl = f'{width}/{length}'
                    # Only label once per key
                    if key not in used_labels:
                        # label = rf'$W/L$ = {subset["width"].iloc[0]}/{subset["length"].iloc[0]}; ' \
                        # rf'Array {subset["array"].iloc[0]}; Meas {subset["sample"].iloc[0]}; '
                        # legend_meas.append(
                        #     Line2D([0], [0],
                        #         marker=marker,
                        #         linestyle='None',
                        #         markerfacecolor='none',
                        #         markeredgecolor='#13073A',
                        #         label=label)
                        # )
                        if wl not in wl_handles:
                            wl_handles[wl] = []
                        wl_handles[wl].append(
                            Line2D([0], [0],
                                marker=marker,
                                linestyle='None',
                                markerfacecolor='none',
                                markeredgecolor='#13073A',
                                markeredgewidth=0.45
                                )
                        )
                        used_labels.add(key)

                    ax.plot(subset['tRec'], subset['Vth'] - subset['Vth_initial'],
                        marker=marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=str_colors[VgStress], alpha=0.5)
                
                DeltaVth_avg = df_avg_Vstr['Vth'] - df_avg_Vstr['Vth_initial']
                time_fit = np.logspace(np.log10(df_avg_Vstr['tRec'].min()), np.log10(df_avg_Vstr['tRec'].max()), 100)
                DeltaVth_fit,_ = ProcessingLibrary.fit_data_time(df_avg_Vstr['tRec'], DeltaVth_avg, time_fit=time_fit, fit='powerlaw')
                ax.plot(time_fit, DeltaVth_fit,
                        linestyle='-',
                        markeredgecolor="#13073A",color=str_colors[VgStress]
                        )
                

                    # ax.plot(subset['tRec'], subset['Vth_fit'] - subset['Vth_initial'], linestyle='-', markersize=8,color=str_colors[VgStress])

        # if i == 0:
        #     ylim = ax.get_ylim()
        # else:
        #     ax.set_ylim(ylim)
        ax.axhline(0, linestyle='--', color = 'k')
        device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V\n $t_{{str}}$ = {df["tStress"].iloc[0]} s\n$V_{{G,rec}}$ = {df["VgRemain"].iloc[0]} V'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        ax.set_xlabel(r'$t_{\mathsf{relax}}$ [s]', fontsize=8)
        ax.set_ylabel(r'Threshold Shift, $\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
        ax.set_ylim(-0.2, 1.4)
        ax.set_xlim(0.3, 2e4)
        ax.set_xscale('log')
        ax.axvspan(0.1, 1e5, color=color_relax, alpha=0.05)

        legend_handles = []
        legend_labels = []
        for wl in wl_handles:
            handles_tuple = tuple(wl_handles[wl])

            legend_handles.append(handles_tuple)
            width, length = wl.split('/')
            label = rf'$W/L = \frac{{{int(float(width))}\,\mu\mathrm{{m}}}}{{{int(float(length))}\,\mu\mathrm{{m}}}}$'
            legend_labels.append(label) 

        leg1 = ax.legend(
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            ncol=2,
            handlelength=1.4,      # shorter line
            handletextpad=0.3,     # space between line and text
            columnspacing=0.6,     # space between columns
            labelspacing=0.2,      # vertical spacing
            borderpad=0.2,         # padding inside legend box
            fontsize=4,
            frameon=False,
            markerscale=0.5,
            loc='upper center',
            bbox_to_anchor=(0.45, 1)
        )
        # leg1 =ax.legend(handles=legend_meas, fontsize=4, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0)
        leg2 = ax.legend(
            handles=color_legend,
            fontsize=5,
            loc='upper right',
            framealpha=0.0,
            title=r'$V_\mathsf{G,stress}$',
            handlelength=1.2,
            title_fontsize=6
        )
        ax.add_artist(leg1)
        plt.savefig(script_dir+f"/figures/MSM_DeltaVth_duts.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot BTI DeltaVth hbn-encapsulated_vs_non-encapsulated

        df = pd.read_csv(os.path.join(data_folder,'hbn-encapsulated_vs_non-encapsulated','BTI_hbn-encapsulated_vs_non-encapsulated_MSM.csv'))
        df = df[(df['tStress']==100) & (df['temp'] == '300K') & (df['VgRemain'] == 0.0)]
        
        fig, ax = plt.subplots(2,2,figsize=(2.3, 1.75), constrained_layout=False, sharex=True, sharey=True)
        fig_width, fig_height = fig.get_size_inches()
        xlabel_space = 0.4
        ylabel_space = 0.5
        right_space = 0.05
        top_space = 0.05
        plt.subplots_adjust(
            wspace = 0.1/fig_width,
            hspace = 0.1/fig_height,
            left   = ylabel_space / fig_width,
            right  = 1 - right_space / fig_width,
            bottom = xlabel_space / fig_height,
            top    = 1 - top_space / fig_height,
        )
        ax = ax.flatten()
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        VgStress_array = [4.0, 5.0, 6.0, 7.0]
        # Map colors to vacuum conditions
        batch_colors = {
            'TUWien_planar_hbn-encapsulated': '#2E8B57',  # SeaGreen
            'TUWien_planar_15nm': '#1E90FF',  # DodgerBlue
        }
        batch_labels = {
            'TUWien_planar_hbn-encapsulated': 'hBN-enc.',
            'TUWien_planar_15nm': 'non-enc.',
        }
        markers = ['v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        removed_keys = [
            ('TUWien_planar_15nm','M15', 2),
            ('TUWien_planar_hbn-encapsulated', '2A1t1', 2),
            ('TUWien_planar_hbn-encapsulated', '1A11t1', 3),
            ('TUWien_planar_hbn-encapsulated', '1A11t1', 6),
            ('TUWien_planar_hbn-encapsulated', '1A13t1', 2),
            ('TUWien_planar_hbn-encapsulated', '1A13t1', 4),
        ]
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = [k for k in df_groups.keys() if k not in removed_keys]
        n_keys = len(all_keys)
        markers_keys = {k: markers[i % len(markers)] for i, k in enumerate(all_keys)}

        for i, VgStress in enumerate(VgStress_array):

            df_Vstr = df[df['VgStress'] == VgStress]
            df_dict = {
                k: g for k, g in df_Vstr.groupby(['batch', 'dut', 'sample'])
            }
            df_groups = pd.concat(df_dict.values())
            df_avg = (
                df_groups
                .groupby(['batch', 'tRec','tox'])
                .agg({
                    'DeltaVth': ['mean', 'std'],
                    'Vth': ['mean', 'std'],
                    'Vth_initial': ['mean', 'std'],
                })
                .reset_index()
            )
            ax[i].text(0.99, 0.95, rf'$V_\mathsf{{G,str}}$ = {VgStress:.1f} V', transform=ax[i].transAxes, fontsize=5, verticalalignment='top',horizontalalignment='right')
            for idx, key in enumerate(all_keys):
                if key in df_dict:
                    marker = markers_keys[key]
                    subset = df_dict[key]
                    subset = subset.sort_values(by='tRec')
                    Vd = subset['Vd'].iloc[0]
                    batch = subset['batch'].iloc[0]
                    width = str(subset['width'].iloc[0])
                    length = str(subset['length'].iloc[0])
                    tox = subset['tox'].iloc[0]*1e-8 # convert to cm
                    dut = key[1]
                    DeltaVth = (subset['Vth'] - subset['Vth_initial'])/tox*1e-6 # convert to MV/cm
                    n_meas_batch = df_groups[df_groups['batch'] == batch].groupby(['batch', 'dut', 'sample']).ngroups
                    if n_meas_batch > 1:
                        line_dut, = ax[i].plot(subset['tRec'], DeltaVth,
                        marker=marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=batch_colors[batch], alpha=0.25)
            batch_elements = []
            for batch in df['batch'].unique():
                # n_meas_batch = df_groups[df_groups['batch'] == batch].groupby(['batch', 'dut', 'sample']).ngroups
                # if n_meas_batch > 1:
                tox = df_avg[df_avg['batch'] == batch]['tox'].iloc[0]*1e-8
                batch_elements.append(
                    Line2D([0], [0],
                        marker=None,
                        linestyle='-',
                        linewidth=2,
                        color=batch_colors[batch],
                        label=batch_labels[batch])
                )
                time_fit = np.logspace(np.log10(df_avg['tRec'].min()), np.log10(df_avg['tRec'].max()), 100)
                DeltaVth_avg = (df_avg[df_avg['batch'] == batch][('Vth','mean')] - df_avg[df_avg['batch'] == batch][('Vth_initial','mean')])/tox*1e-6
                DeltaVth_fit,_ = ProcessingLibrary.fit_data_time(df_avg[df_avg['batch'] == batch]['tRec'], DeltaVth_avg, time_fit=time_fit, fit='powerlaw')
                ax[i].plot(df_avg[df_avg['batch'] == batch]['tRec'], DeltaVth_avg, linestyle=' ', marker='o', color=batch_colors[batch], alpha=1)
                ax[i].plot(time_fit, DeltaVth_fit, linestyle='-', color=batch_colors[batch], alpha=1)

            ax[i].set_xscale('log')
            ax[i].set_xlim(0.3, 2e4)
            ax[i].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
            ax[i].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
            ax[i].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([1,3])))
            ax[i].axhline(0, linestyle='--', color = 'k')
            device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V\n $t_{{str}}$ = {df["tStress"].iloc[0]} s\n$V_{{G,rec}}$ = {df["VgRemain"].iloc[0]} V'
            ax[i].axvspan(0.1, 1e5, color=color_relax, alpha=0.05)
            # ax[i].text(0.05, 0.95, device_text, transform=ax[i].transAxes, 
            #     fontsize=6, verticalalignment='top',
            #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

            # if i == 0 or i == 2:
            #     ax[i].set_ylabel(r'$\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
            # if i >= 2:
            #     ax[i].set_xlabel(r'$t_{\mathsf{relax}}$ [s]', fontsize=8)
        
        leg_batch = ax[0].legend(
            handles=batch_elements,
            fontsize=5.5,
            loc='upper left',
            frameon=False,
            handlelength=0.45,
            bbox_to_anchor=(-0.025, 0.94)
        )

        fig.supylabel(r'$\Delta V_{\mathsf{th}}/t_{\mathsf{ox}}$ [MV/cm]', fontsize=8,y=0.57,x=0.02)
        fig.supxlabel(r'Relaxation time, $t_{\mathsf{relax}}$ [s]', fontsize=8, x=0.55)
        #ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
        plt.savefig(script_dir+f"/figures/MSM_DeltaVth_comparison.pdf", bbox_inches=None)
        plt.close()
    
    ############## Inverter plots ##############
    if 0: # Plot inverter VTC and schematic
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','inv-transfer_TUWien_planar_hbn-encapsulated.csv'))
        df = df[(df['temp'] == '300K') & (df['sample'] == 4) & (df['Vdd'] == 3.0)]
        for c in ['Voutput','Vinput','Voutput_fit','Vinput_fit','dVoutdVin','dVoutdVin_fit']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(1,1,figsize=(2.1, 2.0), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        xlabel_space = 0.4
        ylabel_space = 0.6
        right_space = 0.05
        top_space = 0.05
        plt.subplots_adjust(
            wspace = 0.0/fig_width,
            left   = ylabel_space / fig_width,
            right  = 1 - right_space / fig_width,
            bottom = xlabel_space / fig_height,
            top    = 1 - top_space / fig_height,
        )

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        removed_keys = [
            ('TUWien_planar_hbn-encapsulated', 'sinINV3A', 4),
            ('TUWien_planar_hbn-encapsulated', 'INV2A9t1', 4),
        ]
        selected_keys = [
            ('TUWien_planar_hbn-encapsulated', 'INV4A1t1', 4),
        ]
        #all_keys = [k for k in df_groups.keys() if k not in removed_keys]
        all_keys = [k for k in df_groups.keys() if k in selected_keys]
        n_keys = len(all_keys)
        df_filtered = df[
            ~df.set_index(['batch','dut','sample']).index.isin(removed_keys)
            #df.set_index(['batch','dut','sample']).index.isin(selected_keys)
        ]
        color_key = {key: viridis(i / n_keys) for i, key in enumerate(all_keys)}
        for key in all_keys:
            df_key = df_filtered[df_filtered.set_index(['batch','dut','sample']).index == key]
            width_driver = df_key['width_driver'].iloc[0]
            width_load = df_key['width_load'].iloc[0]
            k = width_driver / width_load
            Vin = df_key['Vinput'].values[0]
            Vout = df_key['Voutput'].values[0]
            Vdd = df_key['Vdd'].values[0]
            Vm = df_key['Vm'].values[0]
            gain = df_key['gain'].values[0]
            intersection = Vdd/2 + Vm*gain
            ax.plot(Vin, Vout, '-', color=plasma(width_load/45), label=f'{width_driver}/{width_load}')
            ax.hlines(y=np.max(Vout), linestyle='-', color='k', alpha = 0.3, xmin=-1.25, xmax=-0.25)
            ax.hlines(y=np.min(Vout), linestyle='-', color='k', alpha = 0.3, xmin=-1.25, xmax=3)
            ax.hlines(y=Vdd/2, linestyle='-', color='k', alpha = 0.3, xmin=-1.25, xmax=Vm)
            ax.vlines(x=Vm, linestyle='-', color='k', alpha = 0.3, ymin=-0.25, ymax=Vdd/2)
            ax.vlines(x=Vdd, linestyle='-', color='k', alpha = 0.3, ymin=-0.25, ymax=np.min(Vout))
            ax.vlines(x=np.min(Vout), linestyle='-', color='k', alpha = 0.3, ymin=-0.25, ymax=np.max(Vout))
            ax.scatter(Vm, Vdd/2, color='k', marker='x', s=10, alpha=1)
            ax.plot(Vin, -gain*np.array(Vin) + intersection, '--', color='k', alpha = 1)
            ax.text(Vm, Vdd, r'Gain', fontsize=5, verticalalignment='bottom', horizontalalignment='left')
            # pick a reference point (center is usually best)
            # x0 = np.mean(Vm)
            # y0 = -gain * x0 + intersection
            # # define horizontal step
            # dx = 0.005 * (np.max(Vin) - np.min(Vin))  # adjustable
            # # slope gives vertical step
            # dy = -gain * dx
            # # triangle
            # ax.plot([x0, x0 + dx], [y0, y0], color='k')          # Δx
            # ax.plot([x0 + dx, x0 + dx], [y0, y0 + dy], color='k') # Δy

        ax.text(0.75, 0.95, r'Depletion-load' + '\n' + r'Inverter', fontsize=6, verticalalignment='top', horizontalalignment='center', transform=ax.transAxes)
        ax.text(0.825, 0.7, f'Circuit \n Schematic', fontsize=6, verticalalignment='top', horizontalalignment='center', transform=ax.transAxes)
        ax.set_xlabel(r'Input Voltage, $V_\mathsf{in}$ [V]', fontsize=8)
        ax.set_ylabel(r'Output Voltage, $V_\mathsf{out}$ [V]', fontsize=8)
        ax.set_yticks([np.min(Vout),Vdd/2,Vdd])
        ax.set_yticklabels([r'$V_\mathsf{out,low}$', r'$V_\mathsf{DD}/2$', r'$V_\mathsf{out,high}$' + '\n' + r'$\left(= V_\mathsf{DD}\right)$'], fontsize=6)
        ax.set_xticks([np.min(Vout),Vm, Vdd])
        ax.set_xticklabels([r'$V_\mathsf{out,low}$', r'$V_\mathsf{M}$', r'$V_\mathsf{out,high}$'], fontsize=6)
        ax.set_xlim(-1, 4)
        ax.set_ylim(-0.25, 3.25)
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=6, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        plt.savefig(script_dir+"/figures/inverter_VTC.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot inverter gain for different Vdd
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','inv-transfer_TUWien_planar_hbn-encapsulated.csv'))
        df = df[(df['dut'] == 'INV4A1t1') & (df['temp'] == '300K') & (df['sample'] == 4)]
        for c in ['Voutput','Vinput','Voutput_fit','Vinput_fit','dVoutdVin','dVoutdVin_fit']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(1,1,figsize=(2.2, 2.0), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        xlabel_space = 0.4
        ylabel_space = 0.4
        right_space = 0.05
        top_space = 0.05
        plt.subplots_adjust(
            wspace = 0.0/fig_width,
            left   = ylabel_space / fig_width,
            right  = 1 - right_space / fig_width,
            bottom = xlabel_space / fig_height,
            top    = 1 - top_space / fig_height,
        )

        # Inset
        axins = ax.inset_axes([0.5, 0.3, 0.48, 0.48])
        axins.tick_params(axis='x',
            # which='both',
            # bottom=False,
            # labelbottom=False,
            labelsize=5
        )
        axins.tick_params(axis='y', labelsize=5)

        Vdd_array = df['Vdd'].sort_values().unique()
        Vdd_array_gain = [3.5, 4.0, 4.5, 5.0]
        colors = viridis(np.linspace(0.1, 0.9, len(Vdd_array)))
        Vdd_colors = {Vdd: colors[i] for i, Vdd in enumerate(Vdd_array)}
        for Vdd in Vdd_array[::-1]:
            df_Vdd = df[df['Vdd'] == Vdd]
            Vin = df_Vdd['Vinput'].values[0]
            Vout = df_Vdd['Voutput'].values[0]
            Vin_fit = df_Vdd['Vinput_fit'].values[0]
            Vout_fit = df_Vdd['Voutput_fit'].values[0]
            dVoutdVin = df_Vdd['dVoutdVin'].values[0]
            dVoutdVin_fit = df_Vdd['dVoutdVin_fit'].values[0]
            ax.plot(Vin, Vout, '.', color=Vdd_colors[Vdd], label=f'{Vdd:.1f} V', markeredgewidth=0.00, markeredgecolor="#13073A", markerfacecolor=Vdd_colors[Vdd], alpha = 0.7)
            ax.plot(Vin_fit, Vout_fit, '-', color=Vdd_colors[Vdd], alpha = 1)
            if Vdd in Vdd_array_gain:
                axins.plot(Vin, dVoutdVin, '.', color=Vdd_colors[Vdd], markeredgewidth=0.1, markeredgecolor="#13073A", markerfacecolor=Vdd_colors[Vdd], alpha = 0.5)
                axins.plot(Vin_fit, dVoutdVin_fit, '-', color=Vdd_colors[Vdd], alpha = 1, label=f'{Vdd:.1f} V')
                ymin, ymax = axins.get_ylim()
                axins.vlines(x=Vin_fit[np.argmax(dVoutdVin_fit)], ymin=ymin, ymax=np.max(dVoutdVin_fit), color=Vdd_colors[Vdd], linestyle='--', alpha=0.7)
        
        x1, x2 = 0.5,0.65
        axins.set_xlim(x1, x2)
        axins.set_ylim(0, 140)
        axins.set_xticks([0.5, 0.55, 0.6, 0.65])
        axins.set_xticklabels([0.50, 0.55, 0.60, None])
        ax.axvspan(x1, x2, color='gray', alpha=0.15)
        con1 = ConnectionPatch(
            xyA=(x1, ax.get_ylim()[0]), coordsA=ax.transData,
            xyB=(0, 0), coordsB=axins.transAxes,
            color="0.5"
        )

        axins.legend(fontsize=4, loc='upper left', framealpha=0.0, title=r'$V_\mathsf{DD}$', title_fontsize=4, handlelength=0.8)

        con2 = ConnectionPatch(
            xyA=(x2, ax.get_ylim()[0]), coordsA=ax.transData,
            xyB=(1, 0), coordsB=axins.transAxes,
            color="0.5"
        )
        axins.set_title(r'Gain, $\left|\mathrm{d}V_\mathsf{out}/\mathrm{d}V_\mathsf{in}\right|$', fontsize=6, pad=2)

        ax.add_artist(con1)
        ax.add_artist(con2)
        ax.set_xlabel(r'Input Voltage, $V_\mathsf{in}$ [V]', fontsize=8)
        ax.set_ylabel(r'Output Voltage, $V_\mathsf{out}$ [V]', fontsize=8)
        ax.set_xlim(-1.25, 4.25)
        ax.set_ylim(-0.25, 5.75)
        #ax.set_title('Voltage Transfer Characteristic', fontsize=8)
        #ax.legend(fontsize=6, loc='upper right', framealpha=0.0, title=r'$V_\mathsf{DD}$', title_fontsize=6)
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=6, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        # img_inverter = mpimg.imread('figures/inverter.png')
        # imagebox = OffsetImage(img_inverter, zoom=0.05)
        # ab = AnnotationBbox(
        #     imagebox,
        #     (0.8, 0.8),               # position
        #     xycoords='axes fraction',
        #     frameon=False
        # )
        # ax.add_artist(ab)
        # Vdd arrow
        min_Vdd = np.min(Vdd_array)
        max_Vdd = np.max(Vdd_array)
        ax.annotate('', xy=(-0.5, 0.25), xytext=(-0.5, 5.25),
            arrowprops=dict(arrowstyle='<|-', color='black',shrinkA=0, shrinkB=0))
        ax.text(-0.5, 0.25, rf'$V_\mathsf{{DD}}$ = {min_Vdd:.1f} V', fontsize=5, verticalalignment='top', horizontalalignment='center')
        ax.text(-0.5, 5.25, rf'$V_\mathsf{{DD}}$ = {max_Vdd:.1f} V', fontsize=5, verticalalignment='bottom', horizontalalignment='center')
        plt.savefig(script_dir+"/figures/inverter_gain.png", bbox_inches=None, dpi=600)
        plt.close()
        
    ##################### Other plots ######################
    if 0: # Plot hysteresis DeltaVth vs freq comparisons
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['Vd']==0.5) & (df['precondition'] == False)]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        cmap = ['#0047AB', '#4169E1', '#6495ED', '#87CEEB', '#ADD8E6', '#B0E0E6', '#00BFFF', '#1E90FF', '#4A90E2', '#5B9BD5']
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            color = cmap[idx % len(cmap)]
            marker = markers[idx % len(markers)]
            label = f'Device {key[1]}, Meas number {key[2]}'

            if key in df_groups:
                subset = df_groups[key]
                ax.plot(subset['freq'], subset['DeltaVth'],
                        marker=marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=color, label=label)

                ax.plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-', markersize=8,color=color)
        
        ax.axhline(0, linestyle='--', color = 'k')
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        ax.set_xlabel(r'$f$ [Hz]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        # device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        Vd_text = rf"$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]:.1f} V"
        ax.text(0.05, 0.05, Vd_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        #plt.title('Hysteresis Width, $V_h$')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_Vd0.5.pdf", bbox_inches="tight", transparent=True)
        plt.close()

        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['Vd']==0.1) & (df['precondition'] == False)]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        cmap = ['#0B6623', '#2D5016', '#228B22', '#32CD32', '#00FA9A', '#3CB371', '#90EE90', '#98FB98', '#00FF7F', '#7CFC00']
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            color = cmap[idx % len(cmap)]
            marker = markers[idx % len(markers)]
            label = f'Device {key[1]}, Meas number {key[2]}'

            if key in df_groups:
                subset = df_groups[key]
                ax.plot(subset['freq'], subset['DeltaVth'],
                        marker=marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=color, label=label)
                
                ax.plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-',color=color)
        
        ax.axhline(0, linestyle='--', color = 'k')
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        ax.set_xlabel(r'$f$ [Hz]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        # device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        Vd_text = rf"$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]} V"
        ax.text(0.05, 0.05, Vd_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        #plt.title('Hysteresis Width, $V_h$')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_Vd0.1.pdf", bbox_inches="tight", transparent=True)
        plt.close()

        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['Vd']==1) & (df['sample']==1) & (df['precondition'] == False)]

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        cmap = ['#FF0000', '#FF4444', '#FF6666', '#FF8888', '#FFAAAA', '#FFCCCC', '#DD0000', '#CC0000', '#BB0000', '#AA0000']
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            color = cmap[idx % len(cmap)]
            marker = markers[idx % len(markers)]
            label = f'Device {key[1]}'

            if key in df_groups:
                subset = df_groups[key]
                ax.plot(subset['freq'], subset['DeltaVth'],
                        marker=marker, linestyle=' ', markersize=10,
                        markeredgecolor="#13073A", markeredgewidth=2,
                        markerfacecolor=color, label=label)
                
                ax.plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-', markersize=8,color=color)
        
        ax.axhline(0, linestyle='--', color = 'k')
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        ax.set_xlabel(r'$f$ [Hz]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        ax.set_xlim(1e-3,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        Vd_text = rf"$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]} V"
        ax.text(0.05, 0.05, Vd_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        #plt.title('Hysteresis Width, $V_h$')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_Vd1.pdf", bbox_inches="tight", transparent=True)
        plt.close()
    
    if 0: # Plot hysteresis DeltaVth vs freq comparisons
        df = pd.read_csv("data/hyst_hbn-encapsulated_vs_non-encapsulated.csv")
        df = df[(df['Vd']==0.1)]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))

        n_keys = len(all_keys)
        # Map colors to vacuum conditions
        batch_colors = {'TUWien_planar_hbn-encapsulated': '#0047AB', 'TUWien_planar_20nm': '#FF0000'}
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                batch = subset['batch'].iloc[0]
                ax.plot(subset['freq'], subset['DeltaVth'],
                    marker=marker, linestyle=' ', markersize=10,
                    markeredgecolor="#13073A", markeredgewidth=2,
                    markerfacecolor=batch_colors[batch], label=subset['batch_info'].iloc[0])

                ax.plot(subset['freq'], subset['DeltaVth_fit'], linestyle='-', markersize=8,color=batch_colors[batch])
        
        ax.axhline(0, linestyle='--', color = 'k')
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        ax.set_xlabel(r'$f$ [Hz]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        ax.set_xlim(0.5e-1,200)
        #ax.grid(True, which='both', alpha=0.3)
        
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        #plt.title('Hysteresis Width, $V_h$')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_enc_vs_non.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Plot BTI MSM DeltaVth all duts vs Eod,str
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_MSM.csv'))
        df = df[(df['VgRemain'] == 0.0) & (df['tStress']==100)]
        
        fig, ax = plt.subplots(figsize=(3.3, 2.25), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        plt.subplots_adjust(left=0.65/fig_width, right=1 - 0.2/fig_width, top=1 - 0.1/fig_height, bottom=0.4/fig_height)

        VgStress_array = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))
        str_colors = {Vg: colors[i] for i, Vg in enumerate(VgStress_array)}

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        removed_keys = [
            ('TUWien_planar_hbn-encapsulated', '2A1t1', 2),
            ('TUWien_planar_hbn-encapsulated', '1A11t1', 3),
            ('TUWien_planar_hbn-encapsulated', '1A11t1', 6),
            ('TUWien_planar_hbn-encapsulated', '1A13t1', 2),
            ('TUWien_planar_hbn-encapsulated', '1A13t1', 4),
            #('TUWien_planar_hbn-encapsulated', '1A15t1', 4),
            ('TUWien_planar_hbn-encapsulated', '1A15t1', 5),
            ('TUWien_planar_hbn-encapsulated', '1A15t1', 6),
        ]
        all_keys = [k for k in df_groups.keys() if k not in removed_keys]
        n_keys = len(all_keys)
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h']

        # Extract unique DUTs
        unique_duts = sorted(df['dut'].unique())

        # Map each DUT to a marker
        dut_marker_map = {
            dut: markers[i % len(markers)]
            for i, dut in enumerate(unique_duts)
        }

        used_labels = set()
        legend_elements = []
        for i,VgStress in enumerate(VgStress_array[::-1]):
            df_Vstr = df[(df['VgStress']==VgStress)]
            df_groups_Vstr = dict(tuple(df_Vstr.groupby(['batch','dut','sample'])))
            for idx, key in enumerate(all_keys):
                marker = dut_marker_map[key[1]]
                if key in df_groups_Vstr:
                    subset = df_groups_Vstr[key]
                    subset = subset.sort_values(by='tRec')
                    subset_begin = subset[subset['tRec']==0.5]
                    subset_max = subset.loc[[subset['DeltaVth'].idxmax()]]

                    if key not in used_labels:
                        label = rf'$W/L$ = {subset["width"].iloc[0]}/{subset["length"].iloc[0]}; ' \
                        rf'Array {subset["array"].iloc[0]}; Meas {subset["sample"].iloc[0]}; '
                        legend_elements.append(
                            Line2D([0], [0],
                                marker=marker,
                                linestyle='None',
                                markerfacecolor='none',
                                markeredgecolor='#13073A',
                                label=label)
                        )
                        used_labels.add(key)
                    else:
                        label = None
                    

                    ax.plot(subset_max['area'], subset_max['Vth'] - subset_max['Vth_initial'],
                        marker = marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=str_colors[VgStress],label=label)
                    
                    ax.plot(subset_begin['area'], subset_begin['Vth'] - subset_begin['Vth_initial'],
                        marker = marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=str_colors[VgStress],label=label)

                    # ax.plot(subset['tRec'], subset['Vth_fit'] - subset['Vth_initial'], linestyle='-', markersize=8,color=str_colors[VgStress])

        # if i == 0:
        #     ylim = ax.get_ylim()
        # else:
        #     ax.set_ylim(ylim)
        ax.axhline(0, linestyle='--', color = 'k')
        device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V\n $t_{{str}}$ = {df["tStress"].iloc[0]} s\n$V_{{G,rec}}$ = {df["VgRemain"].iloc[0]} V'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        ax.set_xlabel(r'Overdrive electric field, $E_{\mathsf{od,str}}$ [MV/cm]', fontsize=8)
        ax.set_ylabel(r'Threshold Shift, $\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
        ax.legend(handles=legend_elements, fontsize=5, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0)
        plt.savefig(script_dir+f"/figures/MSM_DeltaVth_vs_Eod_duts.pdf", bbox_inches=None)
        plt.close()
    
    if 0: # Measuring Vth vs measuring one point Id
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))

        VgStress_array = [3.0, 4.0, 5.0, 6.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))
        n_keys = len(all_keys)
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h']
        colors = viridis(np.linspace(0, 1, len(all_keys)))
        device_colors = {key: colors[idx % len(colors)] for idx, key in enumerate(all_keys)}
        
        for i,VgStress in enumerate(VgStress_array[::-1]):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax2 = ax.twinx()
            df_Vstr = df[(df['VgStress']==VgStress)]
            df_groups_Vstr = dict(tuple(df_Vstr.groupby(['batch','dut','sample'])))
            for idx, key in enumerate(all_keys):
                marker = markers[idx % len(markers)]
                if key in df_groups_Vstr:
                    subset = df_groups_Vstr[key]
                    if subset['Vth'].notna().any():
                        ax.plot(subset['tStress'], subset['Vth'],
                            marker=marker, linestyle=' ', markersize=10,
                            markeredgecolor="#13073A", markeredgewidth=2,
                            markerfacecolor='b',label=rf"$W/L$ = {subset['width'].iloc[0]}/{subset['length'].iloc[0]}; Array {subset['array'].iloc[0]}; Meas {subset['sample'].iloc[0]}; $E_{{od,str}}$ = {subset['Eod_str'].iloc[0]:.2f} MV/cm")
                    else:
                        ax2.plot(subset['tStress'], subset['I'],
                            marker=marker, linestyle=' ', markersize=10,
                            markeredgecolor="#13073A", markeredgewidth=2,
                            markerfacecolor='r',label=rf"$W/L$ = {subset['width'].iloc[0]}/{subset['length'].iloc[0]}; Array {subset['array'].iloc[0]}; Meas {subset['sample'].iloc[0]}; $E_{{od,str}}$ = {subset['Eod_str'].iloc[0]:.2f} MV/cm")
                        
            ax2.set_yscale('log')
            ax2.set_ylabel(r'$I_D$ [A]', fontsize=textSize, color='r')

            device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V'
            ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
                fontsize=22, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
            ax.set_xlabel(r'$t_{\mathsf{str}}$ [s]', fontsize=textSize)
            ax.set_ylabel(r'$\Delta V_{\mathsf{th}}$ [V]', fontsize=textSize)
            ax.set_xscale('log')
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            all_handles = handles1 + handles2
            all_labels = labels1 + labels2
            ax.legend(all_handles, all_labels, fontsize=textSizeLegend, loc='upper left', bbox_to_anchor=(0.85, 1), framealpha=0.9)
            plt.savefig(script_dir+f"/figures/OTF_Vth_vs_I_duts_{VgStress}.pdf", bbox_inches="tight", transparent=True)
            plt.close()
    
    if 0: # One point Id BTI MSM
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_MSM.csv'))
        df = df[(df['VgRemain'] == 0.0) & (df['tStress']==100) & ~(df['tRec']== 0.5)]
        #df = df[(df['dut'] == '1A15t1') & (df['temp'] == '300K') & (df['sample'] == 6)]

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))
        n_keys = len(all_keys)
        cycles = sorted(df['cycle'].unique())
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h']
        #Cycle 
        cycle_colors = plasma(np.linspace(0.1, 0.9, len(cycles)))
        cycle_color_map = {cycle: cycle_colors[i] for i, cycle in enumerate(cycles)}
        #VgStress
        VgStress_array = sorted(df['VgStress'].unique())
        VgStress_colors = viridis(np.linspace(0.1, 0.9, len(VgStress_array)))
        VgStress_color_map = {Vg: VgStress_colors[i] for i, Vg in enumerate(VgStress_array)}
        
        for i, Vgstr in enumerate(VgStress_array[::-1]):
            # df_cycle = df[(df['cycle']==cycle)]
            # df_groups_cycle = dict(tuple(df_cycle.groupby(['batch','dut','sample'])))
            # cycle_color = cycle_color_map[cycle]
            fig, ax = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True)
            ax2 = ax.twinx()
            df_Vstr = df[(df['VgStress']==Vgstr)]
            df_groups_cycle = dict(tuple(df_Vstr.groupby(['batch','dut','sample'])))
            VgStress_color = VgStress_color_map[Vgstr]
            for idx, key in enumerate(all_keys):
                marker = markers[idx % len(markers)]
                if key in df_groups_cycle:
                    subset = df_groups_cycle[key]
                    if subset['Vth'].notna().any():
                        ax.plot(subset['tRec'], subset['Vth'],
                            marker=marker, linestyle=' ', markersize=10,
                            markeredgecolor="#13073A", markeredgewidth=2,
                            markerfacecolor='b',label=rf"$W/L$ = {subset['width'].iloc[0]}/{subset['length'].iloc[0]}; Array {subset['array'].iloc[0]}; Meas {subset['sample'].iloc[0]}; $E_{{od,str}}$ = {subset['Eod_str'].iloc[0]:.2f} MV/cm")
                    else:
                        ax2.plot(subset['tRec'], subset['I'],
                            marker=marker, linestyle=' ', markersize=10,
                            markeredgecolor="#13073A", markeredgewidth=2,
                            markerfacecolor='r',label=rf"$W/L$ = {subset['width'].iloc[0]}/{subset['length'].iloc[0]}; Array {subset['array'].iloc[0]}; Meas {subset['sample'].iloc[0]}; $E_{{od,str}}$ = {subset['Eod_str'].iloc[0]:.2f} MV/cm")
                        
            ax2.set_yscale('log')
            #ax2.set_ylim(5e-9, 1e-8)
            ax2.set_ylabel(r'$I_D$ [A]', fontsize=8, color='r')

            device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V'
            ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
                fontsize=6, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
            ax.set_xlabel(r'$t_{\mathsf{rec}}$ [s]', fontsize=8)
            ax.set_ylabel(r'$\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
            ax.set_xscale('log')
            # ax.legend(all_handles, all_labels, fontsize=6, loc='upper left', bbox_to_anchor=(0.85, 1), framealpha=0.9)
            plt.savefig(script_dir+f"/figures/MSM_Vth_vs_I_duts_VgStress{Vgstr}.pdf", bbox_inches="tight", transparent=True)
            plt.close()
    
    sys.exit()
