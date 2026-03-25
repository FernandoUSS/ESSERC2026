import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import plasma, viridis
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.colors import to_rgb, LogNorm
import matplotlib.colors as mcolors
import numpy as np
import os
import sys
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
#from mpl_toolkits.axes_grid1.inset_locator import indicate_inset_zoom
import json5

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
    
    data_folder = os.path.join('/','home','delossa','Nextcloud','DataProcessingLab','data')

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

        fig, ax = plt.subplots(figsize=(3.3, 2.5), constrained_layout=False)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
        
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
        ax.annotate('', xy=(arrow_x, arrow_y_min), xytext=(arrow_x, arrow_y_max),
                arrowprops=dict(arrowstyle='<|-', color='black'))
        ax.text(arrow_x, arrow_y_min, 
               rf'$V_\mathsf{{D}}$ = {np.min(vds_values):.1f} V',
                fontsize=7,
               verticalalignment='top', ha='center')
        ax.text(arrow_x, arrow_y_max-1.5e-1, 
               rf'$V_\mathsf{{D}}$ = {np.max(vds_values):.1f} V',
               fontsize=7,
               verticalalignment='bottom', ha='center')
        
        ax.text(4,1e-8,r'$I_\mathsf{G}$')
        # Set axis labels with units
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]')
        ax.set_ylabel(r'$I_\mathsf{D}/W$ [$\mu$A/$\mu$m]')
        
        # Add device info text
        device_text = rf'$T$ = {df["temp"].iloc[0].replace("K", " K")}' + '\n' + rf'$W/L$ = {df['width'].iloc[0]:.0f}/{df['length'].iloc[0]:.0f}'
        ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
                verticalalignment='top')
               #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Id_SS = 10**(1/SS*(vg - Vzero_current))
        # ax.plot(vg,Id_SS/width*1e6,'--', color='k',alpha=0.5)

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        
        # Create text with the metrics (Ion/Ioff,SS,Imax)
        metrics_text = rf'$I_\mathsf{{on}}/I_\mathsf{{off}}$ $>$ $10^{int(np.floor(np.log10(IonIoff)))}$; ' + rf'$SS$ = {SS:.0f} mV/dec' + '\n' + rf'$I_{{max}}$ = {Imax/width*1e6:.2f} $\mu$A/$\mu$m; ' + rf'$I_\mathsf{{G}}/A$ $<$ $10^{{{int(np.ceil(np.log10(Igate_A)))}}}$ A/cm$^2$'
        ax.text(0.75, 0.05, metrics_text, transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=6,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.77, 0.07, rf'@ V$_\mathsf{{D}}$ = {vds} V', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontsize=6)

        plt.savefig(os.path.join(inputdir,'figures','IdVg_encapsulated_1.pdf'), bbox_inches=None)
        plt.close()
    
    if 0: # IdVg curves for non-encapsulated devices
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-non-encapsulated',
                                      "IdVg_TUWien_planar_20nm.csv"))
        df = df[(df['dut'] == 'M11') & (df['temp'] == '380K') & (df['sample'] == 1)]
        SS = df[(df['Vd'] == 0.5)]['SS'].iloc[0]
        Vzero_current = df[(df['Vd'] == 0.5)]['Vzero_current'].iloc[0]
        IonIoff = df[(df['Vd'] == 0.5)]['Ion/Ioff'].iloc[0]
        Imax = df[(df['Vd'] == 0.5)]['Imax'].iloc[0]

        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot IdVg curves for different Vds values
        vds_values = df['Vd'].unique()
        #colors = viridis(np.linspace(0, 1, len(vds_values)))
        
        for idx, vds in enumerate(sorted(vds_values)):
            df_vds = df[df['Vd'] == vds]
            vg = df_vds['Vg'].values[0]
            id_vals = df_vds['Id'].values[0]
            width = df_vds['width'].values[0]
            ax.plot(vg, id_vals/width*1e6, '-', linewidth=2.5, 
                   markersize=6, label=rf'$V_{{D}}$ = {vds} V', color=Vd_color(vds))
            
        ax.set_yscale("log")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Add legend with smaller font size
        ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-2]], 
             [rf'$V_{{D}}$ = {sorted(vds_values)[0]} V', rf'$V_{{D}}$ = {sorted(vds_values)[-1]} V'],
             fontsize=22, loc='lower right', framealpha=0.0)
        
        # Set axis labels with units
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=textSize)
        
        # Add device info text
        device_text = rf'Device {df["dut"].iloc[0]}' + '\n' + rf'$T$ = {df["temp"].iloc[0]}'
        ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='top')
               #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.savefig(script_dir+"/figures/IdVg_non-encapsulated_0.pdf", bbox_inches="tight", transparent=True)

        Id_SS = 10**(1/SS*(vg - Vzero_current))
        ax.plot(vg,Id_SS/width*1e6,'--')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Set axis labels with units
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=textSize)
        
        # Create text with the metrics (Ion/Ioff,SS,Imax)
        metrics_text = rf'$I_\mathsf{{on}}/I_\mathsf{{off}}$ $\approx$ $10^{int(np.log10(IonIoff))}$' + '\n' f'$SS$ = {SS*1000:.0f} mV/dec\n$I_{{max}}$ = {Imax/width*1e6:.2f} µA/µm'
        ax.text(0.75, 0.45, metrics_text, transform=ax.transAxes, 
               fontsize=20, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.savefig(script_dir+"/figures/IdVg_non-encapsulated_1.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # IdVg curves for different devices with W/L normalization
        ########################################################################################
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

    if 0: # Vth vs area scatter plot and SS vs area scatter plot for different devices
        ########################################################################################
        df = pd.read_csv("data/IdVg_Vth_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['Vd'] == 1.0) & (df['sample']==1)]
        
        # Get unique devices and assign symbols
        devices = df['dut'].unique()
        arrays = df['array'].unique()
        symbols = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x']
        colors = ["#887CAF", "#AA3939"]
        array_colors = {array: colors[idx % len(colors)] for idx, array in enumerate(sorted(arrays))}

        fig, ax = plt.subplots(figsize=(10, 10))

        for idx, device in enumerate(sorted(devices)):
            df_device = df[df['dut'] == device]
            width = df_device['width'].values
            area = df_device['area'].values
            vth = df_device['Vth'].values
            array = df_device['array'].iloc[0]

            ax.plot(area, vth, 
               marker='v',
               linestyle=' ',
               markersize=16,
               markeredgecolor="#13073A",
               markeredgewidth=2,
               markerfacecolor=array_colors[array],
               label=f'{device}')
        
        # Set axis labels
        #ax.set_xlabel(r'Width [$\mu$m]', fontsize=textSize)
        ax.set_xlabel(r'Area [$\mu \mathsf{m}^2$]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{th}$ [V]', fontsize=textSize)
        
        # Add more ticks
        ax.tick_params(axis='both', which='major', labelsize=textSizeLegend)
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add legend
        legend_elements = [Line2D([0], [0], marker='v', color='w', 
                     markerfacecolor=array_colors[array],
                     markeredgecolor="#13073A", markeredgewidth=2,
                     markersize=12, label=f'Array: {array}')
                  for array in sorted(arrays)]
        ax.legend(handles=legend_elements, fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        plt.savefig(script_dir+"/figures/Vth_vs_area.pdf", bbox_inches="tight", transparent=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))

        for idx, device in enumerate(sorted(devices)):
            df_device = df[df['dut'] == device]
            width = df_device['width'].values
            area = np.array(df_device['area'].values[0])
            SS = np.array(df_device['SS'].values[0])*1e3
            array = df_device['array'].iloc[0]

            ax.plot(area, SS, 
               marker='v',
               linestyle=' ',
               markersize=16,
               markeredgecolor="#13073A",
               markeredgewidth=2,
               markerfacecolor=array_colors[array],
               label=f'{device}')
        
        # Set axis labels
        #ax.set_xlabel(r'Width [$\mu$m]', fontsize=textSize)
        ax.set_xlabel(r'Area [$\mu \mathsf{m}^2$]', fontsize=textSize)
        ax.set_ylabel(r'$SS$ [mV/dec]', fontsize=textSize)
        
        # Add more ticks
        ax.tick_params(axis='both', which='major', labelsize=textSizeLegend)
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add legend
        legend_elements = [Line2D([0], [0], marker='v', color='w', 
                     markerfacecolor=array_colors[array],
                     markeredgecolor="#13073A", markeredgewidth=2,
                     markersize=12, label=f'Array: {array}')
                  for array in sorted(arrays)]
        ax.legend(handles=legend_elements, fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        plt.savefig(script_dir+"/figures/SS_vs_area.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # IdVgs Hysteresis
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4) & (df['nom_freq'] == 0.001) & (df['precondition'] == False)]
        for c in ['Id','Vg']:
            if c in df.columns:
                df[c] = df[c].map(json5.loads)

        width = df['width'].iloc[0]
        Vmax = df['Vmax'].iloc[0]
        Vmin = df['Vmin'].iloc[0]

        fig, ax = plt.subplots(figsize=(2.2, 2.5),constrained_layout=True)

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
        device_text = r'$T$ = 300 K' +'\n' + rf'$W/L$ = $\frac{{{df['width'].iloc[0]:.0f}\,\mu m}}{{{df['length'].iloc[0]:.0f}\,\mu m}}$'+ '\n' + rf'$V_\mathsf{{D}}$ = {df["Vd"].iloc[0]} V' + '\n' + rf'$r_\mathsf{{sweep}}$ = {df["nom_freq"].iloc[0]*(Vmax - Vmin):.2f} V/s'
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
        axins.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.5,zorder=0)
        current_max = np.max(np.array(Id)/width*1e6)
        current_decade = current_max / 10
        vg_max = Vg[np.argmax(np.array(Id)/width*1e6)]
        axins.annotate('', xy=(df['Vth_down'].iloc[0], df['Ith'].iloc[0]/width*1e6), xytext=(df['Vth_up'].iloc[0], df['Ith'].iloc[0]/width*1e6),
               arrowprops=dict(arrowstyle='<|-|>', color='black'))
        axins.text(0.5*(df['Vth_down'].iloc[0] + df['Vth_up'].iloc[0]), df['Ith'].iloc[0]/width*1e6, r'$V_\mathsf{H}$', fontsize=6, va='bottom',ha='center')
        axins.scatter([Vth_up], [df['Ith'].iloc[0]/width*1e6], color=color_up, marker='+')
        axins.scatter([Vth_down], [df['Ith'].iloc[0]/width*1e6], color=color_down, marker='+')
        axins.text(df['Vth_up'].iloc[0], df['Ith'].iloc[0]/width*1e6, r'$V_\mathsf{th,up}$', fontsize=6, va='bottom', ha='right', color=color_up)
        axins.text(df['Vth_down'].iloc[0]-0.02, df['Ith'].iloc[0]/width*0.97e6, r'$V_\mathsf{th,down}$', fontsize=6, va='top', ha='left', color=color_down)
        plt.savefig(script_dir+"/figures/hysteresis_IdVg_example.pdf", bbox_inches=None)
        plt.close()

    if 0: # Id vs t Hysteresis
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4) & (df['precondition'] == False)]
        for c in ['Id','Vg','time']:
            if c in df.columns:
                df[c] = df[c].map(json5.loads)
                
        # width = df['width'].iloc[0]

        fig, ax = plt.subplots(figsize=(2.3, 2.5), constrained_layout=True)

        # Get unique sweep frequencies
        freqs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        # freqs = np.unique(df['nom_freq'])
        df['nom_freq'] = np.around(df['nom_freq'], decimals=4)
        colors = plasma(np.linspace(0, 1, len(freqs)))
        freq_to_color = {freq: colors[idx] for idx, freq in enumerate(sorted(freqs))}
        
        # Add rectangles to indicate sweep direction
        ax.add_patch(Rectangle((0, 8e-7), 0.5, 1e-0-8e-7, facecolor=color_up, alpha=0.1))
        ax.add_patch(Rectangle((0.5, 8e-7), 0.5, 1e-0-8e-7, facecolor=color_down, alpha=0.1))
        ax.text(0.25, 0.975, r'Up sweep', fontsize=6, va='top', ha='center', color=color_up, transform=ax.transAxes)
        ax.text(0.75, 0.975, r'Down sweep', fontsize=6, va='top', ha='center', color=color_down, transform=ax.transAxes)

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
            ax.plot((time-np.min(time))/sweep_time, Id/width*1e6, '-', color =freq_to_color[freq], label=f'f = {freq} Hz')
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

        # Add device info
        device_text = r'$T$ = 300 K' +'\n' + rf'$W/L$ = $\frac{{{df['width'].iloc[0]:.0f}\,\mu m}}{{{df['length'].iloc[0]:.0f}\,\mu m}}$'+ '\n' + rf'$V_\mathsf{{D}}$ = {df["Vd"].iloc[0]} V'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #        fontsize=6, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        # Legend with min and max frequencies
        # ax.legend(fontsize=6, loc='lower right', framealpha=0.9,
        #     handlelength=1.5)
        #ax.set_title(r'Hysteresis $I_\mathsf{D}-V_\mathsf{G}$ curve')
        # ax.text(0.2, df['Ith'].iloc[0]/width*1e6*1.1, r'$V_\mathsf{th,up}$', fontsize=5, va='bottom', ha='center')
        # ax.text(0.835, df['Ith'].iloc[0]/width*1e6*1.1, r'$V_\mathsf{th,down}$', fontsize=5, va='bottom', ha='center')
        ax.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.5)
        axins.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k', alpha=0.5)
        ax2.axhline(Vmax, linestyle='--',xmin=0.5, xmax=1, color = 'k', alpha=0.5)
        current_max = np.max(np.array(Id)/width*1e6)
        current_decade = current_max / 10
        vg_max = Vg[np.argmax(np.array(Id)/width*1e6)]
        axins.annotate('', xy=(0.715, 4.5e-3), xytext=(0.79, 4.5e-3), arrowprops=dict(arrowstyle='-|>', color='black'))
        # axins.text(0.755, 4.5e-3, r'$r_\mathsf{sweep}$', fontsize=7, va='bottom',ha='center')
        # Text sweep rate limits
        axins.text(0.72, 4.5e-3, r'$r_\mathsf{sweep}^{\min}$', fontsize=5, va='center', ha='right')
        axins.text(0.79, 4.5e-3, r'$r_\mathsf{sweep}^{\max}$', fontsize=5, va='center', ha='left')
        # Axis 2 texts
        # ax2.text(0.1, 3.5, r'$V_\mathsf{G}$- up sweep', fontsize=6, va='top', ha='left', color=color_up,rotation=30)
        # ax2.text(0.9, 3.5, r'$V_\mathsf{G}$- down sweep', fontsize=6, va='top', ha='right', color=color_down,rotation=-30)
        ax2.text(0.35, 4.25, r'$V_\mathsf{G}$', fontsize=6, va='bottom', ha='right')
        ax.text(0.35, 2e-2, r'$\log(I_\mathsf{D})$', fontsize=6, va='bottom', ha='right')
        # ax2.text(0.5, 6, r'$V_\mathsf{max}$', fontsize=6, va='bottom', ha='center')
        # ax2.text(1, 0, r'$V_\mathsf{min}$', fontsize=6, va='bottom', ha='center')
        plt.savefig(script_dir+"/figures/hysteresis_time_example.pdf", bbox_inches=None)
        plt.close()

    if 0: # IdVgs Hysteresis
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Get unique sweep frequencies
        freqs = df['nom_freq'].unique()
        colors = plasma(np.linspace(0, 1, len(freqs)))
        freq_to_color = {freq: colors[idx] for idx, freq in enumerate(sorted(freqs))}

        width = 39 #um

        for freq in sorted(freqs):
            df_freq = df[df['nom_freq'] == freq]
            vg = df_freq['Vg'].values[0]
            id_vals = df_freq['Id'].values[0]
            ax.semilogy(vg, np.array(id_vals)/width*1e6, '-', linewidth=2.5, 
                   label=f'f = {freq} Hz', color=freq_to_color[freq])
            # axins.plot(vg, np.array(id_vals)/width*1e6, '-', linewidth=2.5, 
            #         color=freq_to_color[freq])


        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=textSize)
        # axins.set_yscale("log")
        # axins.set_xticks([])
        # axins.set_yticks([])
        # Define zoom region
        # x1, x2 = 2, 4
        # y1, y2 = 1e-3, 1e-2
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        # Draw connecting lines
        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # Add device info
        device_text = f'Device {df["dut"].iloc[0]}' + '\n' + r'$T$ = 300 K' +'\n' + rf'$V_\mathsf{{D}}$ = {df["Vd"].iloc[0]} V'
        ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        # Legend with min and max frequencies
        sorted_freqs = sorted(freqs)
        ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
             [rf'$f$ = {sorted_freqs[0]} Hz', rf'$f$ = {sorted_freqs[-1]} Hz'],
             fontsize=textSizeLegend, loc='lower right', framealpha=0.9,
             handlelength=1.5)

        ax.set_title(r'$I_\mathsf{D}-V_\mathsf{G}$ curves during hysteresis')
        plt.savefig(script_dir+"/figures/hysteresis_IdVg_0.pdf", bbox_inches="tight")
        ax.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k')
        #axins.axhline(df['current_level'].iloc[0]/width*1e6, linestyle='--', color = 'k')
        # Add vertical double arrow for one decade
        current_max = np.max(np.array(id_vals)/width*1e6)
        current_decade = current_max / 10
        vg_max = vg[np.argmax(np.array(id_vals)/width*1e6)]
        ax.annotate('', xy=(vg_max, current_max), xytext=(vg_max, current_decade),
               arrowprops=dict(arrowstyle='<|-|>', color='black'))
        ax.text(vg_max, (current_decade), '1-decade criterion', fontsize=textSizeLegend, va='top',ha='right')
        plt.savefig(script_dir+"/figures/hysteresis_IdVg_1.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Precondition IdVgs
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '2A9t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['precondition'] == True)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)
        
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get unique sweep frequencies
        sweep_indexes = df['sweep_index'].unique()[:50]
        colors = plasma(np.linspace(0, 1, len(sweep_indexes)))
        sweep_index_to_color = {freq: colors[idx] for idx, freq in enumerate(sorted(sweep_indexes))}

        width = 27 #um

        for sweep_index in sorted(sweep_indexes):
            df_sweep_index = df[df['sweep_index'] == sweep_index]
            vg = df_sweep_index['Vg'].values[0]
            id_vals = df_sweep_index['Id'].values[0]
            ax.semilogy(vg, np.array(id_vals)/width*1e6, '-', linewidth=2.5, color=sweep_index_to_color[sweep_index])


        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=textSize)
        device_text = f'Device {df["dut"].iloc[0]}' + '\n' + r'$T$ = 300 K' +'\n' + rf'$V_D$ = {df["Vd"].iloc[0]} V' +'\n'+ rf'$f$ = {df["nom_freq"].iloc[0]} Hz'
        ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))

        # Legend with min and max frequencies
        sorted_freqs = sorted(freqs)
        ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
             [f'cycle {sweep_indexes[0]}', f'cycle {sweep_indexes[-1]}'],
             fontsize=textSizeLegend, loc='lower right', framealpha=0.9,
             handlelength=1.5)

        ax.set_title(r'$I_\mathsf{D}-V_\mathsf{G}$ curves during precondition')
        plt.savefig(script_dir+"/figures/precondition_IdVg_0.pdf", bbox_inches="tight", transparent=True)
        current_max = np.max(np.array(id_vals)/width*1e6)
        current_decade = current_max / 10
        ax.axhline(current_decade, linestyle='--', color = 'k')
        # vg_max = vg[np.argmax(np.array(id_vals)/width*1e6)]
        # ax.annotate('', xy=(vg_max, current_max), xytext=(vg_max, current_decade),
        #        arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        # ax.text(vg_max, (current_decade), '1-decade criterion', fontsize=textSizeLegend, va='top',ha='right')
        plt.savefig(script_dir+"/figures/precondition_IdVg.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Precondition Vth and DeltaVth
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '2A9t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['precondition'] == True)]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot Vth_up and Vth_down vs nom_freq
        ax.plot(df['sweep_index'], df['Vth_up'], '^', markersize=10, 
                markeredgecolor="#13073A", markeredgewidth=2,
                markerfacecolor="#AA3939", label=r'$V_{th,up}$')
        ax.plot(df['sweep_index'], df['Vth_down'], 'v', markersize=10,
                markeredgecolor="#13073A", markeredgewidth=2,
                markerfacecolor="#887CAF", label=r'$V_{th,down}$')
        
        ax.set_xlabel(r'Number of precondition cycles', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{th}$ [V]', fontsize=textSize)
        ax.set_xlim(0,50)

        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #         fontsize=22, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        plt.title(r'$V_\mathsf{th}$ during precondition')
        plt.savefig(script_dir+"/figures/hysteresis_precondition_Vth.pdf", bbox_inches="tight", transparent=True)
        plt.close()

        ## Precondition DeltaVth
        ###############################################################################################
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '2A9t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['precondition'] == True)]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot Vth_up and Vth_down vs nom_freq
                # Plot DeltaVth vs frequency
        ax.plot(df['sweep_index'], df['DeltaVth'], 'o', markersize=10,
            markeredgecolor="#13073A", markeredgewidth=2,
            markerfacecolor="#2E8B57", label=r'$\Delta V_{th}$')
        
        ax.set_xlabel(r'Number of precondition cycles', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=textSize)
        ax.set_xlim(0,50)

        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #         fontsize=22, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.axhline(0, linestyle='--', color = 'k')
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        plt.title(r'Hysteresis Width, $V_\mathsf{H}$')
        plt.savefig(script_dir+"/figures/hysteresis_precondition_DeltaVth.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Hysteresis Vth and DeltaVth vs frequency
        ###############################################################################################
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4)]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot Vth_up and Vth_down vs nom_freq
        ax.plot(df['freq'], df['Vth_up'], '^', markersize=10, 
                markeredgecolor="#13073A", markeredgewidth=2,
                markerfacecolor="#AA3939", label=r'$V_{th,\text{up}}$')
        ax.plot(df['freq'], df['Vth_down'], 'v', markersize=10,
                markeredgecolor="#13073A", markeredgewidth=2,
                markerfacecolor="#887CAF", label=r'$V_{th,\text{down}}$')
        
        # Plot fits
        ax.plot(df['freq'], df['Vth_up_fit'], '-', linewidth=2.5,
                color="#AA3939", alpha=0.7)
        ax.plot(df['freq'], df['Vth_down_fit'], '-', linewidth=2.5,
                color="#887CAF", alpha=0.7)
        
        ax.set_xlabel(r'$f$ [Hz]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{th}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        #ax.grid(True, which='both', alpha=0.3)
        
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #         fontsize=22, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        plt.title(r'$V_\mathsf{th}$ during hysteresis')
        plt.savefig(script_dir+"/figures/hysteresis_Vth_vs_freq.pdf", bbox_inches="tight", transparent=True)
        plt.close()
        
        # Plot DeltaVth vs frequency
        df = pd.read_csv("data/hyst_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4)]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs frequency
        ax.plot(df['freq'], df['DeltaVth'], 'o', markersize=10,
            markeredgecolor="#13073A", markeredgewidth=2,
            markerfacecolor="#2E8B57", label=r'$\Delta V_{th}$')
        ax.plot(df['freq'], df['DeltaVth_fit'], '-', linewidth=2.5,
            color="#2E8B57", alpha=0.7)
        
        ax.set_xlabel(r'$f$ [Hz]', fontsize=textSize)
        ax.set_ylabel(r'$V_\mathsf{H}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        #ax.grid(True, which='both', alpha=0.3)
        ax.axhline(df['Ith'].iloc[0]/width*1e6, linestyle='--', color = 'k')

        #device_text = f'Device {df["dut"].iloc[0]}\n$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        #     fontsize=22, verticalalignment='top',
        #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        plt.title(r'Hysteresis Width, $V_\mathsf{H}$')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Hysteresis Vth and DeltaVth vs sweep rate (one figure version)
        ###############################################################################################
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS.csv'))
        df = df[(df['dut'] == '1A13t1') & (df['temp'] == '300K') & (df['sample'] == 4)]

        width = df['width'].iloc[0]
        Vmax = df['Vmax'].iloc[0]
        Vmin = df['Vmin'].iloc[0]

        # Create figure with two stacked axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.2, 2.5), sharex=True, constrained_layout=True)

        # =========================
        # Top plot: Vth
        # =========================
        ax1.plot(df['freq']*(Vmax-Vmin), df['Vth_up'], '^',
                markeredgecolor="#13073A",
                markerfacecolor=color_up, label=r'$V_{th,\mathrm{up}}$')

        ax1.plot(df['freq']*(Vmax-Vmin), df['Vth_down'], 'v',
                markeredgecolor="#13073A",
                markerfacecolor=color_down, label=r'$V_{th,\mathrm{down}}$')

        ax1.plot(df['freq']*(Vmax-Vmin), df['Vth_up_fit'], '-',
                color=color_up, alpha=0.7)

        ax1.plot(df['freq']*(Vmax-Vmin), df['Vth_down_fit'], '-',
                color=color_down, alpha=0.7)

        ax1.set_ylabel(r'$V_\mathsf{th}$ [V]', fontsize=8)
        ax1.set_xscale('log')
        ax1.legend(fontsize=6, loc='best')

        # Remove x labels on top plot
        ax1.tick_params(labelbottom=False)

        # =========================
        # Bottom plot: Delta Vth
        # =========================
        ax2.plot(df['freq']*(Vmax-Vmin), df['DeltaVth'], 'o',
                markeredgecolor="#13073A",
                markerfacecolor="#2E8B57", label=r'$\Delta V_{th}$')

        ax2.plot(df['freq']*(Vmax-Vmin), df['DeltaVth_fit'], '-',
                color="#2E8B57", alpha=0.7)

        ax2.axhline(df['Ith'].iloc[0] / width * 1e6,
                    linestyle='--', color='k')

        ax2.set_ylabel(r'Hysteresis Width, $V_\mathsf{H}$ [V]', fontsize=7.5)
        ax2.set_xlabel(r'Sweep Rate, $r_\mathsf{sw}$ [V/s]', fontsize=8)
        ax2.set_xscale('log')

        # =========================
        # Layout adjustments
        # =========================
        plt.subplots_adjust(hspace=0.08)
        fig.align_ylabels()

        # Save figure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(script_dir, "figures", "hysteresis_Vth_DeltaVth_vs_freq.pdf"), bbox_inches="tight")

        plt.close()

    if 0: # Plot BTI DeltaVth vs total time
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_all.csv'))
        # df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(safe_json_load)
        width = df['width'].iloc[0]

        groups = df.groupby(['batch', 'dut', 'sample','meas_type'])
        selected_keys = [
            ('TUWien_planar_hbn-encapsulated', '2A13t1', 1, 'OTF'),
            ('TUWien_planar_hbn-encapsulated', '2A1t1', 1, 'MSM')
        ]
        markers = ['o', '^']
        df_list = []

        for k in selected_keys:
            g = groups.get_group(k).copy()

            if k == ('TUWien_planar_hbn-encapsulated', '2A1t1', 1, 'MSM'):
                g = g[~g['cycle'].isin([5])]

            df_list.append(g)

        df_selected = pd.concat(df_list)

        max_cycles = df_selected['cycle'].max()

        fig, ax = plt.subplots(1,max_cycles + 1, figsize=(4.3, 2.5), sharey=True, constrained_layout=False)
        plt.subplots_adjust(wspace=0.00, bottom=0.2, top=0.90, left=0.12, right=0.98)

        for key in selected_keys:
            df_dut = df_selected[(df_selected['batch'] == key[0]) & (df_selected['dut'] == key[1]) & (df_selected['sample'] == key[2]) & (df_selected['meas_type'] == key[3])]
            marker = markers[selected_keys.index(key)]
            cycles = sorted(df_dut['cycle'].unique())
            for cycle in cycles:

                df_cycle = df_dut[df_dut['cycle'] == cycle]
                df_initial = df_cycle[df_cycle['initial'] == True]

                df_cycle_stress = df_cycle[
                    (df_cycle['initial'] != True) &
                    (df_cycle['extra'] != True) &
                    (df_cycle['end'] != True)
                ].copy()
                
                meas_type = df_cycle['meas_type'].iloc[0]
                if meas_type == 'OTF':
                    i = cycle
                    tvar = 'tStress'
                    df_cycle_stress = df_cycle_stress.sort_values(by=tvar)
                    t = df_cycle_stress[tvar].values
                    Vth = df_cycle_stress['Vth'].values
                    Vth_initial = df_initial['Vth'].values[0]

                elif meas_type == 'MSM':
                    i = cycle*2
                    tvar = 'tRec'
                    df_cycle_stress = df_cycle_stress.sort_values(by=tvar)
                    t = df_cycle_stress[tvar].values
                    Vth = df_cycle_stress['Vth'].values + 1 # shift up for better visibility
                    Vth_initial = df_initial['Vth'].values[0] + 1

                if cycle == 0:


                    df_cycle_stress = df_cycle_stress.sort_values(by=tvar)

                    ax[i].scatter(t, Vth, c=t, cmap=cmap_precondition, norm=norm, marker=marker, edgecolors='#13073A', linewidths=0.8)

                    if meas_type == 'OTF':
                        ax[i].text(t[0], np.max(Vth)+0.1, f'{df_cycle["meas_type"].iloc[0]} meas', fontsize=6, verticalalignment='bottom',horizontalalignment='left')
                    else:
                        ax[i].text(t[0], np.min(Vth)-0.15, f'{df_cycle["meas_type"].iloc[0]} meas', fontsize=6, verticalalignment='top',horizontalalignment='left')

                        ax[i].axhline(Vth_initial, linestyle='--', color=color_precondition, alpha=0.7)

                else:

                    ax[i].scatter(t, Vth, c=t, cmap=cmap_stress if i % 2 == 1 else cmap_relax, norm=norm, marker=marker,edgecolors='#13073A', linewidths=0.8)

                    if meas_type == 'MSM':
                        ax[i].axhline(Vth_initial, linestyle='--', color=color_stress if i % 2 == 1 else color_relax, alpha=0.7)

        stress_ind = 1
        relax_ind = 1
        for i in range(max_cycles + 1):
            if i == 0:
                ax[i].text(0.5, 0.95, f'Pre-conditioning', transform=ax[i].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='center', rotation=90)

                # --- background rectangle ---
                ax[i].axvspan(t.min(), t.max(), color=color_precondition, alpha=0.05)
                ax[i].annotate(
                        '',
                        xy=(1, -0.05),
                        xytext=(1e4, -0.05),
                        xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<|-|>', color='k', linewidth=1, shrinkA=0, shrinkB=0),
                    )
                ax[i].text(
                    1e2,
                    -0.06,
                    rf'$t_\mathsf{{precond}}$',
                    transform=ax[i].get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=7
                )
            else:
                if i % 2 == 1: # Stress
                    ax[i].text(0.5, 0.95, f'Stress \n #{stress_ind}', transform=ax[i].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='center')
                    ax[i].annotate(
                        '',
                        xy=(1, -0.1),
                        xytext=(1e4, -0.1),
                        xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<|-|>', color='k', linewidth=1, shrinkA=0, shrinkB=0),
                    )
                    ax[i].text(
                        1e2,
                        -0.11,
                        rf'$t_\mathsf{{str,{stress_ind}}}$',
                        transform=ax[i].get_xaxis_transform(),
                        ha='center',
                        va='top',
                        fontsize=7
                    )
                    stress_ind += 1
                else: # Relax
                    ax[i].text(0.5, 0.9, f'Relax \n #{relax_ind}', transform=ax[i].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='center')
                    ax[i].annotate(
                        '',
                        xy=(1, -0.05),
                        xytext=(1e4, -0.05),
                        xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='<|-|>', color='k', linewidth=1, shrinkA=0, shrinkB=0),
                    )
                    ax[i].text(
                        1e2,
                        -0.06,
                        rf'$t_\mathsf{{relax,{relax_ind}}}$',
                        transform=ax[i].get_xaxis_transform(),
                        ha='center',
                        va='top',
                        fontsize=7
                    )
                    relax_ind += 1


                # --- background rectangle ---
                ax[i].axvspan(t.min(), t.max(), color=color_stress if i % 2 == 1 else color_relax, alpha=0.05)

            ax[i].set_xscale('log')
            ax[i].set_ylim(-0.25, 4.5)

            # Hide right spine except last plot
            if i != max_cycles:
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
        ax[0].set_ylabel(r'Threshold Voltage, $V_\mathsf{th}$ [a.u.]', fontsize=8)
        
        legend_elements = [
            Line2D([0], [0], marker='^',markerfacecolor='none', linestyle='none', label='MSM meas. example'),
            Line2D([0], [0], marker='o',markerfacecolor='none', linestyle='none', label='OTF meas. example'),
            Line2D([0], [0], color='k', linestyle='--', label=r'$V_\mathsf{th}$ before stress'),
        ]

        fig.legend(
            handles=legend_elements,
            loc='upper right',
            ncol=4,
            fontsize=7,
            framealpha=0.0,
            columnspacing=1,   # ↓ space between columns (default ~2.0)
            handletextpad=0.4, 
        )

        # Add device info
        #device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{stress}}$ = {df["VgStress"].iloc[0]:.2f} V'
        # ax[0].text(0.05, 0.95, device_text, transform=ax[0].transAxes,fontsize=6, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        # sorted_times = sorted(stress_times)
        # ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
        #      [f'$t_{{stress}}$ = {sorted_times[0]} s', f'$t_{{stress}}$ = {sorted_times[-1]} s'],
        #      fontsize=textSizeLegend, loc='lower right', framealpha=0.9, handlelength=1.5)
        
        plt.savefig(script_dir+"/figures/OTF_stress_time.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot BTI IdVg Recovery
        df = pd.read_csv("data/OTF_raw_data_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '2A13t1') & (df['T'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 6) 
                & ~(df['tMeas'].isin(['initial']))]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)
        df.loc[~df['tMeas'].isin(['initial', 'extra','end']), 'tMeas'] = df.loc[~df['tMeas'].isin(['initial', 'extra','end']),'tMeas'].astype(float)
        width = 39 #um

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get unique stress times and assign colors
        stress_times = df['tMeas'].unique()
        colors = plasma(np.linspace(0, 1, len(stress_times)))
        
        for idx, stress_time in enumerate(sorted(stress_times)):
            df_stress = df[df['tMeas'] == stress_time]
            vg = df_stress['Vg'].values[0]
            id_vals = df_stress['Id'].values[0]
            ax.semilogy(vg, np.array(id_vals)/width*1e6, '-', linewidth=2.5, 
               label=f'$t_{{stress}}$ = {stress_time} s', color=colors[len(colors) - 1 - idx])
        
        # Set axis labels
        ax.set_xlabel(r'$V_\mathsf{G}$ [V]', fontsize=textSize)
        ax.set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=textSize)
        
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add device info
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["T"].iloc[0]}\n$V_D$={df["Vd"].iloc[0]:.2f}\n$V_{{rec}}$ = {df["VgStress"].iloc[0]:.2f} V'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #        fontsize=22, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        sorted_times = sorted(stress_times)
        ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
             [f'$t_{{rec}}$ = {sorted_times[0]} s', f'$t_{{rec}}$ = {sorted_times[-1]} s'],
             fontsize=textSizeLegend, loc='lower right', framealpha=0.9, handlelength=1.5)
        
        plt.title(r'$I_\mathsf{D}-V_\mathsf{G}$ curves during recovery')
        plt.savefig(script_dir+"/figures/OTF_IdVg_recovery.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Plot BTI IdVg (joint plot with recovery)
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 7)]
        for c in ['Id','Vg']:
            df[c] = df[c].map(json5.loads)

        width = df['width'].iloc[0]

        fig, ax = plt.subplots(1,2,figsize=(2.7, 2.5),sharey=True, constrained_layout=False)
        plt.subplots_adjust(wspace=0.00, bottom=0.15, top=0.9, left=0.21, right=0.98)
        
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
        ax[0].text(-0.95, df_stress['Ith'].iloc[0]/width*1e6 - 0.5*1e-4, r'$I_\mathsf{th}$-criterion', fontsize=5, verticalalignment='top', horizontalalignment='left')

        # Set axis labels
        ax[0].spines['right'].set_linestyle('-')
        ax[0].spines['right'].set_alpha(0.5)
        ax[0].set_yscale('log')
        ax[0].set_ylim(4e-9, 1e-2)
        ax[0].set_xlim(-1.15, 2.7)
        ax[0].axvspan(-1.15, 2.7, color=color_stress, alpha=0.05)
        ax[0].text(0.05, 0.95, 'Stress #4', transform=ax[0].transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left')
        ax[0].set_ylabel(r'Drain Current, $I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=8)
        
       # annotate with the time of the first and last stress points
        first_time = sorted(stress_times)[0]
        last_time = sorted(stress_times)[-1]
        ax[0].annotate('', xy=(0.65, 4e-4), xytext=(1.55, 4e-5), arrowprops=dict(arrowstyle='<|-', color='black',shrinkA=0, shrinkB=0))
        ax[0].text(1.1, 2e-5, rf'$t_\mathsf{{str}}$ = $10^{{ {int(np.log10(last_time))} }}$ s', fontsize=5, va='bottom', ha='left')
        ax[0].text(0.65, 4e-4, rf'$t_\mathsf{{str}}$ = {first_time} s', fontsize=5, va='center', ha='right')

        # Plot BTI IdVg Recovery
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 8)]
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
        ax[1].text(0.05, 0.95, 'Relax #4', transform=ax[1].transAxes, fontsize=7, verticalalignment='top', horizontalalignment='left')
        fig.text(0.6, 0.03, r'Gate Voltage, $V_\mathsf{G}$ [V]', ha='center', fontsize=8)
        # ax[1].set_ylabel(r'$I_\mathsf{D}$ [$\mu$A/$\mu$m]', fontsize=8)
        
        # annotate with the time of the first and last stress points
        first_time = sorted(stress_times)[0]
        last_time = sorted(stress_times)[-1]
        ax[1].annotate('', xy=(0.65, 4e-4), xytext=(1.55, 4e-5), arrowprops=dict(arrowstyle='-|>', color='black',shrinkA=0, shrinkB=0))
        ax[1].text(0.65, 4e-4, rf'$t_\mathsf{{relax}}$ = $10^{{ {int(np.log10(last_time))} }}$ s', fontsize=5, va='bottom', ha='right')
        ax[1].text(1.1, 2e-5, rf'$t_\mathsf{{relax}}$ = {first_time} s', fontsize=5, va='center', ha='left')

        
        # Add device info
        # device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{stress}}$ = {df["VgStress"].iloc[0]:.2f} V'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #        fontsize=22, verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        # sorted_times = sorted(stress_times)
        # ax.legend([plt.gca().get_lines()[0], plt.gca().get_lines()[-1]], 
        #      [f'$t_{{stress}}$ = {sorted_times[0]} s', f'$t_{{stress}}$ = {sorted_times[-1]} s'],
        #      fontsize=textSizeLegend, loc='lower right', framealpha=0.9, handlelength=1.5)
        
        plt.savefig(script_dir+"/figures/OTF_IdVg_stressrelax.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot BTI DeltaVth vs time Stress
        df = pd.read_csv("data/OTF_Vth_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 5) 
                & ~(df['tStress'].isin(['initial','extra','end']))]
        df.loc[~df['tStress'].isin(['initial', 'extra','end']), 'tStress'] = df.loc[~df['tStress'].isin(['initial', 'extra','end']),'tStress'].astype(float)
        
        df_fit = pd.read_csv("data/OTF_Vthfit_tStress_TUWien_planar_hbn-encapsulated.csv")
        df_fit = df_fit[(df_fit['dut'] == '2A13t1') & (df_fit['temp'] == '300K') & (df_fit['sample'] == 1) & (df_fit['cycle'] == 5) 
                & ~(df_fit['tStress'].isin(['initial','extra','end']))]
        df_fit.loc[~df_fit['tStress'].isin(['initial', 'extra','end']), 'tStress'] = df_fit.loc[~df_fit['tStress'].isin(['initial', 'extra','end']),'tStress'].astype(float)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs stress time
        ax.plot(df['tStress'], df['Vth_t'] - df['Vth_initial'], 'o', markersize=10,
            markeredgecolor="#13073A", markeredgewidth=2,
            markerfacecolor="#2E8B57", label='exp. data')
        ax.plot(df_fit['tStress'], df_fit['Vth_fit'] - df['Vth_initial'].iloc[0], '-', linewidth=2.5,
            color="#2E8B57", alpha=0.7,label='power-law fit')
        
        ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        # Set axis labels
        ax.axhline(0, linestyle='--', color = 'k')
        ax.set_xlabel(r'$t_{\mathrm{stress}}$ [s]', fontsize=textSize)
        ax.set_ylabel(r'$\Delta V_{\mathsf{th}}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add device info
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{stress}}$ = {df["VgStress"].iloc[0]} V'
        #ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
        #       fontsize=22, verticalalignment='top',
        #       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(r'$\Delta V_\mathsf{th}$ during stress')
        plt.savefig(script_dir+"/figures/OTF_DeltaVth_stress.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    # Plot BTI DeltaVth vs time Recovery
        df = pd.read_csv("data/OTF_Vth_TUWien_planar_hbn-encapsulated.csv")
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1) & (df['cycle'] == 6) 
                & ~(df['tStress'].isin(['initial','extra','end']))]
        df.loc[~df['tStress'].isin(['initial', 'extra','end']), 'tStress'] = df.loc[~df['tStress'].isin(['initial', 'extra','end']),'tStress'].astype(float)
        
        df_fit = pd.read_csv("data/OTF_Vthfit_tStress_TUWien_planar_hbn-encapsulated.csv")
        df_fit = df_fit[(df_fit['dut'] == '2A13t1') & (df_fit['temp'] == '300K') & (df_fit['sample'] == 1) & (df_fit['cycle'] == 6) 
                & ~(df_fit['tStress'].isin(['initial','extra','end']))]
        df_fit.loc[~df_fit['tStress'].isin(['initial', 'extra','end']), 'tStress'] = df_fit.loc[~df_fit['tStress'].isin(['initial', 'extra','end']),'tStress'].astype(float)
        width = 39 #um

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs stress time
        ax.plot(df['tStress'], df['Vth_t'] - df['Vth_initial'], 'o', markersize=10,
            markeredgecolor="#13073A", markeredgewidth=2,
            markerfacecolor="#2E8B57", label='exp. data')
        ax.plot(df_fit['tStress'], df_fit['Vth_fit'] - df['Vth_initial'].iloc[0], '-', linewidth=2.5,
            color="#2E8B57", alpha=0.7,label='powerlaw fit')
        
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        # Set axis labels
        ax.axhline(0, linestyle='--', color = 'k')
        ax.set_xlabel(r'$t_{\mathsf{rec}}$ [s]', fontsize=textSize)
        ax.set_ylabel(r'$\Delta V_{\mathsf{th}}$ [V]', fontsize=textSize)
        ax.set_xscale('log')
        
        #ax.grid(True, which='both', alpha=0.3)
        
        # Add device info
        device_text = f'Device {df["dut"].iloc[0]}\n$T$ = {df["temp"].iloc[0]}\n$V_{{rec}}$ = {df["VgStress"].iloc[0]:.2f} V'
        #ax.text(0.05, 0.05, device_text, transform=ax.transAxes, 
        #       fontsize=22, verticalalignment='bottom',
        #       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(r'$\Delta V_\mathsf{th}$ during recovery')
        plt.savefig(script_dir+"/figures/OTF_DeltaVth_recovery.pdf", bbox_inches="tight", transparent=True)
        plt.close()

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
        
        Vd_text = rf'$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]:.1f} V'
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
        
        Vd_text = rf'$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]} V'
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
        
        Vd_text = rf'$V_\mathsf{{D}}$ = {df['Vd'].iloc[0]} V'
        ax.text(0.05, 0.05, Vd_text, transform=ax.transAxes, 
               fontsize=22, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        #ax.legend(fontsize=textSizeLegend, loc='best', framealpha=0.9)
        
        #plt.title('Hysteresis Width, $V_h$')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_Vd1.pdf", bbox_inches="tight", transparent=True)
        plt.close()

    if 0: # Plot hysteresis DeltaVth vs freq comparisons vacuum vs ambient
        df = pd.read_csv("data/hyst_Vth_TUWien_planar_hbn-encapsulated_amb_vs_vac.csv")
        df_fit = pd.read_csv("data/hyst_Vthfit_freq_TUWien_planar_hbn-encapsulated_amb_vs_vac.csv")
        df = df[(df['dut']=='1A15t1') & (df['sample'].isin([3,4])) & (df['precondition'] == False)]
        df_fit = df_fit[(df_fit['dut']=='1A15t1') & (df_fit['sample'].isin([3,4])) & (df_fit['precondition'] == False)]
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        df_fit_groups = dict(tuple(df_fit.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys()) + list(df_fit_groups.keys())))

        n_keys = len(all_keys)
        # Map colors to vacuum conditions
        vacuum_colors = {'vacuum': '#0047AB', 'ambient': '#FF0000'}
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        # Plot each unique (batch, dut, sample) with distinct color & marker
        for idx, key in enumerate(all_keys):
            # Determine vacuum condition from sample number
            marker = markers[idx % len(markers)]
            if key in df_groups:
                subset = df_groups[key]
                vacuum = subset['vacuum'].iloc[0]
                ax.plot(subset['freq'], subset['DeltaVth'],
                    marker=marker, linestyle=' ', markersize=10,
                    markeredgecolor="#13073A", markeredgewidth=2,
                    markerfacecolor=vacuum_colors[vacuum], label=vacuum)

            if key in df_fit_groups:
                subset_fit = df_fit_groups[key]
                vacuum = subset_fit['vacuum'].iloc[0]
                # plot fits with same color but different line/marker style (no duplicate label)
                ax.plot(subset_fit['freq'], subset_fit['DeltaVth_fit'], linestyle='-', markersize=8,color=vacuum_colors[vacuum])
        
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
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_vacuum_vs_amb.pdf", bbox_inches="tight", transparent=True)
        plt.close()

        ##### Plot hysteresis DeltaVth vs freq comparisons encapsulated vs non-encapsulated
    
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

        ##### Plot hysteresis DeltaVth vs freq different Vd

    if 0: # Plot hysteresis DeltaVth vs freq different Vd
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
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_differentVd.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot hysteresis DeltaVth vs freq duts
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','hyst_TUWien_planar_hbn-encapsulated_nMOS_Eod.csv'))
        df = df[(df['Vd']==1) & (df['precondition'] == False)]

        fig, ax = plt.subplots(figsize=(3.3, 2.25), constrained_layout=False)
        plt.subplots_adjust(left=0.18, right=0.95, top=0.98, bottom=0.18)
        
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
                line_dut, = ax.plot(subset['freq']*(Vmax-Vmin), subset['DeltaVth'],
                    marker=marker, linestyle=' ',
                    markeredgecolor="#13073A",
                    markerfacecolor=WL_color(width/length), label=rf'$W/L$ = {subset['width'].iloc[0]:.0f}/{subset['length'].iloc[0]:.0f};' +f' Array {subset["array"].iloc[0]}; '+ f'Meas {subset["sample"].iloc[0]}' )
                dut_handles.append(line_dut)
                dut_labels.append(dut + rf'$V_{{g,range}}$ = [{subset["Vmin"].iloc[0]:.2f}, {subset["Vmax"].iloc[0]:.2f}] V' )

                ax.plot(subset['freq']*(Vmax-Vmin), subset['DeltaVth_fit'], linestyle='-', markersize=8,color=WL_color(width/length))
        
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
                
        ax.set_xlabel(r'Sweep rate, $r_{\mathsf{sw}}$ [V/s]', fontsize=8)
        ax.set_ylabel(r'Hysteresis Width, $V_\mathsf{H}$ [V]', fontsize=8)
        ax.set_xscale('log')
        #ax.set_xlim(1e-3,200)
        ax.set_ylim(-0.2, 0.4)
        #ax.grid(True, which='both', alpha=0.3)
        
        device_text = r'$T$ = 300 K'
        # ax.text(0.05, 0.95, device_text, transform=ax.transAxes,
        # fontsize=6, verticalalignment='top',
        # bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
        #plt.title('Different devices and voltage ranges')
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_duts.pdf", bbox_inches=None)
        plt.close()

    if 0: # Plot hysteresis DeltaVth different Vd ranges
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
        
        plt.savefig(script_dir+"/figures/hysteresis_DeltaVth_vs_freq_differentVd_range.pdf", bbox_inches=None, transparent=True)
        plt.close()
    
    if 1: # Plot BTI OTF DeltaVth vs different VgStress
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv'))
        df = df[(df['dut'] == '2A13t1') & (df['temp'] == '300K') & (df['sample'] == 1)]
        
        fig, ax = plt.subplots(1,2,figsize=(3.3, 2.25), constrained_layout=False)
        plt.subplots_adjust(wspace=0.0,left=0.18, right=0.95, top=0.98, bottom=0.18)
        
        VgStress_array = [3.0, 4.0, 5.0, 6.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))

        for i,VgStress in enumerate(VgStress_array):
            subset = df[df['VgStress']==VgStress].sort_values(by='tStress')
            ax[0].plot(subset['tStress'], subset['Vth'] - subset['Vth_initial'], 'o', markeredgecolor="#13073A",markerfacecolor=colors[i], label=f'{VgStress:.0f} V' )
            ax[0].plot(subset['tStress'], subset['Vth_fit'] - subset['Vth_initial'].iloc[0], '-',color=colors[i], alpha=0.7)
        
        ax[0].text(0.05, 0.95, 'Stress', transform=ax[0].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='left')
        ax[0].legend(fontsize=6, loc='upper left', framealpha=0.0, title=r'$V_\mathsf{G,stress}$', bbox_to_anchor=(0, 0.925),title_fontsize=7)
        ax[0].axhline(0, linestyle='--', color = 'k')
        # xlim = ax[0].get_xlim()
        # ylim = ax[0].get_ylim()
        ax[0].set_ylim(-0.2, 1.4)

        ax[0].set_xlabel(r'$t_{\mathsf{stress}}$ [s]', fontsize=8)
        ax[0].set_ylabel(r'Threshold Shift, $\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
        ax[0].set_xscale('log')
        ax[0].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[0].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([1,3])))
        ax[0].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))

        
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
        
        ax[1].text(0.95, 0.95, 'Relax \n' + r'$V_\mathsf{{G,rec}}$=0 V', transform=ax[1].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='right')

        # ax[1].text(0.05, 0.25,r'$V_\mathsf{{G,rec}}$ = 0 V', transform=ax[1].transAxes, fontsize=7, verticalalignment='top',horizontalalignment='left')
                   
        # Set axis labels
        ax[1].tick_params(axis='both', which='major', left=False, labelleft=False)
        ax[1].axhline(0, linestyle='--', color = 'k')
        ax[1].set_ylim(-0.2, 1.4)
        ax[1].set_xlabel(r'$t_{\mathsf{relax}}$ [s]', fontsize=8)
        ax[1].set_xscale('log')
        ax[1].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=15))
        ax[1].xaxis.set_major_formatter(FuncFormatter(make_log_formatter([1,3])))
        ax[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
        

        plt.savefig(script_dir+"/figures/OTF_DeltaVth_strrec_differentVstr.pdf", bbox_inches=None)
        plt.close()

        ##### Plot BTI MSM DeltaVth all duts vs different VgStress
    
    if 1: # Plot BTI MSM DeltaVth all duts vs different VgStress
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_MSM.csv'))
        df = df[(df['VgRemain'] == 0.0) & (df['tStress']==100)]
        
        fig, ax = plt.subplots(figsize=(2.7, 2.25), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        plt.subplots_adjust(left=0.65/fig_width, right=1 - 0.04/fig_width, top=1 - 0.1/fig_height, bottom=0.4/fig_height)

        VgStress_array = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))
        str_colors = {Vg: colors[i] for i, Vg in enumerate(VgStress_array)}
        color_legend = [Line2D([0], [0], color=str_colors[Vg],lw=2,
           label=rf'{Vg:.1f} V') for Vg in VgStress_array]

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))
        n_keys = len(all_keys)
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h']

        used_labels = set()
        legend_meas = []
        for i,VgStress in enumerate(VgStress_array[::-1]):
            df_Vstr = df[(df['VgStress']==VgStress)]
            df_groups_Vstr = dict(tuple(df_Vstr.groupby(['batch','dut','sample'])))
            for idx, key in enumerate(all_keys):
                marker = markers[idx % len(markers)]
                if key in df_groups_Vstr:
                    subset = df_groups_Vstr[key]
                    subset = subset.sort_values(by='tRec')

                    # Only label once per key
                    if key not in used_labels:
                        label = rf'$W/L$ = {subset["width"].iloc[0]}/{subset["length"].iloc[0]}; ' \
                        rf'Array {subset["array"].iloc[0]}; Meas {subset["sample"].iloc[0]}; '
                        legend_meas.append(
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

                    ax.plot(subset['tRec'], subset['Vth'] - subset['Vth_initial'],
                        marker=marker, linestyle=' ',
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
        ax.set_xlabel(r'$t_{\mathsf{rec}}$ [s]', fontsize=8)
        ax.set_ylabel(r'Threshold Shift, $\Delta V_{\mathsf{th}}$ [V]', fontsize=8)
        ax.set_xscale('log')
        leg1 =ax.legend(handles=legend_meas, fontsize=6, loc='upper left', bbox_to_anchor=(1, 1), framealpha=1)
        leg2 = ax.legend(
            handles=color_legend,
            fontsize=6,
            loc='upper right',
            framealpha=0.0,
            title=r'$V_\mathsf{G,stress}$',
            title_fontsize=7
        )
        #ax.add_artist(leg1)
        plt.savefig(script_dir+f"/figures/MSM_DeltaVth_duts.pdf", bbox_inches=None)
        plt.close()

    if 1: # Plot BTI MSM DeltaVth all duts vs Eod,str
        df = pd.read_csv(os.path.join(data_folder,'TUWien_planar_hbn-encapsulated','BTI_TUWien_planar_hbn-encapsulated_MSM.csv'))
        df = df[(df['VgRemain'] == 0.0) & (df['tStress']==100)]
        
        fig, ax = plt.subplots(figsize=(3.3, 2.25), constrained_layout=False)
        fig_width, fig_height = fig.get_size_inches()
        plt.subplots_adjust(left=0.65/fig_width, right=1 - 0.2/fig_width, top=1 - 0.1/fig_height, bottom=0.4/fig_height)

        VgStress_array = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        colors = plasma(np.linspace(0.1, 0.9, len(VgStress_array)))
        str_colors = {Vg: colors[i] for i, Vg in enumerate(VgStress_array)}

        df_groups = dict(tuple(df.groupby(['batch','dut','sample'])))
        all_keys = sorted(set(list(df_groups.keys())))
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
                    

                    ax.plot(subset_max['Eod_str'], subset_max['Vth'] - subset_max['Vth_initial'],
                        marker = marker, linestyle=' ',
                        markeredgecolor="#13073A",
                        markerfacecolor=str_colors[VgStress],label=label)
                    
                    ax.plot(subset_begin['Eod_str'], subset_begin['Vth'] - subset_begin['Vth_initial'],
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

    if 0: # Plot BTI DeltaVth hbn-encapsulated_vs_non-encapsulated

        df = pd.read_csv("data/MSM_average_fit_t_hbn-encapsulated_vs_non-encapsulated.csv")
        df = df[(df['tStress']==100) & (df['temp'] == '300K') & (df['VgRemain'] == 0.0) & ~(df['tRec'].isin(['initial','extra','end']))]
        df.loc[~df['tRec'].isin(['initial', 'extra','end']), 'tRec'] = df.loc[~df['tRec'].isin(['initial', 'extra','end']),'tRec'].astype(float)
        
        # Plot DeltaVth vs frequency for different devices and samples
        # Prepare groups and style lists
        VgStress_array = [5.0, 6.0, 7.0]
        # Map colors to vacuum conditions
        batch_colors = {'TUWien_planar_hbn-encapsulated': '#0047AB', 'TUWien_planar_15nm': '#FF0000'}
        markers = ['o', 'v', '^', 's', 'D', 'p', '*', 'h', '+', 'x', '<', '>']

        for i, VgStress in enumerate(VgStress_array[::-1]):

            fig, ax = plt.subplots(figsize=(10, 10))

            df_Vstr = df[df["VgStress"] == VgStress]

            for batch_name, df_batch in df_Vstr.groupby("batch"):

                ax.plot(df_batch['tRec'], df_batch['DeltaVth_mean'],
                        marker='o', linestyle=' ', markersize=10,
                        markeredgecolor="#13073A", markeredgewidth=2,
                        markerfacecolor=batch_colors[batch_name],label=f'{df_batch['batch_info'].iloc[0]}')
                
                ax.plot(df_batch['tRec'], df_batch['DeltaVth_fit'], linestyle='-', markersize=8,color=batch_colors[batch_name])             
            
            if i == 0:
                ylim = ax.get_ylim()
            else:
                ax.set_ylim(ylim)
            ax.axhline(0, linestyle='--', color = 'k')
            device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V\n $t_{{str}}$ = {df["tStress"].iloc[0]} s\n$V_{{G,rec}}$ = {df["VgRemain"].iloc[0]} V'
            ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
                fontsize=22, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        
            ax.set_xlabel(r'$t_{\mathsf{rec}}$ [s]', fontsize=textSize)
            ax.set_ylabel(r'$\Delta V_{\mathsf{th}}$ [V]', fontsize=textSize)
            ax.set_xscale('log')
            ax.legend(fontsize=textSizeLegend, loc='upper right', framealpha=0.9)
            plt.savefig(script_dir+f"/figures/MSM_DeltaVth_comparison_{VgStress}.pdf", bbox_inches="tight", transparent=True)
            plt.close()
        
        for i, VgStress in enumerate(VgStress_array[::-1]):

            fig, ax = plt.subplots(figsize=(10, 10))

            df_Vstr = df[df["VgStress"] == VgStress]

            for batch_name, df_batch in df_Vstr.groupby("batch"):
                tox = df_batch['tox'].iloc[0]
                ax.plot(df_batch['tRec'], df_batch['DeltaVth_mean']/tox*10e-8,
                        marker='o', linestyle=' ', markersize=10,
                        markeredgecolor="#13073A", markeredgewidth=2,
                        markerfacecolor=batch_colors[batch_name],
                        label=f'{df_batch['batch_info'].iloc[0]} raw')
                
                ax.plot(df_batch['tRec'], df_batch['DeltaVth_fit']/tox*10e-8,
                        linestyle='-', markersize=8, color=batch_colors[batch_name],
                        label=f'{df_batch['batch_info'].iloc[0]} fit')             
            
            if i == 0:
                ylim = ax.get_ylim()
            else:
                ax.set_ylim(ylim)
            ax.axhline(0, linestyle='--', color = 'k')
            device_text = f'$T$ = {df["temp"].iloc[0]}\n$V_{{G,str}}$ = {VgStress:.1f} V'
            # ax.text(0.05, 0.95, device_text, transform=ax.transAxes, 
            #     fontsize=22, verticalalignment='top',
            #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
            ax.set_xlabel(r'$t_{\mathsf{rec}}$ [s]', fontsize=textSize)
            ax.set_ylabel(r'$\Delta V_{\mathsf{th}}/t_\mathsf{ox}$ [MV/cm]', fontsize=textSize)
            ax.set_xscale('log')
            ax.legend(fontsize=textSizeLegend, loc='upper right', framealpha=0.9)
            plt.savefig(script_dir+f"/figures/MSM_DeltaVth_tox_comparison_{VgStress}.pdf", bbox_inches="tight", transparent=True)
            plt.close()
    
    if 0: # Measuring Vth vs measuring one point Id
        df = pd.read_csv("data/TUWien_planar_hbn-encapsulated/BTI_TUWien_planar_hbn-encapsulated_OTF_nMOS.csv")
        df.loc[~df['tStress'].isin(['initial', 'extra','end']), 'tStress'] = df.loc[~df['tStress'].isin(['initial', 'extra','end']),'tStress'].astype(float)

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
                            markerfacecolor='b',label=rf'$W/L$ = {subset['width'].iloc[0]}/{subset['length'].iloc[0]}; Array {subset['array'].iloc[0]}; Meas {subset['sample'].iloc[0]}; $E_{{od,str}}$ = {subset['Eod_str'].iloc[0]:.2f} MV/cm')
                    else:
                        ax2.plot(subset['tStress'], subset['I'],
                            marker=marker, linestyle=' ', markersize=10,
                            markeredgecolor="#13073A", markeredgewidth=2,
                            markerfacecolor='r',label=rf'$W/L$ = {subset['width'].iloc[0]}/{subset['length'].iloc[0]}; Array {subset['array'].iloc[0]}; Meas {subset['sample'].iloc[0]}; $E_{{od,str}}$ = {subset['Eod_str'].iloc[0]:.2f} MV/cm')
                        
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
    
    sys.exit()
