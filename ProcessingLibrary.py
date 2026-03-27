import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import re
import os
import copy
import pandas as pd
import scienceplots
from scipy import interpolate, optimize, special
from scipy.ndimage import uniform_filter1d
from datetime import datetime
from pathlib import Path
import InterpolationHelper
import json5

plt.style.use(['nature'])
plt.rcParams['font.family'] = 'sans-serif'

script_dir = os.path.dirname(os.path.abspath(__file__))

def read_data(folders, meas, dataset, submeas=None):
    data = []
    if submeas is None:
        submeas = meas
    for folder in folders:
        folder_path = os.path.join(script_dir, 'data', folder, meas)
        #os.makedirs(folder_path, exist_ok=True)
        batch = folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.crv'):
                meas_type = filename.split('_', 1)[0]
                if meas_type == 'IdVg':
                    match = re.match(
                    rf'IdVg_{batch}_(\w+)_(\d+)_(\w+)_Vd_(-?(?:\d+(?:\.\d+)?|\.\d+))V(?:_Vbg_(-?(?:\d+(?:\.\d+)?|\.\d+))V_)?(?:_sweep_(\d+)_)?.crv',
                    filename
                    )
                    dut = match.group(1)
                    sample = int(match.group(2))
                    T = match.group(3)
                    Vd = float(match.group(4))
                    Vbg = float(match.group(5)) if match.group(5) is not None else None
                    sweep = int(match.group(6)) if match.group(6) is not None else None
                    date = extract_date_start(os.path.join(folder_path, filename))
                    df_meas = extract_data_crv(os.path.join(folder_path, filename))
                    row = {
                        'batch': batch,
                        'dut':dut,
                        'sample': sample,
                        'T': T,
                        'sweep': sweep,
                        'Vd': Vd,
                        'Vbg': Vbg,
                        'date': date,
                        **{col: df_meas[col].tolist() for col in df_meas.columns}
                    }
                    data.append(row)
                elif meas_type == 'inv-transfer':
                    match = re.match(
                    rf'inv-transfer_{batch}_(\w+)_(\d+)_(\w+)_Vsupply_(-?(?:\d+(?:\.\d+)?|\.\d+))V(?:_sweep(\d+))?.crv',
                    filename
                    )
                    dut = match.group(1)
                    sample = int(match.group(2))
                    T = match.group(3)
                    Vsupply = float(match.group(4))
                    sweep = int(match.group(5)) if match.group(5) is not None else None
                    # Vbg = float(match.group(6)) if match.group(6) is not None else None
                    date = extract_date_start(os.path.join(folder_path, filename))
                    df_meas = extract_data_crv(os.path.join(folder_path, filename))
                    data.append({
                        'meas_type': 'inv-transfer',
                        'batch': batch,
                        'dut':dut,
                        'sample': sample,
                        'T': T,
                        'Vdd': Vsupply,
                        'sweep':sweep,
                        'date': date,
                        **{col: df_meas[col].tolist() for col in df_meas.columns}
                    })
                elif meas_type == 'MSM':
                    match = re.match(
                    rf'MSM_{batch}_(\w+)_(\d+)_(\w+)_tStress_(\d+)s_VtgStress_(-?(?:\d+(?:\.\d+)?|\.\d+))V_(?:VtgRemain_(-?(?:\d+(?:\.\d+)?|\.\d+))V_)?(?:cycle(\d+)_)?(initial|\d+(?:\.\d+)?s|extra|end).crv',
                    filename
                    )
                    dut = match.group(1)
                    sample = int(match.group(2))
                    T = match.group(3)
                    tStress = int(match.group(4))
                    VtgStress = float(match.group(5))
                    if match.group(6) == None:
                        Vgremain = extract_vremain(os.path.join(folder_path, filename))
                    else:
                        Vgremain = float(match.group(6))
                    date = extract_date_start(os.path.join(folder_path, filename))
                    cycle = int(match.group(7)) if match.group(7) != None else None
                    tMeas = match.group(8) if match.group(8) in ['initial','extra','end'] else float(match.group(8)[:-1])
                    df_meas = extract_data_crv(os.path.join(folder_path, filename))
                    data.append({
                        'meas_type': meas_type,
                        'batch': batch,
                        'dut':dut,
                        'sample': sample,
                        'T': T,
                        'tStress': tStress,
                        'VgStress': VtgStress,
                        'VgRemain': Vgremain,
                        'cycle': cycle,
                        'tMeas': tMeas,
                        'date': date,
                        **{col: df_meas[col].tolist() for col in df_meas.columns}
                    })
                elif meas_type == 'OTF':
                    match = re.match(
                    rf'OTF_{batch}_(\w+)_(\d+)_(\w+)_VtgStress_(-?(?:\d+(?:\.\d+)?|\.\d+))V_(?:cycle(\d+)_)?(initial|\d+(?:\.\d+)?s|end|extra).crv',
                    filename
                    )
                    dut = match.group(1)
                    sample = int(match.group(2))
                    T = match.group(3)
                    VtgStress = float(match.group(4))
                    cycle = int(match.group(5)) if match.group(5) != None else None
                    tMeas = match.group(6) if match.group(6) in ['initial','extra','end'] else float(match.group(6)[:-1])
                    date = extract_date_start(os.path.join(folder_path, filename))
                    df_meas = extract_data_crv(os.path.join(folder_path, filename))
                    data.append({
                        'meas_type': meas_type,
                        'batch': batch,
                        'dut':dut,
                        'sample': sample,
                        'T': T,
                        'VgStress': VtgStress,
                        'cycle': cycle,
                        'tMeas':tMeas,
                        'date': date,
                        **{col: df_meas[col].tolist() for col in df_meas.columns}
                    })
                elif meas_type == 'fts':
                        match = re.match(
                        rf'fts_{batch}_{dut}_{sample}_{T}_Vmin(-?(?:\d+(?:\.\d+)?|\.\d+))_Vmax(-?(?:\d+(?:\.\d+)?|\.\d+))_dVtg([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)_tsampling([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)_N_steps(-?(?:\d+(?:\.\d+)?|\.\d+))_cycle(\d+)(?:_sweep(\d+))?.crv',
                        filename
                        )
                        Vmin = float(match.group(1))
                        Vmax = float(match.group(2))
                        dV = float(match.group(3))
                        tsampling = float(match.group(4))
                        N_steps = int(float(match.group(5)))
                        cycle = int(match.group(6))
                        SweepIndex = int(match.group(7)) if match.group(7) else None
                        date = extract_date_start(os.path.join(folder_path, filename))
                        df_meas = extract_data_crv(os.path.join(folder_path, filename))
                        N_steps_actual = len(df_meas['Vinput'])
                        SweepRate = np.mean(np.abs(dV / tsampling))
                        nom_freq = SweepRate / (2 * abs(Vmin - Vmax))
                        precondition = False
                        dV_mean = np.mean(np.abs(df_meas['Vinput'].diff()))  # Mean of dV
                        dt_mean = np.mean(df_meas['t'].diff())  # Mean of dt
                        act_SweepRate = np.mean(np.abs(dV_mean / dt_mean))
                        act_freq = act_SweepRate / (2 * abs(np.max(df_meas['Vinput']) - np.min(df_meas['Vinput'])))
                        data.append({
                            'batch': batch,
                            'dut':dut,
                            'meas_type': meas_type,
                            'sample': sample,
                            'T': T,
                            'cycle': cycle,
                            "precondition": precondition,
                            "SweepIndex": SweepIndex,
                            "nom_freq": nom_freq,
                            "freq": act_freq,
                            'date': date,
                            **{col: df_meas[col].tolist() for col in df_meas.columns}
                        })
            elif filename.endswith('.csv'):
                meas_type = filename.split('_', 1)[0]
                if meas_type == 'hyst':
                    match = re.match(
                    rf'hyst_{batch}_(\w+)_(\d+)_(\w+)_(Hyst|PreCond)_Vd_(-?(?:\d+(?:\.\d+)?|\.\d+))_f_(-?(?:\d+(?:\.\d+)?|\.\d+))Hz-(-?(?:\d+(?:\.\d+)?|\.\d+))Hz_TotalPreCond_(-?(?:\d+(?:\.\d+)?|\.\d+))min.csv',
                    filename
                    )
                    dut = match.group(1)
                    sample = int(match.group(2))
                    T = match.group(3)
                    precondition = True if match.group(4) == 'PreCond' else False
                    # Vd = float(match.group(5))
                    # fmax = float(match.group(6))
                    # fmin = float(match.group(7))
                    # precondtime = float(match.group(8))
                    with open(os.path.join(folder_path, filename), "r") as f:
                        Dict = json5.loads(f.readline())   # line 1 → JSON
                        Data = pd.read_csv(f)                    # rest → CSV
                    data_axel = extract_data_AxelHyst(Dict, Data, precondition, batch, dut, sample, T)
                    data.extend(data_axel)
                if meas_type == 'IdVg':
                    match = re.match(
                    rf'IdVg_{batch}_(\w+)_(\d+)_(\w+)_(Vd|VarVd)_(?:VdStart_)?(-?(?:\d+(?:\.\d+)?|\.\d+))V_(?:(?:VdEnd_)?(-?(?:\d+(?:\.\d+)?|\.\d+))V)?_f_(-?(?:\d+(?:\.\d+)?|\.\d+))Hz.csv',
                    filename
                    )
                    dut = match.group(1)
                    sample = int(match.group(2))
                    T = match.group(3)
                    varVd = True if match.group(4) == 'VarVd' else False
                    with open(os.path.join(folder_path, filename), "r") as f:
                        Dict = json5.loads(f.readline())   # line 1 → JSON
                        Data = pd.read_csv(f)                    # rest → CSV
                    data_axel = extract_data_AxelIdVg(Dict, Data, varVd, batch, dut, sample, T)
                    data.extend(data_axel)
    df = pd.DataFrame(data)
    for c in ['time','Id','Vg','Ig','Vinput','Voutput','Isupply','Vd']:
        if c in df.columns:
            df[c] = df[c].map(json5.dumps)
    if meas == 'BTI':
        for (batch,dut,sample), subset_meas in df.groupby(['batch','dut','sample']):
            if subset_meas['cycle'].isna().any():
                initial_dates = subset_meas[subset_meas['tMeas'] == 'initial']['date'].sort_values()
                for _,row in subset_meas.iterrows():
                    n_before = (initial_dates <= row['date']).sum()
                    df.loc[row.name, 'cycle'] = max(1, n_before)
    if meas == 'IdVg' or meas == 'inv-transfer' or meas == 'fts':
        for (batch,dut,sample), subset_meas in df.groupby(['batch','dut','sample']):
            if subset_meas['sweep'].isna().any():
                dates = subset_meas['date'].sort_values()
                for _,row in subset_meas.iterrows():
                    n_before = (dates <= row['date']).sum()
                    df.loc[row.name, 'sweep'] = n_before
    filename = f'{submeas}_raw_data_{dataset}.csv'
    folderpath = os.path.join(script_dir,'data',dataset)
    os.makedirs(folderpath, exist_ok=True)
    df.to_csv(os.path.join(folderpath,filename), index=False)
    return df

def read_data_from_meas(dataset, meas, meas_type, name=None):
    data = []
    meas_types = {'BTI': [{'MSM':'crv'},{'OTF':'crv'}],
    'IdVg': [{'IdVg':'crv'},{'IdVgAxel':'csv'}],
    'inv-transfer': [{'inv-transfer':'crv'}],
    'hyst': [{'hyst':'csv'},{'fts':'crv'}],
    'OTF': [{'OTF':'crv'}],
    'MSM': [{'MSM':'crv'}]
    }
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    for batch,batch_meas in meas.items():
        folder_path = os.path.join(script_dir, 'data', batch, meas_type)
        for me in meas_types[meas_type]:
            mtype, exten = next(iter(me.items()))
            for m in batch_meas['meas'][meas_type].values():
                dut = m['dut']['name']
                sample = m['sample']
                T = m['temp']
                prefix = f'{mtype}_{batch}_{dut}_{sample}_{T}_'
                files = list(Path(folder_path).glob(prefix + f"*{exten}")) 
                for file in files:
                    filename = file.name
                    if mtype == 'IdVg':
                        match = re.match(
                        rf'IdVg_{batch}_{dut}_{sample}_{T}_Vd_(-?(?:\d+(?:\.\d+)?|\.\d+))V(?:_Vbg_(-?(?:\d+(?:\.\d+)?|\.\d+))V_)?(?:_sweep_(\d+)_)?.crv',
                        filename
                        )
                        Vd = float(match.group(1))
                        Vbg = float(match.group(2)) if match.group(2) is not None else None
                        sweep = int(match.group(3)) if match.group(3) is not None else None
                        date = extract_date_start(os.path.join(folder_path, filename))
                        df_meas = extract_data_crv(os.path.join(folder_path, filename))
                        row = {
                            'batch': batch,
                            'dut':dut,
                            'sample': sample,
                            'T': T,
                            'sweep': sweep,
                            'Vd': Vd,
                            'Vbg': Vbg,
                            'date': date,
                            **{col: df_meas[col].tolist() for col in df_meas.columns}
                        }
                        data.append(row)
                    elif mtype == 'inv-transfer':
                        match = re.match(
                        rf'inv-transfer_{batch}_{dut}_{sample}_{T}_Vsupply_(-?(?:\d+(?:\.\d+)?|\.\d+))V(?:_sweep(\d+))?.crv',
                        filename
                        )
                        Vsupply = float(match.group(1))
                        sweep = int(match.group(2)) if match.group(2) is not None else None
                        date = extract_date_start(os.path.join(folder_path, filename))
                        df_meas = extract_data_crv(os.path.join(folder_path, filename))
                        data.append({
                            'meas_type': 'inv-transfer',
                            'batch': batch,
                            'dut':dut,
                            'sample': sample,
                            'T': T,
                            'Vdd': Vsupply,
                            'sweep':sweep,
                            'date': date,
                            **{col: df_meas[col].tolist() for col in df_meas.columns}
                        })
                    elif mtype == 'MSM':
                        match = re.match(
                        rf'MSM_{batch}_{dut}_{sample}_{T}_tStress_(\d+)s_VtgStress_(-?(?:\d+(?:\.\d+)?|\.\d+))V_(?:VtgRemain_(-?(?:\d+(?:\.\d+)?|\.\d+))V_)?(?:cycle(\d+)_)?(initial|\d+(?:\.\d+)?s|extra|end).crv',
                        filename
                        )
                        tStress = int(match.group(1))
                        VtgStress = float(match.group(2))
                        if match.group(3) == None:
                            Vgremain = extract_vremain(os.path.join(folder_path, filename))
                        else:
                            Vgremain = float(match.group(3))
                        date = extract_date_start(os.path.join(folder_path, filename))
                        cycle = int(match.group(4)) if match.group(4) != None else None
                        tMeas = match.group(5) if match.group(5) in ['initial','extra','end'] else float(match.group(5)[:-1])
                        df_meas = extract_data_crv(os.path.join(folder_path, filename))
                        data.append({
                            'meas_type': mtype,
                            'batch': batch,
                            'dut':dut,
                            'sample': sample,
                            'T': T,
                            'tStress': tStress,
                            'VgStress': VtgStress,
                            'VgRemain': Vgremain,
                            'cycle': cycle,
                            'tMeas': tMeas,
                            'date': date,
                            **{col: df_meas[col].tolist() for col in df_meas.columns}
                        })
                    elif mtype == 'OTF':
                        match = re.match(
                        rf'OTF_{batch}_{dut}_{sample}_{T}_VtgStress_(-?(?:\d+(?:\.\d+)?|\.\d+))V_(?:cycle(\d+)_)?(initial|\d+(?:\.\d+)?s|end|extra).crv',
                        filename
                        )
                        VtgStress = float(match.group(1))
                        cycle = int(match.group(2)) if match.group(2) != None else None
                        tMeas = match.group(3) if match.group(3) in ['initial','extra','end'] else float(match.group(3)[:-1])
                        date = extract_date_start(os.path.join(folder_path, filename))
                        df_meas = extract_data_crv(os.path.join(folder_path, filename))
                        data.append({
                            'meas_type': mtype,
                            'batch': batch,
                            'dut':dut,
                            'sample': sample,
                            'T': T,
                            'VgStress': VtgStress,
                            'cycle': cycle,
                            'tMeas':tMeas,
                            'date': date,
                            **{col: df_meas[col].tolist() for col in df_meas.columns}
                        })
                    elif mtype == 'fts':
                        match = re.match(
                        rf'fts_{batch}_{dut}_{sample}_{T}_Vmin(-?(?:\d+(?:\.\d+)?|\.\d+))_Vmax(-?(?:\d+(?:\.\d+)?|\.\d+))_dVtg([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)_tsampling([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)_N_steps(-?(?:\d+(?:\.\d+)?|\.\d+))_cycle(\d+)(?:_sweep(\d+))?.crv',
                        filename
                        )
                        Vmin = float(match.group(1))
                        Vmax = float(match.group(2))
                        dV = float(match.group(3))
                        tsampling = float(match.group(4))
                        N_steps = int(float(match.group(5)))
                        cycle = int(match.group(6))
                        SweepIndex = int(match.group(7)) if match.group(7) else None
                        date = extract_date_start(os.path.join(folder_path, filename))
                        df_meas = extract_data_crv(os.path.join(folder_path, filename))
                        N_steps_actual = len(df_meas['Vinput'])
                        SweepRate = np.mean(np.abs(dV / tsampling))
                        nom_freq = SweepRate / (2 * abs(Vmin - Vmax))
                        precondition = False
                        dV_mean = np.mean(np.abs(df_meas['Vinput'].diff()))  # Mean of dV
                        dt_mean = np.mean(df_meas['t'].diff())  # Mean of dt
                        act_SweepRate = np.mean(np.abs(dV_mean / dt_mean))
                        act_freq = act_SweepRate / (2 * abs(np.max(df_meas['Vinput']) - np.min(df_meas['Vinput'])))
                        data.append({
                            'batch': batch,
                            'dut':dut,
                            'meas_type': mtype,
                            'sample': sample,
                            'T': T,
                            'cycle': cycle,
                            "precondition": precondition,
                            "SweepIndex": SweepIndex,
                            "nom_freq": nom_freq,
                            "freq": act_freq,
                            'date': date,
                            **{col: df_meas[col].tolist() for col in df_meas.columns}
                        })
                    elif mtype == 'hyst':
                        match = re.match(
                        rf'hyst_{batch}_{dut}_{sample}_{T}_(Hyst|PreCond)_Vd_(-?(?:\d+(?:\.\d+)?|\.\d+))_f_(-?(?:\d+(?:\.\d+)?|\.\d+))Hz-(-?(?:\d+(?:\.\d+)?|\.\d+))Hz_TotalPreCond_(-?(?:\d+(?:\.\d+)?|\.\d+))min.csv',
                        filename
                        )
                        precondition = True if match.group(1) == 'PreCond' else False
                        with open(os.path.join(folder_path, filename), "r") as f:
                            Dict = json5.loads(f.readline())   # line 1 → JSON
                            Data = pd.read_csv(f)                    # rest → CSV
                        data_axel = extract_data_AxelHyst(Dict, Data, precondition, batch, dut, sample, T)
                        data.extend(data_axel)
                    elif meas_type == 'IdVg':
                        match = re.match(
                        rf'IdVg_{batch}_{dut}_{sample}_{T}_(Vd|VarVd)_(?:VdStart_)?(-?(?:\d+(?:\.\d+)?|\.\d+))V_(?:(?:VdEnd_)?(-?(?:\d+(?:\.\d+)?|\.\d+))V)?_f_(-?(?:\d+(?:\.\d+)?|\.\d+))Hz.csv',
                        filename
                        )
                        varVd = True if match.group(1) == 'VarVd' else False
                        with open(os.path.join(folder_path, filename), "r") as f:
                            Dict = json5.loads(f.readline())   # line 1 → JSON
                            Data = pd.read_csv(f)                    # rest → CSV
                        data_axel = extract_data_AxelIdVg(Dict, Data, varVd, batch, dut, sample, T)
                        data.extend(data_axel)
    df = pd.DataFrame(data)
    for c in ['t','Id','Vg','Ig','Vinput','Voutput','Isupply','Vd']:
        if c in df.columns:
            df[c] = df[c].map(json5.dumps)
    if meas_type == 'BTI':
        for (batch,dut,sample), subset_meas in df.groupby(['batch','dut','sample']):
            if subset_meas['cycle'].isna().any():
                initial_dates = subset_meas[subset_meas['tMeas'] == 'initial']['date'].sort_values()
                for _,row in subset_meas.iterrows():
                    n_before = (initial_dates <= row['date']).sum()
                    df.loc[row.name, 'cycle'] = max(1, n_before)
    if meas_type == 'IdVg' or meas_type == 'inv-transfer' or meas_type == 'fts':
        for (batch,dut,sample), subset_meas in df.groupby(['batch','dut','sample']):
            if subset_meas['sweep'].isna().any():
                dates = subset_meas['date'].sort_values()
                for _,row in subset_meas.iterrows():
                    n_before = (dates <= row['date']).sum()
                    df.loc[row.name, 'sweep'] = n_before
    folderpath = os.path.join(script_dir,'data',dataset)
    if name is not None:
        filename = f'{meas_type}_raw_data_{dataset}_{name}.csv'
    else:
        filename = f'{meas_type}_raw_data_{dataset}.csv'
    os.makedirs(folderpath, exist_ok=True)
    df.to_csv(os.path.join(folderpath,filename), index=False)
    return df

def data_IdVg(dataset, add_data='all', meas=None, name=None):
    IdVg_data = []
    additional_data = {}
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'IdVg_raw_data_{dataset}.csv'))
    for c in ['VTopGate','IDrain','ITopGate','VDrain']:
        df[c] = df[c].map(json5.loads)
    for batch,batch_meas in meas.items():
        tox = batch_meas['oxide_thickness'] if 'oxide_thickness' in batch_meas else None
        tox = float(re.sub(r'[^0-9.+-]', '', tox))
        additional_data['tox'] = tox
        additional_data['batch_info'] = batch_meas['batch_info'] if 'batch_info' in batch_meas else None
        for m in batch_meas['meas']['IdVg'].values():
            dut = m['dut']['name']
            width = parse_length(m['dut']['width'], target_unit="um")
            length = parse_length(m['dut']['length'], target_unit="um")
            area = width*length
            sample = m['sample']
            T = m['temp']
            fit_IdVg = m['fit_IdVg'] if 'fit_IdVg' in m else None
            noise_level = m['noise_level'] if 'noise_level' in m else 0
            additional_data['width'] = width
            additional_data['length'] = length
            additional_data['area'] = area
            additional_data['array'] = m['dut']['array'] if 'array' in m['dut'] else None
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['T'] == T) &
                    (df['sample'] == sample)
                ]
            for ind,row in df_meas.iterrows():
                sweep = row['sweep']
                Id = np.abs(row['IDrain']).tolist()
                Vg = row['VTopGate']
                Vd = row['VDrain']
                Ig = np.abs(row['ITopGate']).tolist() if 'ITopGate' in row else None
                main_data = {'batch':batch,'dut':dut,'sample':sample,'temp':T,'Vd':row['Vd'],'sweep':sweep}
                if add_data == 'all':
                    remaining_data = ['IdVg','IdVg_fit','Vth','SS','Imax','Ig','Ion/Ioff','gate_leakage'] + list(additional_data.keys())
                else:
                    remaining_data = copy.deepcopy(add_data)
                if 'IdVg' in add_data or add_data == 'all':
                    main_data['Id'] = Id 
                    main_data['Vg'] = Vg
                    remaining_data.remove('IdVg')
                if 'Ig' in add_data or add_data == 'all':
                    if Ig is not None:
                        main_data['Ig'] = Ig
                    remaining_data.remove('Ig')
                if 'IdVg_fit' in add_data or add_data == 'all':
                    if fit_IdVg is not None:
                        Id_fit,Vg_fit,_,_ = fit_data_IdVg(Id,Vg,Id,Vg,fit_IdVg,noise_level=noise_level,temp=float(T.rstrip('K')))
                        main_data['Id_fit'] = Id_fit.tolist()
                        main_data['Vg_fit'] = Vg_fit.tolist()
                        main_data['fit_IdVg'] = fit_IdVg
                    remaining_data.remove('IdVg_fit')
                if 'Vth' in add_data or add_data == 'all':
                    if m['vth_extract'] and 'method' in m['vth_extract']:
                        vth_extract = m['vth_extract']['method']
                        if vth_extract == 'constant_current' or 'constant_current_L/W':
                            current_level = m['vth_extract']['current_level']
                        else:
                            current_level = None
                        Vth,Ith,_,_ = Vth_extraction(Vg, Id, Vd, vth_extract, current_level, width=width, length=length)
                        main_data['Vth'] = Vth
                        main_data['Ith'] = Ith
                        main_data['vth_extract'] = vth_extract
                        remaining_data.remove('Vth')
                if 'SS' in add_data or add_data == 'all':
                    if 'ss_extract' in m:
                        ss_extract = m['ss_extract']['method']
                        if ss_extract == 'orders_above_noise':
                            order = m['ss_extract']['order']
                        else:
                            order = None
                        SS,Vzero_current,_,_ = SS_extraction(Vg,Id,Vd,ss_extract=ss_extract,noise_level=noise_level,order=order)
                        main_data['SS'] = SS*1000  if SS is not None else None
                        main_data['ss_extract'] = ss_extract 
                        main_data['Vzero_current'] = Vzero_current
                        remaining_data.remove('SS')
                if 'Ion/Ioff' in add_data or add_data == 'all':
                    Ion = np.max(np.abs(Id))
                    Ioff = np.min([noise_level,np.min(np.abs(Id))])
                    main_data['Ion/Ioff'] = Ion / Ioff
                    remaining_data.remove('Ion/Ioff')
                if 'Imax' in add_data or add_data == 'all':
                    Imax = np.max(np.abs(Id))
                    main_data['Imax'] = Imax
                    remaining_data.remove('Imax')
                if 'gate_leakage' in add_data or add_data == 'all':
                    if Ig is not None:
                        main_data['gate_leakage'] = np.max(np.abs(Ig))/area*1e8 if np.max(np.abs(Ig)) > noise_level else noise_level/area*1e8
                    remaining_data.remove('gate_leakage')
                for info in remaining_data:
                    main_data[info] = additional_data[info]
                IdVg_data.append(main_data)
    df_IdVg = pd.DataFrame(IdVg_data)
    for c in ['Id','Vg','Id_fit','Vg_fit','Ig']:
        if c in df_IdVg.columns:
            df_IdVg[c] = df_IdVg[c].apply(
            lambda x: json5.dumps(x.tolist())
            if isinstance(x, np.ndarray)
            else x
        )
    if name is None:
        df_IdVg.to_csv(os.path.join(script_dir,'data',dataset,f'IdVg_{dataset}.csv'), index=False)
    else:
        df_IdVg.to_csv(os.path.join(script_dir,'data',dataset,f'IdVg_{dataset}_{name}.csv'), index=False)
    return df_IdVg

def data_sweep(dataset, add_data='all', meas=None, df=None, meas_type='inv-transfer', name=None):
    sweep_data = []
    additional_data = {}
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    if df is None:
        df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'{meas_type}_raw_data_{dataset}.csv'))
    for c in ['t','VGate','IDrain','IGate','Vinput','Voutput','Isupply','VDrain','Vsupply']:
        if c in df.columns:
            df[c] = df[c].map(safe_json_load)
    for batch,batch_meas in meas.items():
        additional_data['tox'] = float(re.sub(r'[^0-9.+-]', '', batch_meas['oxide_thickness'])) if 'oxide_thickness' in batch_meas else None
        additional_data['batch_info'] = batch_meas['batch_info'] if 'batch_info' in batch_meas else None
        for m in batch_meas['meas'][meas_type].values():
            dut = m['dut']['name']
            sample = m['sample']
            T = m['temp']
            meas_type = m['meas_type'] if 'meas_type' in m else meas_type
            additional_data['width'] = parse_length(m['dut']['width'],target_unit='um') if 'width' in m['dut'] else None
            additional_data['length'] = parse_length(m['dut']['length'],target_unit='um') if 'length' in m['dut'] else None
            additional_data['area'] = parse_length(m['dut']['width'],target_unit='um')*parse_length(m['dut']['length'],target_unit='um') if 'width' in m['dut'] and 'length' in m['dut'] else None
            additional_data['array'] = m['dut']['array'] if 'array' in m['dut'] else None
            additional_data['vacuum'] = m['vacuum'] if 'vacuum' in m else None
            additional_data['noise_level']=m['noise_level'] if 'noise_level' in m else None
            additional_data['stress'] = m['stress'] if 'stress' in m else None
            additional_data['acc_cycle'] = m['acc_cycle'] if 'acc_cycle' in m else None
            additional_data['ref'] = m['ref'] if 'ref' in m else None
            additional_data['fit_invtransfer'] = m['fit_invtransfer'] if 'fit_invtransfer' in m else None
            vth_extract = m['vth_extract']['method'] if'vth_extract' in m else None
            if vth_extract == 'constant_current':
                current_level = m['vth_extract']['current_level']
            else:
                current_level = None
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['T'] == T) &
                    (df['sample'] == sample) &
                    (df['meas_type'] == meas_type)
                ]
            for row in df_meas.iterrows():
                row = row[1]  # Extract the actual row data from the iterrows tuple
                sweep = row['sweep']
                main_data = {'meas_type':meas_type,'batch':batch,'dut':dut,'sample':sample,'temp':T,'sweep':sweep}
                if meas_type == 'IdVg':
                    Vd = row['Vd']
                    Vg = np.array(row['Vg'])
                    Id = np.array(row['Id'])
                    mask = ~np.isnan(Vg) & ~np.isnan(Id)
                    Vg = Vg[mask]
                    Id = np.abs(Id[mask])
                    main_data['Vd'] = row['Vd']
                    main_data['Vg'] = Vg
                    main_data['Id'] = Id
                    if add_data == 'all':
                        remaining_data = ['IdVg','IdVg_fit','Vth','SS','I_max'] + list(additional_data.keys())
                    else:
                        remaining_data = add_data.copy()
                    if 'sweep_rate' in add_data or add_data == 'all':
                        total_time = time[-1] - time[0] if len(time) > 1 else 0
                        sweep_rate = (Vg[-1] - Vg[0]) / total_time if total_time > 0 else None
                        main_data['sweep_rate'] = sweep_rate
                        remaining_data.remove('sweep_rate')
                    if 'IdVg' in add_data or add_data == 'all':
                        main_data['Id'] = Id 
                        main_data['Vg'] = Vg
                        remaining_data.remove('IdVg')
                    if 'IdVg_fit' in add_data or add_data == 'all':
                        if additional_data['fit_IdVg'] is not None:
                            Id_fit,Vg_fit,_,_ = fit_data_IdVg(Id,Vg,Id,Vg,additional_data['fit_IdVg'],noise_level=additional_data['noise_level'],temp=float(T.rstrip('K')))
                            main_data['Id_fit'] = Id_fit.tolist()
                            main_data['Vg_fit'] = Vg_fit.tolist()
                            main_data['fit_IdVg'] = additional_data['fit_IdVg']
                        remaining_data.remove('IdVg_fit')
                    if 'Vth' in add_data or add_data == 'all':
                        if m['vth_extract'] and 'method' in m['vth_extract']:
                            vth_extract = m['vth_extract']['method']
                            if vth_extract == 'constant_current' or 'constant_current_L/W':
                                current_level = m['vth_extract']['current_level']
                            else:
                                current_level = None
                            Vth,Ith,_,_ = Vth_extraction(Vg, Id, Vd, vth_extract, current_level, width=additional_data['width'], length=additional_data['length'])
                            main_data['Vth'] = Vth
                            main_data['Ith'] = Ith
                            main_data['vth_extract'] = vth_extract
                            remaining_data.remove('Vth')
                        else:
                            Vth = None
                            remaining_data.remove('Vth')
                            if 'Vth_initial' in add_data or add_data == 'all':
                                remaining_data.remove('Vth_initial')
                            if 'Vth_ref' in add_data or add_data == 'all':
                                remaining_data.remove('Vth_ref')
                    if 'I_max' in add_data or add_data == 'all':
                        I_max = np.max(Id)
                        main_data['I_max'] = I_max
                        remaining_data.remove('I_max')
                    if 'SS' in add_data or add_data == 'all':
                        if 'ss_extract' in m and 'method' in m['ss_extract']:
                            ss_extract = m['ss_extract']['method']
                            if ss_extract == 'orders_above_noise':
                                order = m['ss_extract']['order']
                            else:
                                order = None
                            SS,Vzero_current,_,_ = SS_extraction(Vg,Id,Vd,ss_extract=ss_extract,noise_level=additional_data['noise_level'],order=order)
                            main_data['SS'] = SS
                            main_data['ss_extract'] = ss_extract 
                            main_data['Vzero_current'] = Vzero_current
                            remaining_data.remove('SS')
                        else:
                            main_data['SS'] = None
                            main_data['ss_extract'] = None
                            main_data['Vzero_current'] = None
                            remaining_data.remove('SS')
                elif meas_type == 'inv-transfer':
                    Vdd = row['Vdd']
                    Vinput = np.array(row['Vinput'])
                    Voutput = np.array(row['Voutput'])
                    Isupply = np.array(row['Isupply']) if 'Isupply' in row and np.any(~np.isnan(np.asarray(row['Isupply'], dtype=float))) else None
                    time = np.array(row['t'])
                    mask = ~np.isnan(Vinput) & ~np.isnan(Voutput) 
                    Vinput = Vinput[mask]
                    Voutput = Voutput[mask]
                    time = time[mask]
                    mask = np.arange(len(Vinput)) >= 5 # remove the first 5 points
                    Vinput = Vinput[mask]
                    Voutput = Voutput[mask]
                    time = time[mask]
                    Voutput = uniform_filter1d(Voutput, size=25)# moving average filter to reduce noise
                    main_data['Vdd'] = Vdd
                    if add_data == 'all':
                        remaining_data = ['sweep_rate','transfer','Isupply','fit_invtransfer', 'dVoutdVin', 'dVoutdVin_fit','gain', 'gain_fit', 'Vm'] + list(additional_data.keys())
                    else:
                        remaining_data = add_data.copy()
                    if 'sweep_rate' in add_data or add_data == 'all':
                        total_time = time[-1] - time[0] if len(time) > 1 else 0
                        sweep_rate = (Vinput[-1] - Vinput[0]) / total_time if total_time > 0 else None
                        main_data['sweep_rate'] = sweep_rate
                        remaining_data.remove('sweep_rate')
                    if 'transfer' in add_data or add_data == 'all':
                        main_data['Vinput'] = Vinput
                        main_data['Voutput'] = Voutput
                        remaining_data.remove('transfer')
                    if 'Isupply' in add_data or add_data == 'all':
                        if Isupply is not None:
                            main_data['Isupply'] = Isupply 
                        remaining_data.remove('Isupply')
                    if 'dVoutdVin' in add_data or add_data == 'all':
                        dVoutdVin = np.abs(np.gradient(Voutput, Vinput))
                        main_data['dVoutdVin'] = dVoutdVin.tolist()
                        remaining_data.remove('dVoutdVin')
                    if 'gain' in add_data or add_data == 'all':
                        dVoutdVin = np.abs(np.gradient(Voutput, Vinput))
                        gain = np.max(dVoutdVin) if len(dVoutdVin) > 0 else None
                        main_data['gain'] = gain
                        remaining_data.remove('gain')
                    if 'Vm' in add_data or add_data == 'all':
                        Vm, _, _ = InterpolationHelper.get_abscissa(
                        Vinput, Voutput, 0.5*Vdd ,ymin=0.5*Vdd/10, ymax=0.5*Vdd*10, scale="lin-lin", interpolator_function=None)
                        main_data['Vm'] = Vm
                        remaining_data.remove('Vm')
                    if 'fit_invtransfer' in add_data or add_data == 'all':
                        if additional_data['fit_invtransfer'] is not None:
                            Vinput_fit = np.linspace(Vm-0.1, Vm+0.1, 500)
                            mask = (Vinput >= Vm-0.1) & (Vinput <= Vm+0.1)
                            Vinput_fit,Voutput_fit,_,_ = fit_data_invtransfer(Vinput[mask],Voutput[mask],Vinput_fit,additional_data['fit_invtransfer'], Vdd = Vdd, noise_level=additional_data['noise_level'])
                            main_data['Vinput_fit'] = Vinput_fit
                            main_data['Voutput_fit'] = Voutput_fit
                            main_data['fit_invtransfer'] = additional_data['fit_invtransfer']
                        remaining_data.remove('fit_invtransfer')
                    if 'dVoutdVin_fit' in add_data or add_data == 'all':
                        if additional_data['fit_invtransfer'] is not None:
                            dVoutdVin_fit = np.abs(np.gradient(Voutput_fit, Vinput_fit)) if Vinput_fit is not None and Voutput_fit is not None else None
                            main_data['dVoutdVin_fit'] = dVoutdVin_fit
                        remaining_data.remove('dVoutdVin_fit')
                    if 'gain_fit' in add_data or add_data == 'all':
                        if additional_data['fit_invtransfer'] is not None:
                            dVoutdVin_fit = np.abs(np.gradient(Voutput_fit, Vinput_fit)) if Vinput_fit is not None and Voutput_fit is not None else None
                            gain_fit = np.max(dVoutdVin_fit) if dVoutdVin_fit is not None and len(dVoutdVin_fit) > 0 else None
                            main_data['gain_fit'] = gain_fit
                        remaining_data.remove('gain_fit')
                for info in remaining_data:
                    main_data[info] = additional_data[info]
                sweep_data.append(main_data)
                    
    df_sweep = pd.DataFrame(sweep_data)
    for c in ['IDrain','VTopGate','Id_fit','Vg_fit','Vinput','Voutput','Isupply', 'Vinput_fit','Voutput_fit', 'dVoutdVin','dVoutdVin_fit']:
        if c in df_sweep.columns:
            df_sweep[c] = df_sweep[c].apply(
            lambda x: json5.dumps(x.tolist())
            if isinstance(x, np.ndarray)
            else x
        )
    if name is None:
        df_sweep.to_csv(os.path.join(script_dir,'data',dataset,f'{meas_type}_{dataset}.csv'), index=False)
    else:
        df_sweep.to_csv(os.path.join(script_dir,'data',dataset,f'{meas_type}_{dataset}_{name}.csv'), index=False)
    return df_sweep
  
def data_BTI(dataset, add_data='all', meas=None, add_data_time_fit=[], name=None):
    BTI_data = []
    additional_data = {}
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    if name is not None:
        df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_raw_data_{dataset}_{name}.csv'))
    else:
        df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_raw_data_{dataset}.csv'))
    for c in ['t','VTopGate','IDrain','ITopGate','Vinput','Voutput','Isupply','VDrain','Vsupply']:
        if c in df.columns:
            df[c] = df[c].map(safe_json_load)
    for batch,batch_meas in meas.items():
        additional_data['tox'] = float(re.sub(r'[^0-9.+-]', '', batch_meas['oxide_thickness'])) if 'oxide_thickness' in batch_meas else None
        additional_data['batch_info'] = batch_meas['batch_info'] if 'batch_info' in batch_meas else None
        for m in batch_meas['meas']['BTI'].values():
            dut = m['dut']['name']
            sample = m['sample']
            T = m['temp']
            cycles = m['cycles']
            meas_type = m['meas_type'] if 'meas_type' in m else meas_type
            additional_data['width'] = parse_length(m['dut']['width'],target_unit='um') if 'width' in m['dut'] else None
            additional_data['length'] = parse_length(m['dut']['length'],target_unit='um') if 'length' in m['dut'] else None
            additional_data['area'] = parse_length(m['dut']['width'],target_unit='um')*parse_length(m['dut']['length'],target_unit='um') if 'width' in m['dut'] and 'length' in m['dut'] else None
            additional_data['array'] = m['dut']['array'] if 'array' in m['dut'] else None
            additional_data['vacuum'] = m['vacuum'] if 'vacuum' in m else None
            additional_data['stress'] = m['stress']['type'] if 'stress' in m and 'type' in m['stress'] else None
            additional_data['acc_cycle'] = m['acc_cycle'] if 'acc_cycle' in m else None
            additional_data['ref'] = m['ref'] if 'ref' in m else None
            additional_data['fit_IdVg'] = m['fit_IdVg'] if 'fit_IdVg' in m else None
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['T'] == T) &
                    (df['sample'] == sample) &
                    (df['meas_type'] == meas_type)
                ]
            if additional_data['ref'] != 'changing':
                cycles.insert(0, cycles.pop(cycles.index(additional_data['ref'])))
            for cycle in cycles:
                subset = df_meas[
                    (df_meas['cycle'] == cycle)
                ]
                subset_sorted = subset.sort_values(
                    by='tMeas',
                    key=lambda col: (col != 'initial')
                )
                for ind,row in subset_sorted.iterrows():
                    Vgstr = row['VgStress']
                    main_data = {'meas_type':meas_type,'batch':batch,'dut':dut,'sample':sample,'temp':T,'cycle':cycle,'VgStress':Vgstr}
                    initial = True if row['tMeas'] == 'initial' else False
                    extra = True if row['tMeas'] == 'extra' else False
                    end = True if row['tMeas'] == 'end' else False
                    main_data['initial'] = initial
                    main_data['extra'] = extra
                    main_data['end'] = end
                    if extra or end:
                        additional_data['noise_level'] = m['extra_noise_level'] if 'extra_noise_level' in m else (m['noise_level'] if 'noise_level' in m else None)
                    else:
                        additional_data['noise_level'] = m['noise_level'] if 'noise_level' in m else None
                    if meas_type == 'MSM':
                        tstr = row['tStress']
                        Vgrem = row['VgRemain']
                        if not initial and not extra and not end:
                            tRec = row['tMeas']
                            main_data['tRec'] = float(tRec)
                        main_data['VgRemain'] = Vgrem
                        main_data['tStress'] = tstr
                    elif meas_type == 'OTF':
                        if not initial and not extra and not end:
                            tstr = row['tMeas']
                            main_data['tStress'] = float(tstr)
                    if 'type' not in m['dut'] or m['dut']['type'] in ['nMOS', 'pMOS']:
                        Vg = np.array(row['VTopGate'])
                        Id = np.array(row['IDrain'])
                        mask = ~np.isnan(Vg) & ~np.isnan(Id)
                        Vg = Vg[mask]
                        Id = np.abs(Id[mask])
                        Vd = row['VDrain']
                        main_data['Vd'] = Vd
                        if add_data == 'all':
                            remaining_data = ['IdVg','IdVg_fit','I','I_initial','I_ref','Vth','Vth_initial','Vth_ref','DeltaVth','DeltaI','SS','Eod_str','I_max'] + list(additional_data.keys())
                        else:
                            remaining_data = add_data.copy()
                        if 'IdVg' in add_data or add_data == 'all':
                            main_data['Id'] = Id 
                            main_data['Vg'] = Vg
                            remaining_data.remove('IdVg')
                        if 'IdVg_fit' in add_data or add_data == 'all':
                            if additional_data['fit_IdVg'] is not None:
                                Id_fit,Vg_fit,_,_ = fit_data_IdVg(Id,Vg,Id,Vg,additional_data['fit_IdVg'],noise_level=additional_data['noise_level'],temp=float(T.rstrip('K')))
                                main_data['Id_fit'] = Id_fit.tolist()
                                main_data['Vg_fit'] = Vg_fit.tolist()
                                main_data['fit_IdVg'] = additional_data['fit_IdVg']
                            remaining_data.remove('IdVg_fit')
                        if 'Vth' in add_data or add_data == 'all':
                            vth_extract = 'extra_vth_extract' if extra == True or end == True else 'vth_extract'
                            if vth_extract in m and 'method' in m[vth_extract]:
                                vth_extract_method = m[vth_extract]['method']
                                if vth_extract_method == 'constant_current' or vth_extract_method == 'constant_current_L/W':
                                    current_level = m[vth_extract]['current_level']
                                else:
                                    current_level = None
                                Vth,Ith,_,_ = Vth_extraction(Vg, Id, Vd, vth_extract_method, current_level, width=additional_data['width'], length=additional_data['length'])
                                main_data['Vth'] = Vth
                                main_data['Ith'] = Ith
                                main_data['vth_extract'] = vth_extract_method
                                remaining_data.remove('Vth')
                                if 'Vth_initial' in add_data or add_data == 'all':
                                    if initial:
                                        Vth_initial = Vth
                                    main_data['Vth_initial'] = Vth_initial
                                    remaining_data.remove('Vth_initial')
                                if 'Vth_ref' in add_data or add_data == 'all':
                                    if cycle == additional_data['ref']:
                                        Vth_ref = Vth_initial 
                                    elif additional_data['ref'] == 'changing':
                                        Vth_ref = Vth_initial
                                    main_data['Vth_ref'] = Vth_ref
                                    remaining_data.remove('Vth_ref')
                            else:
                                Vth = None
                                Vth_initial = None
                                Vth_ref = None
                                remaining_data.remove('Vth')
                                if 'Vth_initial' in add_data or add_data == 'all':
                                    remaining_data.remove('Vth_initial')
                                if 'Vth_ref' in add_data or add_data == 'all':
                                    remaining_data.remove('Vth_ref')
                        if 'I' in add_data or add_data == 'all':
                            main_data['I'] = Id[0]
                            remaining_data.remove('I')
                            if 'I_initial' in add_data or add_data == 'all':
                                if initial:
                                    I_initial = Id[0]
                                main_data['I_initial'] = I_initial
                                remaining_data.remove('I_initial')
                            if 'I_ref' in add_data or add_data == 'all':
                                if cycle == additional_data['ref']:
                                    I_ref = I_initial
                                elif additional_data['ref'] == 'changing':
                                    I_ref = I_initial
                                main_data['I_ref'] = I_ref
                                remaining_data.remove('I_ref')
                        if 'I_max' in add_data or add_data == 'all':
                            I_max = np.max(Id)
                            main_data['I_max'] = I_max
                            remaining_data.remove('I_max')
                        if 'SS' in add_data or add_data == 'all':
                            ss_extract = 'extra_ss_extract' if extra == True or end == True else 'ss_extract'
                            if ss_extract in m and 'method' in m[ss_extract]:
                                ss_extract_method = m[ss_extract]['method']
                                if ss_extract_method == 'orders_above_noise':
                                    order = m[ss_extract]['order']
                                else:
                                    order = None
                                SS,Vzero_current,_,_ = SS_extraction(Vg,Id,Vd,ss_extract=ss_extract_method,noise_level=additional_data['noise_level'],order=order)
                                main_data['SS'] = SS*1000 if SS is not None else None
                                main_data['ss_extract'] = ss_extract_method
                                main_data['Vzero_current'] = Vzero_current
                                remaining_data.remove('SS')
                            else:
                                main_data['SS'] = None
                                main_data['ss_extract'] = None
                                main_data['Vzero_current'] = None
                                remaining_data.remove('SS')
                        if 'DeltaVth' in add_data or add_data == 'all':
                            DeltaVth = Vth - Vth_ref if Vth_ref is not None and Vth is not None else None
                            main_data['DeltaVth'] = DeltaVth
                            remaining_data.remove('DeltaVth')
                        if 'DeltaI' in add_data or add_data == 'all':
                            I = Id[0]
                            DeltaI = I - I_ref
                            main_data['DeltaI'] = DeltaI
                            remaining_data.remove('DeltaI')
                        if 'Eod_str' in add_data or add_data == 'all':
                            if additional_data['stress'] == 'NBTI':
                                Eod_str = (Vgstr)/additional_data['tox']*10 # MV/cm
                            elif additional_data['stress'] == 'PBTI':
                                if Vth_initial:
                                    Eod_str = (Vgstr - Vth_initial)/additional_data['tox']*10 # MV/cm
                                else:
                                    Eod_str = None
                            main_data['Eod_str'] = Eod_str
                            remaining_data.remove('Eod_str')
                        for info in remaining_data:
                            main_data[info] = additional_data[info]
                        if add_data_time_fit:
                            main_data['fit_time'] = m['fit_time'] if 'fit_time' in m else None
                        BTI_data.append(main_data)
                    
                    elif m['dut']['type'] == 'inverter':
                        if add_data == 'all':
                            remaining_data = ['transfer', 'Isupply', 'Vm', 'Vm_initial', 'Vm_ref', 'DeltaVm'] + list(additional_data.keys())
                        else:
                            remaining_data = add_data.copy()
                        Vdd = np.mean(row['Vsupply'])
                        Vinput = np.array(row['Vinput'])
                        Voutput = np.array(row['Voutput'])
                        Isupply = np.array(row['Isupply']) if 'Isupply' in row  and np.any(~np.isnan(row['Isupply'])) else None
                        time = np.array(row['t'])
                        mask = ~np.isnan(Vinput) & ~np.isnan(Voutput) 
                        Vinput = Vinput[mask]
                        Voutput = Voutput[mask]
                        time = time 
                        main_data['Vdd'] = Vdd
                        if 'transfer' in add_data or add_data == 'all':
                            main_data['Vinput'] = Vinput
                            main_data['Voutput'] = Voutput
                            remaining_data.remove('transfer')
                        if 'Isupply' in add_data or add_data == 'all':
                            if Isupply is not None:
                                main_data['Isupply'] = Isupply[mask]
                            remaining_data.remove('Isupply')
                        if 'Vm' in add_data or add_data == 'all':
                            Vm, _, _ = InterpolationHelper.get_abscissa(
                            Vinput, Voutput, 0.5*Vdd ,ymin=0.5*Vdd/10, ymax=0.5*Vdd*10, scale="lin-lin", interpolator_function=None)
                            main_data['Vm'] = Vm
                            remaining_data.remove('Vm')
                            if 'Vm_initial' in add_data or add_data == 'all':
                                if initial:
                                    Vm_initial = Vm
                                main_data['Vm_initial'] = Vm_initial
                                remaining_data.remove('Vm_initial')
                            if 'Vm_ref' in add_data or add_data == 'all':
                                if cycle == additional_data['ref']:
                                    Vm_ref = Vm_initial 
                                elif additional_data['ref'] == 'changing':
                                    Vm_ref = Vm_initial
                                main_data['Vm_ref'] = Vm_ref
                                remaining_data.remove('Vm_ref')
                        if 'DeltaVm' in add_data or add_data == 'all':
                            DeltaVm = Vm - Vm_ref if Vm_ref is not None and Vm is not None else None
                            main_data['DeltaVm'] = DeltaVm
                            remaining_data.remove('DeltaVm')
                        for info in remaining_data:
                            main_data[info] = additional_data[info]
                        if add_data_time_fit:
                            main_data['fit_time'] = m['fit_time'] if 'fit_time' in m else None
                        BTI_data.append(main_data)
                        
    df_BTI = pd.DataFrame(BTI_data)

    for y_var in add_data_time_fit:
        group_cols = ['batch','dut','sample','temp','cycle']
        df_out = []
        for _, group in df_BTI.groupby(group_cols):
            group = group.copy()
            t_var = 'tStress' if meas_type == 'OTF' else 'tRec'
            #y_var = 'Vth' if 'Vth' in group.columns and group['Vth'].notna().any() else 'I'
            group_valid = group[group[t_var].notna()]
            time_array = group_valid[t_var].to_numpy()
            data_array = group_valid[y_var].to_numpy()

            fit_time = group['fit_time'].iloc[0]
            if fit_time and not pd.isna(fit_time) and group[y_var].notna().any():
                data_fit, fit_used = fit_data_time(time_array, data_array, time_array, fit_time)

                group.loc[group[t_var].notna(), f'{y_var}_fit'] = data_fit
                group['fit_time'] = fit_used
            else:
                group[f'{y_var}_fit'] = None
                group['fit_time'] = None
            df_out.append(group)
        df_BTI = pd.concat(df_out, ignore_index=True)

    for c in ['Id','Vg','Id_fit','Vg_fit','Vinput','Voutput','Isupply', 'Vinput_fit','Voutput_fit']:
        if c in df_BTI.columns:
            df_BTI[c] = df_BTI[c].apply(
            lambda x: json5.dumps(x.tolist())
            if isinstance(x, np.ndarray)
            else x
        )
    if name is None:
        df_BTI.to_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}.csv'), index=False)
    else:
        df_BTI.to_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}_{name}.csv'), index=False)
    return df_BTI

def data_BTI_Vth(meas,dataset,meas_type,name=None, add_info=[]): # deprecated, use data_BTI instead with add_data parameter
    Vth_data = []
    additional_info ={}
    df = pd.read_csv(os.path.join(script_dir,'data',f'{meas_type}_raw_data_{dataset}.csv'))
    for c in ['Id','Vg']:
        df[c] = df[c].map(json5.loads)
    df.loc[~df['tMeas'].isin(['initial', 'extra','end']), 'tMeas'] = df.loc[~df['tMeas'].isin(['initial', 'extra','end']),'tMeas'].astype(float)
    for batch,batch_meas in meas.items():
        info = batch_meas['batch_info'] if 'batch_info' in batch_meas else None
        tox = batch_meas['oxide_thickness'] if 'oxide_thickness' in batch_meas else None
        tox = parse_length(tox, target_unit="nm")
        additional_info['tox'] = tox
        additional_info['batch_info'] = info
        for m in batch_meas['meas'][meas_type].values():
            dut = m['dut']['name']
            sample = m['sample']
            cycles = m['cycles']
            T = m['temp']
            acc_cycle = m['acc_cycle']
            ref = m['ref']
            stress = m['stress']
            width = parse_length(m['dut']['width'],target_unit='um')
            length = parse_length(m['dut']['length'],target_unit='um')
            area = width*length
            array = m['dut']['array']
            additional_info['width'] = width
            additional_info['length'] = length
            additional_info['area'] = area
            additional_info['array'] = array
            if m['vth_extract']:
                vth_extract = m['vth_extract']['method']
                if vth_extract == 'constant_current' or 'constant_current_L/W':
                    current_level = m['vth_extract']['current_level']
                else:
                    current_level = None
            else:
                vth_extract = None
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['meas'] == meas_type) &
                    (df['T'] == T) &
                    (df['sample'] == sample)
                ]
            # cycles = np.arange(1,total_cycle+1) if not acc_cycle else np.arange(0,total_cycle+1)
            VgStress = [df_meas[(df_meas['cycle']==cycle)]['VgStress'].iloc[0] for cycle in cycles]
            if ref == 'changing':
                Vth_ref = None
            else:
                subset = df_meas[(df_meas['cycle'] == int(ref))]
                row = subset[subset['tMeas'] == 'initial'].iloc[0]
                Vg = np.array(row['Vg'])
                Id = np.array(row['Id'])
                Vd = np.array(row['Vd'])
                if vth_extract:
                    Vth,_,_,_ = Vth_extraction(Vg, Id, Vd, vth_extract, current_level, width=width, length=length)
                    Vth_initial = Vth
                    Vth_ref = Vth_initial
                    I_initial = current_level
                    I_ref = current_level
                else:   
                    Vth_ref = None
                    Vth_initial = None
                    I_initial = Id[0]
                    I_ref = I_initial
            for cycle,Vgstr in zip(cycles, VgStress):
                subset = df_meas[
                    (df_meas['cycle'] == cycle) &
                    (df_meas['VgStress'] == Vgstr)
                ]
                row = subset[subset['tMeas'] == 'initial'].iloc[0]
                Vg = np.array(row['Vg'])
                Id = np.array(row['Id'])
                Vd = np.array(row['Vd'])
                if vth_extract:
                    Vth,_,_,_ = Vth_extraction(Vg, Id, Vd, vth_extract, current_level, width=width, length=length)
                    Vth_initial = Vth
                    I_initial = current_level
                else:   
                    Vth_initial = None
                    I_initial = Id[0]
                if meas_type == 'MSM':
                    tstr = df_meas[(df_meas['cycle']==cycle)]['tStress'].iloc[0]
                    Vgrem = df_meas[(df_meas['cycle']==cycle)]['VgRemain'].iloc[0]
                if stress == 'NBTI':
                    Eod_str = (Vgstr)/tox*10 # MV/cm
                elif stress == 'PBTI':
                    if Vth_initial:
                        Eod_str = (Vgstr - Vth_initial)/tox*10 # MV/cm
                    else:
                        Eod_str = None
                additional_info['Eod_str'] = Eod_str
                if ref == 'changing':
                    Vth_ref = Vth_initial
                tMeas = subset[~df['tMeas'].isin(['initial', 'extra','end'])]['tMeas'].unique().tolist()
                for tMeas_val in sorted(tMeas):
                    row = subset[subset['tMeas'] == tMeas_val].iloc[0]
                    Vg = np.array(row['Vg'])
                    Id = np.array(row['Id'])
                    Vd = np.array(row['Vd'])
                    if vth_extract:
                        Vth,_,_,_ = Vth_extraction(Vg, Id, Vd, vth_extract, current_level,width=width, length=length)
                        I_t = current_level
                    else:   
                        Vth = None
                        I_t = Id[0]
                    DeltaVth = None if (Vth is None or Vth_initial is None) else Vth - Vth_initial
                    if meas_type == 'MSM':
                        main_data = {'meas_type':meas_type,'batch':batch,'dut':dut,'sample':sample,'temp':T,'cycle':cycle,'tStress':tstr,'VgRemain':Vgrem,'VgStress':Vgstr,'tRec':tMeas_val,'Vth_t':Vth,'Vth_initial':Vth_initial,'Vth_ref':Vth_ref,'DeltaVth':DeltaVth,'I_t':I_t,'I_initial':I_initial,'I_ref':I_ref,'vth_extract': vth_extract}
                        for info in additional_info:
                            main_data[info] = additional_info[info]
                        Vth_data.append(main_data)
                    elif meas_type == 'OTF':
                        main_data={'meas_type':meas_type,'batch':batch,'dut':dut,'sample':sample,'temp':T,'cycle':cycle,'tStress':tMeas_val,'VgStress':Vgstr,'Vth_t':Vth,'Vth_initial':Vth_initial,'Vth_ref':Vth_ref,'DeltaVth':DeltaVth,'I_t':I_t,'I_initial':I_initial,'I_ref':I_ref,'vth_extract': vth_extract}
                        for info in add_info:
                            main_data[info] = additional_info[info]
                        Vth_data.append(main_data)
    df_Vth = pd.DataFrame(Vth_data)
    if name is None:
        df_Vth.to_csv(os.path.join(script_dir,'data',f'{meas_type}_Vth_{dataset}.csv'), index=False)
    else:
        df_Vth.to_csv(os.path.join(script_dir,'data',f'{meas_type}_Vth_{dataset}_{name}.csv'), index=False)
    return df_Vth

def data_hyst(dataset, add_data='all', df=None, meas=None, add_data_freq_fit=[], cycle_fit=None, name=None):
    hyst_data = []
    additional_data = {}
    if meas == None:
        with open(f"IdVg_meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_raw_data_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_raw_data_{dataset}.csv'))
    for c in ['t','time','VTopGate','IDrain','ITopGate','Vinput','Voutput','Isupply','VDrain','Vsupply','Vg','Id']:
        if c in df.columns:
            df[c] = df[c].map(safe_json_load)
    for batch,batch_meas in meas.items():
        additional_data['tox'] = float(re.sub(r'[^0-9.+-]', '', batch_meas['oxide_thickness'])) if 'oxide_thickness' in batch_meas else None
        additional_data['batch_info'] = batch_meas['batch_info'] if 'batch_info' in batch_meas else None
        for m in batch_meas['meas']['hyst'].values():
            dut = m['dut']['name']
            sample = m['sample']
            T = m['temp']
            precondition = m['precondition']
            total_cycle = m['cycles']
            additional_data['width'] = parse_length(m['dut']['width'],target_unit='um') if 'width' in m['dut'] else None
            additional_data['length'] = parse_length(m['dut']['length'],target_unit='um') if 'length' in m['dut'] else None
            additional_data['area'] = parse_length(m['dut']['width'],target_unit='um')*parse_length(m['dut']['length'],target_unit='um') if 'width' in m['dut'] and 'length' in m['dut'] else None
            additional_data['array'] = m['dut']['array'] if 'array' in m['dut'] else None
            additional_data['vacuum'] = m['vacuum'] if 'vacuum' in m else None
            additional_data['noise_level']=m['noise_level'] if 'noise_level' in m else None
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['meas_type'].isin(['hyst','fts'])) &
                    (df['T'] == T) &
                    (df['sample'] == sample)
                ]
            cycles = np.arange(1,total_cycle+1)
            preconditions = [True, False] if precondition else [False]
            for cycle in cycles:
                df_meas_cycle = df_meas[df_meas['cycle'] == cycle]
                for precond in preconditions:
                    df_meas_hyst = df_meas_cycle[df_meas_cycle['precondition'] == precond]
                    indexes = df_meas_hyst['SweepIndex'].unique().tolist()
                    for sweep_index in indexes:
                        row = df_meas_hyst[df_meas_hyst['SweepIndex'] == sweep_index].iloc[0]
                        freq = row['freq']
                        nom_freq = row['nom_freq']
                        main_data = {'batch':batch,'dut':dut,'sample':sample,'temp':T,'cycle':cycle,'precondition':precond,'sweep_index':sweep_index, 'freq':freq,'nom_freq':nom_freq}
                        if 'type' not in m['dut'] or m['dut']['type'] in ['nMOS', 'pMOS']:
                            Vg = np.array(row['Vg'])
                            Id = np.array(row['Id'])
                            Vd = np.array(row['Vd'])
                            mask = ~np.isnan(Vg) & ~np.isnan(Id)
                            Vg = Vg[mask]
                            Id = np.abs(Id[mask])
                            main_data['Vd'] = Vd
                            if add_data == 'all':
                                remaining_data = ['IdVg','time', 'Eod', 'Ioff', 'Vminmax', 'Vth_updown', 'DeltaVth'] + list(additional_data.keys())
                            else:
                                remaining_data = add_data.copy()
                            if 'IdVg' in add_data or add_data == 'all':
                                main_data['Id'] = Id.tolist()
                                main_data['Vg'] = Vg.tolist()
                                remaining_data.remove('IdVg')
                            if 'time' in add_data or add_data == 'all':
                                time = np.array(row['time'])
                                time_corrected = np.zeros_like(time)
                                time_corrected[0] = time[0]
                                offset = 0
                                for i in range(1, len(time)):
                                    if time[i] < time[i-1]:  # reset detected
                                        # preserve step size
                                        dt_prev = time_corrected[i-1] - time_corrected[i-2] if i > 1 else 0
                                        offset += time_corrected[i-1] - time[i] + dt_prev
                                    time_corrected[i] = time[i] + offset
                                main_data['time'] = time_corrected[mask].tolist()
                                remaining_data.remove('time')
                            if 'Eod' in add_data or add_data == 'all':
                                if 'Eod' in df_meas_cycle.columns:
                                    Eod = df_meas_cycle['Eod'].iloc[0]
                                    main_data['Eod'] = Eod
                                remaining_data.remove('Eod')
                            if 'Ioff' in add_data or add_data == 'all':
                                if 'Ioff' in df_meas_cycle.columns:
                                    Ioff = df_meas_cycle['Ioff'].iloc[0]
                                    main_data['Ioff'] = Ioff
                                remaining_data.remove('Ioff')
                            if 'Vminmax' in add_data or add_data == 'all':
                                main_data['Vmin'] = np.min(Vg)
                                main_data['Vmax'] = np.max(Vg)
                                remaining_data.remove('Vminmax')
                            if 'Vth_updown' in add_data or 'DeltaVth' in add_data or add_data == 'all':
                                if m['vth_extract'] and 'method' in m['vth_extract']:
                                    vth_extract = m['vth_extract']['method']
                                    if vth_extract == 'constant_current' or vth_extract == 'constant_current_L/W':
                                        current_level = m['vth_extract']['current_level']
                                    else:
                                        current_level = None
                                    LenSweepData = len(Vg)
                                    UpDownSplit = int(np.floor(LenSweepData/2))
                                    Id_UpDown=np.array([Id[0:UpDownSplit+1],
                                            np.flip(Id[UpDownSplit:LenSweepData])])
                                    Vg_UpDown=np.array([Vg[0:UpDownSplit+1],
                                                        np.flip(Vg[UpDownSplit:LenSweepData])])
                                    Vth_UpDown = []
                                    for Data_i in range(len(Id_UpDown)):
                                        Id_Data = Id_UpDown[Data_i]
                                        Vg_Data = Vg_UpDown[Data_i]
                                        Vth,Ith,_,_ = Vth_extraction(Vg_Data, Id_Data, Vd, vth_extract, current_level)
                                        Vth_UpDown.append(Vth)
                                    DeltaVth = Vth_UpDown[1]-Vth_UpDown[0]
                                    if 'Vth_updown' in add_data or add_data == 'all':
                                        main_data['Vth_up'] = Vth_UpDown[0]
                                        main_data['Vth_down'] = Vth_UpDown[1]
                                        main_data['vth_extract'] = vth_extract
                                        main_data['Ith'] = Ith
                                        remaining_data.remove('Vth_updown')
                                    if 'DeltaVth' in add_data or add_data == 'all':
                                        main_data['DeltaVth'] = DeltaVth
                                        remaining_data.remove('DeltaVth')
                            for data in remaining_data:
                                main_data[data] = additional_data[data]
                            if add_data_freq_fit:
                                main_data['fit_freq'] = m['fit_freq'] if 'fit_freq' in m else None
                            if cycle_fit:
                                main_data['fit_cycle'] = m['fit_cycle'] if 'fit_cycle' in m else None
                            hyst_data.append(main_data)

                        elif m['dut']['type'] == 'inverter':
                            if add_data == 'all':
                                remaining_data = ['transfer', 'Isupply', 'Vm_updown', 'DeltaVm'] + list(additional_data.keys())
                            else:
                                remaining_data = add_data.copy()
                            Vdd = np.mean(row['Vsupply'])
                            Vinput = np.array(row['Vinput'])
                            Voutput = np.array(row['Voutput'])
                            Isupply = np.array(row['Isupply']) if 'Isupply' in row else None
                            time = np.array(row['t'])
                            mask = ~np.isnan(Vinput) & ~np.isnan(Voutput) 
                            Vinput = Vinput[mask]
                            Voutput = Voutput[mask]
                            time = time 
                            main_data['Vdd'] = Vdd
                            if 'transfer' in add_data or add_data == 'all':
                                main_data['Vinput'] = Vinput
                                main_data['Voutput'] = Voutput
                                remaining_data.remove('transfer')
                            if 'Isupply' in add_data or add_data == 'all':
                                if Isupply is not None:
                                    main_data['Isupply'] = Isupply[mask]
                                remaining_data.remove('Isupply')
                            if 'Vm_updown' in add_data or 'DeltaVm' in add_data or add_data == 'all':
                                LenSweepData = len(Vinput)
                                UpDownSplit = int(np.floor(LenSweepData/2))
                                Vinput_UpDown=[Vinput[0:UpDownSplit+1],
                                        np.flip(Vinput[UpDownSplit:LenSweepData])]
                                Voutput_UpDown=[Voutput[0:UpDownSplit+1],
                                        np.flip(Voutput[UpDownSplit:LenSweepData])]
                                Vm_UpDown = []
                                for Data_i in range(len(Vinput_UpDown)):
                                    Vin_Data = Vinput_UpDown[Data_i]
                                    Vout_Data = Voutput_UpDown[Data_i]
                                    Vm, _, _ = InterpolationHelper.get_abscissa(
                                    Vin_Data, Vout_Data, 0.5*Vdd ,ymin=0.5*Vdd/10, ymax=0.5*Vdd*10, scale="lin-lin", interpolator_function=None)
                                    Vm_UpDown.append(Vm)
                                DeltaVm = Vm_UpDown[1]-Vm_UpDown[0]
                                if 'Vm_updown' in add_data or add_data == 'all':
                                    main_data['Vm_up'] = Vm_UpDown[0]
                                    main_data['Vm_down'] = Vm_UpDown[1]
                                    remaining_data.remove('Vm_updown')
                                if 'DeltaVm' in add_data or add_data == 'all':
                                    main_data['DeltaVm'] = DeltaVm
                                    remaining_data.remove('DeltaVm')
                            for data in remaining_data:
                                main_data[data] = additional_data[data]
                            if add_data_freq_fit:
                                main_data['fit_freq'] = m['fit_freq'] if 'fit_freq' in m else None
                            if cycle_fit:
                                main_data['fit_cycle'] = m['fit_cycle'] if 'fit_cycle' in m else None
                            hyst_data.append(main_data)

    df_hyst = pd.DataFrame(hyst_data)

    for y_var in add_data_freq_fit:
        group_cols = ['batch','dut','sample','temp','cycle','precondition']
        df_out = []
        for _, group in df_hyst.groupby(group_cols):

            group = group.copy()
            if group['precondition'].iloc[0] == False:
                freq_array = np.array(group['freq'].unique())
                yvar_array = np.array(group[y_var].tolist())

                fit_freq = group['fit_freq'].iloc[0]
                if fit_freq:
                    yvar_fit, fit_used = fit_data_freq(freq_array, yvar_array, freq_array, fit_freq)

                    group[y_var + '_fit'] = yvar_fit
                    group['fit_freq'] = fit_used
                else:
                    group[y_var + '_fit'] = None
                    group['fit_freq'] = None

            else: 
                group[y_var + '_fit'] = None
                group['fit_freq'] = None
            df_out.append(group)
        df_hyst = pd.concat(df_out, ignore_index=True)
    
    if cycle_fit:
        group_cols = ['batch','dut','sample','temp','cycle','precondition']
        df_out = []
        for _, group in df_hyst.groupby(group_cols):

            group = group.copy()
            if group['precondition'].iloc[0] == True:
                cycle_array = np.array(group['sweep_index'].unique())
                Vth_up_array = np.array(group['Vth_up'].tolist())
                Vth_down_array = np.array(group['Vth_down'].tolist())
                DeltaVth_array = np.array(group['DeltaVth'].tolist())
                
                fit_cycle = group['fit_cycle'].iloc[0]
                if fit_cycle:
                    Vth_up_fit, fit_used = fit_data_freq(cycle_array, Vth_up_array, cycle_array, fit_cycle)
                    Vth_down_fit, fit_used = fit_data_freq(cycle_array, Vth_down_array, cycle_array, fit_cycle)
                    DeltaVth_fit,fit_used = fit_data_freq(cycle_array, DeltaVth_array, cycle_array, fit_cycle)

                    group['Vth_up_fit_cycle'] = Vth_up_fit
                    group['Vth_down_fit_cycle'] = Vth_down_fit
                    group['DeltaVth_fit_cycle'] = DeltaVth_fit
                    group['fit_cycle'] = fit_used
                else:
                    group['Vth_up_fit_cycle'] = None
                    group['Vth_down_fit_cycle'] = None
                    group['DeltaVth_fit_cycle'] = None
                    group['fit_cycle'] = None
            else: 
                group['Vth_up_fit_cycle'] = None
                group['Vth_down_fit_cycle'] = None
                group['DeltaVth_fit_cycle'] = None
                group['fit_cycle'] = None
            df_out.append(group)
        df_hyst = pd.concat(df_out, ignore_index=True)

    for c in ['Id','Vg','Id_fit','Vg_fit','Vinput','Voutput','Isupply', 'Vinput_fit','Voutput_fit']:
        if c in df_hyst.columns:
            df_hyst[c] = df_hyst[c].apply(
            lambda x: json5.dumps(x.tolist())
            if isinstance(x, np.ndarray)
            else x
        )
    if name is None:
        df_hyst.to_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}.csv'), index=False)
    else:
        df_hyst.to_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}_{name}.csv'), index=False)
    return df_hyst

def data_BTI_dDeltaVthdt(meas, dataset, meas_type, name=None):
    dDeltaVthdt_data = []
    for batch,batch_meas in meas.items():
        df_Vth = pd.read_csv(os.path.join(script_dir,'data',f'BTI_Vth_{batch}.csv'))
        for m in batch_meas['meas'].values():
            dut = m['dut']['name']
            sample = m['sample']
            T = m['temp']
            total_cycle = m['cycles']
            fit = m['fit']
            acc_cycle = m['acc_cycle']
            meas_type = m['meas_type']
            df_Vth_meas = df_Vth[
                    (df_Vth['dut'] == dut) &
                    (df_Vth['temp'] == T) &
                    (df_Vth['sample'] == sample)
                ]
            cycles = np.arange(1,total_cycle+1) if not acc_cycle else np.arange(0,total_cycle+1)
            if meas_type == 'MSM':
                t_cycles_var = 'tStress'
                t_var = 'tRec'
            elif meas_type == 'OTF':
                t_cycles_var = 'tRec'
                t_var = 'tStress'
            t_cycles = [df_Vth_meas[(df_Vth_meas['cycle']==cycle)][t_cycles_var].iloc[0] for cycle in cycles]
            VgStress = [df_Vth_meas[(df_Vth_meas['cycle']==cycle)]['VgStress'].iloc[0] for cycle in cycles]
            VgRemain = [df_Vth_meas[(df_Vth_meas['cycle']==cycle)]['VtgRemain'].iloc[0] for cycle in cycles]
            for cycle,tcycle,Vgstr,Vtgrem in zip(cycles, t_cycles, VgStress, VgRemain):
                df_Vth_t = df_Vth_meas[(df_Vth_meas['cycle'] == cycle)]
                t = np.array(df_Vth_t[t_var].tolist())
                DeltaVth = np.array(df_Vth_t['Vth_t'].tolist()) - np.array(df_Vth_t['Vth_ref'].tolist())
                if fit:
                    DeltaVth_fit,fit_used  = fit_data_time(t,DeltaVth,t,fit)
                else:
                    DeltaVth_fit = DeltaVth
                    fit_used = None
                dDeltaVthdt = np.log(10)*t*np.gradient(DeltaVth_fit,t,axis=0)
                for i,t0 in enumerate(t):
                    if meas_type == 'MSM':
                        tstr = tcycle
                        tRec = t0
                    elif meas_type == 'OTF':
                        tstr = t0
                        tRec = tcycle
                    dDeltaVthdt_data.append({'dut':dut,'sample':sample,'temp':T,'meas_type':meas_type,'cycle':cycle,'tStress':tstr,'VtgRemain':Vtgrem,'VgStress':Vgstr,'tRec':tRec,'dDeltaVthdt':dDeltaVthdt[i],'fit':fit_used})
    df_dDeltaVthdt = pd.DataFrame(dDeltaVthdt_data)
    if name is None:
        df_dDeltaVthdt.to_csv(os.path.join(script_dir,'data',f'BTI_dDeltaVthdt_{batch}.csv'), index=False)
    else:
        df_dDeltaVthdt.to_csv(os.path.join(script_dir,'data',f'BTI_dDeltaVthdt_{name}.csv'), index=False)
    return df_dDeltaVthdt

def plot_IdVg(dataset, meas = None, df=None, name = None, show_lin=False, show_log=True, show_Vth_extract=False, show_SS_extract=False,overwrite=True, title=True, normalizeW = False, normalizeLW=False, fit = True, plot_format = 'pdf', transparent = True):
    fit_labels = {
        'lambert'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    if meas == None:
        with open(f"IdVg_meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    if show_lin and show_log:
        k = 2
    elif show_lin or show_log:
        k = 1
    else:
        print('Error')
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width_lin = 1
    inter_space_width = 1.5
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['IdVg'].values()})
    total_duts = len(duts)
    n_meas = np.sum([len(batch_meas['meas']['IdVg']) for batch, batch_meas in meas.items()])
    plot_file = f'IdVg_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'IdVg')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['IdVg'].keys())[0]
            sample = meas[batch]['meas']['IdVg'][key]['sample']
            T = meas[batch]['meas']['IdVg'][key]['temp']
            plot_file += f'_{sample}_{T}'
            if name:
                plot_file += f'_{name}'
                plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    dut_meas = []
    for batch,dut in duts:
        dut_meas.append(np.sum([1 for m in meas[batch]['meas']['IdVg'].values() if m['dut']['name'] == dut]))
    max_meas = np.max(dut_meas)
    fig, axs = plt.subplots(
        total_duts, k*max_meas,
        figsize=(k*max_meas*axis_width + ylabel_space + right_additional_space + (max_meas-1)*inter_space_width + max_meas*(k-1)*inter_space_width_lin, 
                 total_duts*axis_height + xlabel_space + title_space + (total_duts-1)*inter_space_height), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(total_duts, k*max_meas)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['IdVg'].items() if m['dut']['name'] == dut}
        column = 0
        for l,m in enumerate(meas_dut.values()):
            column_2 = column +1
            left = ylabel_space + column*axis_width + l*inter_space_width + l*(k-1)*inter_space_width_lin
            bottom = title_space + (i+1)*(axis_height) + (i)*inter_space_height
            axs[i, column].set_position([left/W, 1 - bottom/H, axis_width/W, axis_height/H])
            if k==2:
                axs[i, column_2].set_position([(left + axis_width + inter_space_width_lin)/W, 1 - bottom/H, axis_width/W, axis_height/H])
            axs[i, column].sharey(axs[i, 0])
            if column > 0:
                axs[i, column].tick_params(axis='y', labelleft=False)
            if k==2:
                axs[i, column_2].sharex(axs[i, column_2])
                axs[i, column_2].sharey(axs[i, column_2])
            # axs2[i, column] = axs[i,column].twinx()
            # axs2[i, column].set_position(axs[i, column].get_position())
            # axs2[i, column].sharey(axs2[i, 0])
            # axs2[i, column].tick_params(axis='y', right=False, labelright=False)
            # else:
            #     if i < (len(duts) - 1):
            #         axs[Vth_row, column].tick_params(axis='x', labelbottom=False)
            column += 2
        for non_col in range(column, k*max_meas):
            axs[i, non_col].set_visible(False)
            if k==2:
                axs[i, non_col].set_visible(False)
    if df is None:
        df = pd.read_csv(os.path.join(script_dir,'data', dataset, f'IdVg_{dataset}.csv'))
    else:
        df = df
    for c in ['Id','Vg','Id_fit','Vg_fit']:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: np.array(json5.loads(x))
                if isinstance(x, str) and x.strip() != ''
                else x
            )
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['IdVg'].items() if m['dut']['name'] == dut}
        column = 0
        for l,m in enumerate(meas_dut.values()):
            sample = m['sample']
            column_2 = column + 1
            left = ylabel_space + column*axis_width + l*inter_space_width + l*(k-1)*inter_space_width_lin
            bottom = title_space + (i+1)*(axis_height) + (i)*inter_space_height
            if m['vth_extract']:
                vth_extract = m['vth_extract']['method']
                if vth_extract == 'constant_current' or vth_extract == 'constant_current_L/W':
                    current_level = m['vth_extract']['current_level']
                else:
                    current_level = None
            else:
                vth_extract = None
            if 'ss_extract' in m:
                ss_extract = m['ss_extract']['method']
                if ss_extract == 'orders_above_noise':
                    order = m['ss_extract']['order']
                else:
                    order = None
            else:
                ss_extract=None
            T = m['temp']
            varVd = m['varVd']
            fit_IdVg = m['fit_IdVg']
            length = parse_length(m['dut']['length'], target_unit="um")
            width = parse_length(m['dut']['width'], target_unit="um")
            noise_level = m['noise_level'] 
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['temp'] == T) &
                    (df['sample'] == sample)
                ]
            if varVd:
                Vd_array = sorted(np.unique(df_meas['Vd']))
            else:
                Vd = df_meas['Vd'].iloc[0]
                Vd_array = np.array([Vd])
            if normalizeW:
                noise_norm = noise_level/width
            elif normalizeLW:
                noise_norm = noise_level*length/width
            else:
                noise_norm = noise_level
            axs[i,column].axhline(noise_norm, linestyle='--', color = 'k')
            handles = []
            for ind,row in df_meas.iterrows():
                Id = np.abs(np.array(row['Id']))
                Vg = np.array(row['Vg'])
                Vd = row['Vd']
                if normalizeW:
                    Id_norm = Id/width
                elif normalizeLW:
                    Id_norm = Id*length/width
                else:
                    Id_norm = Id
                line, = axs[i,column].plot(Vg,Id_norm,marker='.',linestyle='None',color=plt.cm.viridis((Vd)/np.max(Vd_array)),label = Vd)
                handles.append(line)
                if k==2:
                    axs[i,column_2].plot(Vg,Id_norm,color=plt.cm.viridis((Vd)/np.max(Vd_array)))
            axs[i,column].set_yscale('log')
            xlim = axs[i,column].get_xlim()
            ylim = axs[i,column].get_ylim()
            for ind,row in df_meas.iterrows():
                Id = np.abs(np.array(row['Id']))
                Vg = np.array(row['Vg'])
                Vd = row['Vd']
                if fit_IdVg:
                    Vg_fit = np.array(row['Vg_fit'])
                    Id_fit = np.array(row['Id_fit'])
                    if normalizeW:
                        Id_fit = Id_fit/width
                    elif normalizeLW:
                        Id_fit = Id_fit*length/width
                    axs[i,column].plot(Vg_fit,Id_fit,color=plt.cm.viridis((Vd)/np.max(Vd_array)))
                if show_Vth_extract:
                    if vth_extract:
                        Vth,Ith,extract_Vth_Vg,extract_Vth_Id = Vth_extraction(Vg, Id, Vd, vth_extract, current_level, width=width, length=length)
                        if normalizeW:
                            Ith = Ith/width
                            if vth_extract in ['constant_current','linear_extrapolation']:
                                extract_Vth_Id  = extract_Vth_Id/width
                            elif vth_extract in ['constant_current_L/W']:
                                extract_Vth_Id  = extract_Vth_Id*length
                        elif normalizeLW:
                            Ith = Ith*length/width
                            if vth_extract in ['constant_current','linear_extrapolation']:
                                extract_Vth_Id  = extract_Vth_Id*length/width
                        axs[i,column].plot(extract_Vth_Vg,extract_Vth_Id,'--',color=plt.cm.viridis((Vd)/np.max(Vd_array)))
                if show_SS_extract:
                    if ss_extract:
                        SS,_,extract_SS_Vg,extract_SS_Id = SS_extraction(Vg, Id, Vd, ss_extract, noise_level, order=order, width=width, length=length)
                        if normalizeW:
                            extract_SS_Id = extract_SS_Id/width
                        elif normalizeLW:
                            extract_SS_Id = extract_SS_Id*length/width
                        axs[i,column].plot(extract_SS_Vg,extract_SS_Id,'--',color=plt.cm.viridis((Vd)/np.max(Vd_array)))
            axs[i,column].set_xlim(xlim)
            axs[i,column].set_ylim(ylim)
            if normalizeW:
                ylabel = rf'$I_D/W$ [A/$\mu m$]'
            if normalizeLW:
                ylabel = rf'$I_D \cdot L/W$ [A]'
            else:
                ylabel = rf'$I_D$ [A]'
            axs[i,column].set_ylabel(ylabel)
            #if varVd:
                #axs[i,column].legend(handles=[handles[0],handles[-1]],title=r'$V_D$')
            fig.text(
                (left + 0.5*axis_width)/W,
                1 - (bottom + 0.5*inter_space_height)/H,
                r'$V_{G}$ [V]',
                ha='center', va='center'
            )
            if k==2:
                axs[i,column_2].set_ylabel(ylabel)
                fig.text(
                    (left + inter_space_width_lin + 1.5*axis_width)/W,
                    1 - (bottom + 0.5*inter_space_height)/H,
                    r'$V_{G}$ [V]',
                    ha='center', va='center'
                )
    if title:
        if name:
            fig.text(0.5,1 - (title_space/3)/H,rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5,1 - (title_space/3)/H,rf'IdVg: {dataset}',ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_sweep(dataset, meas_type, xvar_plot, yvar_plot, show_add=None, meas = None, df=None, name = None, overwrite=True, title=True, fit = True, plot_format = 'pdf', transparent = True):
    fit_labels = {
        'lambert'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'Vinput'    : r'$V_\mathsf{in}$ [V]',
        'Voutput'   : r'$V_\mathsf{out}$ [V]',
        'Id'        : r'$I_\mathsf{D}$ [A]',
        'Vg'        : r'$V_\mathsf{G}$ [V]',
        'Vd'        : r'$V_\mathsf{D}$ [V]',
        'dVoutdVin' : r'$\left|\dfrac{dV_\mathsf{out}}{dV_\mathsf{in}}\right|$',
    }
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    k = 1
    if show_add:
        k += 1
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width_lin = 2
    inter_space_width = 1.5
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas'][meas_type].values()})
    total_duts = len(duts)
    n_meas = np.sum([len(batch_meas['meas'][meas_type]) for batch, batch_meas in meas.items()])
    plot_file = f'{meas_type}_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, meas_type)
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas'][meas_type].keys())[0]
            sample = meas[batch]['meas'][meas_type][key]['sample']
            T = meas[batch]['meas'][meas_type][key]['temp']
            plot_file += f'_{sample}_{T}'
            if name:
                plot_file += f'_{name}'
                plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    dut_meas = []
    for batch,dut in duts:
        dut_meas.append(np.sum([1 for m in meas[batch]['meas'][meas_type].values() if m['dut']['name'] == dut]))
    max_meas = np.max(dut_meas)
    fig, axs = plt.subplots(
        total_duts, k*max_meas,
        figsize=(k*max_meas*axis_width + ylabel_space + right_additional_space + (max_meas-1)*inter_space_width + max_meas*(k-1)*inter_space_width_lin, 
                 total_duts*axis_height + xlabel_space + title_space + (total_duts-1)*inter_space_height), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(total_duts, k*max_meas)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas'][meas_type].items() if m['dut']['name'] == dut}
        column = 0
        for l,m in enumerate(meas_dut.values()):
            column_2 = column +1
            left = ylabel_space + column*axis_width + l*inter_space_width + l*(k-1)*inter_space_width_lin
            bottom = title_space + (i+1)*(axis_height) + (i)*inter_space_height
            axs[i, column].set_position([left/W, 1 - bottom/H, axis_width/W, axis_height/H])
            if k==2:
                axs[i, column_2].set_position([(left + axis_width + inter_space_width_lin)/W, 1 - bottom/H, axis_width/W, axis_height/H])
            axs[i, column].sharey(axs[i, 0])
            if column > 0:
                axs[i, column].tick_params(axis='y', labelleft=False)
            if k==2:
                axs[i, column_2].sharex(axs[i, column_2])
                axs[i, column_2].sharey(axs[i, column_2])
            # axs2[i, column] = axs[i,column].twinx()
            # axs2[i, column].set_position(axs[i, column].get_position())
            # axs2[i, column].sharey(axs2[i, 0])
            # axs2[i, column].tick_params(axis='y', right=False, labelright=False)
            # else:
            #     if i < (len(duts) - 1):
            #         axs[Vth_row, column].tick_params(axis='x', labelbottom=False)
            column += k
        for non_col in range(column, k*max_meas):
            axs[i, non_col].set_visible(False)
            if k==2:
                axs[i, non_col].set_visible(False)
    if df is None:
        df = pd.read_csv(os.path.join(script_dir,'data', dataset, f'{meas_type}_{dataset}.csv'))
    else:
        df = df
    for c in ['Id','Vg','Id_fit','Vg_fit','Vd','Vinput','Voutput','Vinput_fit','Voutput_fit', 'dVoutdVin', 'dVoutdVin_fit']:
        if c in df.columns:
            df[c] = df[c].map(safe_json_load)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas'][meas_type].items() if m['dut']['name'] == dut}
        column = 0
        for l,m in enumerate(meas_dut.values()):
            sample = m['sample']
            column_2 = column + 1
            left = ylabel_space + column*axis_width + l*inter_space_width + l*(k-1)*inter_space_width_lin
            bottom = title_space + (i+1)*(axis_height) + (i)*inter_space_height
            T = m['temp']
            fit_invtransfer = m['fit_invtransfer'] if 'fit_invtransfer' in m else None
            length = parse_length(m['dut']['length'], target_unit="um")
            width = parse_length(m['dut']['width'], target_unit="um")
            noise_level = m['noise_level'] if 'noise_level' in m else None
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['temp'] == T) &
                    (df['meas_type'] == meas_type) &
                    (df['sample'] == sample)
                ]
            meas_text = f'{dut} {sample} {T}'
            meas_text += f'\n S. R. = {df_meas['sweep_rate'].iloc[0]:.2f} V/s'
            axs[i,column].text(0.95,0.95,meas_text, transform=axs[i, column].transAxes, horizontalalignment='right', verticalalignment='center')
            #axs[i,column].axhline(noise_level, linestyle='--', color = 'k')
            handles = []
            sweeps = df_meas['sweep'].values
            for ind,row in df_meas.sort_values('sweep').iterrows():
                sweep = row['sweep']
                xvar = np.array(row[xvar_plot])
                yvar = np.array(row[yvar_plot])
                line, = axs[i,column].plot(xvar, yvar, marker='.', linestyle='None', markeredgewidth=1, color=plt.cm.viridis((sweep)/np.max(sweeps)), label = sweep)
                handles.append(line)
                if k==2:
                    yvar2 = np.array(row[show_add])
                    axs[i,column_2].plot(xvar, yvar2, marker='.', linestyle='None', markeredgewidth=1, color=plt.cm.viridis((sweep)/np.max(sweeps)))
            #axs[i,column].set_yscale('log')
            xlim = axs[i,column].get_xlim()
            ylim = axs[i,column].get_ylim()
            xlim_2  = [row['Vm'] - 0.35, row['Vm'] + 0.15]
            #ylim_2 = [row['gain']*0.1, row['gain']*10]
            if fit:
                for ind,row in df_meas.sort_values('sweep').iterrows():
                    sweep = row['sweep']
                    if fit_invtransfer:
                        xvar_fit = np.array(row[f'{xvar_plot}_fit'])
                        yvar_fit = np.array(row[f'{yvar_plot}_fit']) if row[f'{yvar_plot}_fit'] is not None else np.full(len(xvar_fit), np.nan)
                        axs[i,column].plot(xvar_fit, yvar_fit, color=plt.cm.viridis((sweep)/np.max(sweeps)), alpha=0.7)
                        if k==2 and show_add + '_fit' in row:
                            yvar2_fit = np.array(row[show_add + '_fit']) if row[f'{show_add}_fit'] is not None else np.full(len(xvar_fit), np.nan)
                            axs[i,column_2].plot(xvar_fit, yvar2_fit, color=plt.cm.viridis((sweep)/np.max(sweeps)), alpha=0.7)
            else:
                for ind,row in df_meas.sort_values('sweep').iterrows():
                    sweep = row['sweep']
                    xvar = np.array(row[xvar_plot])
                    yvar = np.array(row[yvar_plot])
                    axs[i,column].plot(xvar, yvar, color=plt.cm.viridis((sweep)/np.max(sweeps)))
                    if k==2:
                        yvar2 = np.array(row[show_add])
                        axs[i,column_2].plot(xvar, yvar2, color=plt.cm.viridis((sweep)/np.max(sweeps)))

            axs[i,column].set_xlim(xlim)
            axs[i,column].set_ylim(ylim)
            axs[i,column].set_ylabel(var_labels[yvar_plot])
            axs[i,column].set_xlabel(var_labels[xvar_plot])
            if k==2:
                axs[i,column_2].set_xlim(xlim_2)
                #axs[i,column_2].set_ylim(ylim_2)
                axs[i,column_2].set_ylabel(var_labels[show_add])
                axs[i,column_2].set_xlabel(var_labels[xvar_plot])
            column += k
    if title:
        if name:
            fig.text(0.5,1 - (title_space/3)/H,rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5,1 - (title_space/3)/H,rf'{meas_type}: {dataset}',ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_IdVg_duts(dataset,df=None, grouping=None, showing=None, multiple=None, filter_plot={}, legend = ['dut','sample'], texts=None, name=None, fit=False, across=None, show_add=False, overwrite=True, plot_format='pdf', transparent=True, title=True):     
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'tStress'        : r'$t_{str}$ [s]',
        'VgStress'      : r'$V_{G,str}$ [V]',
        'VgRemain'      : r'$V_{G,rec}$ [V]',
        'tRec'           : r'$t_{rec}$ [s]',
        'Vd'            :r'$V_D$ [V]',
        'temp'          : r'T',
        'Eod_str'          : r'$E_{od,str}$ [MV/cm]',
        'Vth_initial' : r'$V_{th,initial}$ [V]',
    }
    if show_add:
        k = 2
    else:
        k = 1
    plot_file = f'IdVg_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'IdVg')
    if grouping:
        plot_file += '_grouping'
        for grouping_var in grouping:
            plot_file += f'_{grouping_var}'
            if grouping_var in filter_plot:
                plot_file += f'_{filter_plot[grouping_var]}'
    if showing:
        plot_file += '_showing'
        for showing_var in showing:
            plot_file += f'_{showing_var}'
            if showing_var in filter_plot:
                plot_file += f'_{filter_plot[showing_var]}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    if df is None:
        if name is None:
            df_IdVg_meas = pd.read_csv(os.path.join(script_dir,'data', dataset, f'IdVg_{dataset}.csv'))
        else:
            df_IdVg_meas = pd.read_csv(os.path.join(script_dir,'data', dataset, f'IdVg_{dataset}_{name}.csv'))
    else:
        df_IdVg_meas = df
    for c in ['Id','Vg','Id_fit','Vg_fit']:
        if c in df_IdVg_meas.columns:
            df_IdVg_meas[c] = df_IdVg_meas[c].apply(
                lambda x: np.array(json5.loads(x))
                if isinstance(x, str) and x.strip() != ''
                else x
            )
    for variable,plot in filter_plot.items():
        df_IdVg_meas = df_IdVg_meas[df_IdVg_meas[variable].isin(plot)]
        if fit:
            df_IdVgfit = df_IdVgfit[df_IdVgfit[variable].isin(plot)]
    df_IdVg_subset = df_IdVg_meas
    if fit:
        df_IdVgfit_subset = df_IdVgfit
    groups = (df_IdVg_subset.groupby(grouping) if grouping else [(None, df_IdVg_subset)])
    if across:
        n_rows = sum(subset.groupby(across).ngroups > 1
        for _, subset in groups)
        df_IdVg_subset = df_IdVg_subset.groupby(grouping).filter(
                lambda subset: subset.groupby(across).ngroups > 1
            )
        if fit:
            df_IdVgfit_subset = df_IdVgfit_subset.groupby(grouping).filter(
            lambda subset: subset.groupby(across).ngroups > 1
            )
    else:
        n_rows = df_IdVg_subset.groupby(grouping).ngroups if grouping else 1
    if n_rows == 0:
        print(f'No sufficient data to plot IdVgs for multiple {across} under the chosen conditions.')
        return
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width = 2.5
    fig, axs = plt.subplots(
        n_rows, k,
        figsize=(k*axis_width + ylabel_space + right_additional_space + (k-1)*inter_space_width, 
                 axis_height*n_rows + xlabel_space + title_space + (n_rows-1)*inter_space_height), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    plt.subplots_adjust(wspace=inter_space_width/axis_width, hspace=inter_space_height/axis_height)
    axs = np.array(axs).reshape(n_rows, k)
    for i in range(n_rows):
        axs[i,0].sharex(axs[0,0])
        axs[i,0].sharey(axs[0,0])
        # if i != n_rows-1:
        #     axs[i,0].tick_params(axis='x', labelbottom=False)
        #if show_add:
            #axs[i,k-1].sharex(axs[0,k-1])
            #axs[i,k-1].sharey(axs[0,k-1])
            # if i != n_rows-1:
            #     axs[i,k-1].tick_params(axis='x', labelbottom=False)
    markers = ['o','v','^','<','>','s','p','*','h','H','+','x','D','d','|','_','1','2','3','4','.',',']
    all_samples = df_IdVg_subset[['batch','dut','sample']].drop_duplicates()
    all_samples = [tuple(x) for x in all_samples.values]
    global_marker_map = {
        combo: markers[i % len(markers)]
        for i, combo in enumerate(all_samples)
    }
    if showing:
        all_showed = df_IdVg_subset[showing].drop_duplicates()#.sort_values(by=showing[0])
        if len(showing)==1:
            all_showed = all_showed.sort_values(by=showing[0])
        all_showed = list(map(tuple, all_showed.to_numpy()))
        colors = plt.cm.viridis(np.linspace(0.1,0.9,len(all_showed)))
        global_color_map = {
            combo: colors[i % len(colors)]
            for i, combo in enumerate(all_showed)
        }
    else:
        colors = plt.cm.tab10(np.linspace(0,1,10))
    edge_colors = colors
    i=0
    marker_handles = []
    group_map = {}
    for group_elements, subset in groups:
        group_text = ''
        if grouping:
            for j,elem in enumerate(grouping):
                if elem in var_labels.keys():
                    group_text += var_labels[elem] + f'= {subset[elem].iloc[0]}; '
                else:
                    group_text += f'{subset[elem].iloc[0]}'
        axs[i,0].text(0.05, 0.95, group_text,
                horizontalalignment='left', verticalalignment='center', transform=axs[i,0].transAxes, color ='k')
        # fig.text((ylabel_space+(k*(axis_width)+(k-1)*inter_space_width)*0.5)/W,1-(title_space + i*(axis_height+inter_space_height))/H,group_text, 
        #         horizontalalignment='center', verticalalignment='bottom', color ='k')
        sublabel_handles = []
        groups_showing = [(None, subset)] if showing is None else subset.groupby(showing)
        for group_showing, subset_showing in groups_showing:
            for (batch,dut,sample), subset_sample in subset_showing.groupby(['batch','dut','sample']):
                marker = global_marker_map[(batch, dut, sample)]
                label = ''
                for j,elem in enumerate(legend):
                    if elem in var_labels.keys():
                        label += var_labels[elem] + f'= {subset_sample[elem].iloc[0]} '
                    else:
                        label += f'{subset_sample[elem].iloc[0]} '
                if (showing is not None):
                    color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
                else:
                    color = colors[i%len(colors)]
                for index,row in subset_sample.iterrows():
                    Vg = np.array(row['Vg'])
                    Id = np.array(row['Id'])
                    axs[i,0].plot(Vg, Id, marker=marker, color = color, alpha=1)
                    if show_add == 'Id/W':
                        width = row['width']
                        axs[i,k-1].plot(Vg, Id/width, marker=marker, color = color, alpha=1)
                    elif show_add == 'Id_L/W':
                        width = row['width']
                        length = row['length']
                        axs[i,k-1].plot(Vg, Id*length/width, marker=marker, color = color, alpha=1)
                    if texts:
                        sublabel = ''
                        for j,elem in enumerate(texts):
                            if elem in var_labels.keys():
                                sublabel += var_labels[elem] + f'= {subset_sample[elem].iloc[0]:.2f}'
                            else:
                                sublabel += f'{subset_sample[elem].iloc[0]}'
                        sublabel_handles.append(Line2D([], [],marker=marker,markerfacecolor=color,markeredgecolor=color,label=sublabel))
                        #axs[i,0].text(var_plot[len(var_plot) // 2], DeltaVth[len(var_plot) // 2], text, horizontalalignment='center', 
                        #    verticalalignment='top', color= color)
                if not any(h.get_label() == label for h in marker_handles):
                    marker_handles.append(Line2D([], [], marker=marker, markerfacecolor=color, markeredgecolor=color, alpha=1,label=label))
        if legend:
            leg = axs[i,k-1].legend(handles=sublabel_handles, ncol=1,loc = 'best', handlelength=0.5, framealpha=0.1, fontsize=18)
            axs[i,k-1].add_artist(leg) 
        axs[i,0].set_ylabel(rf'$I_D$ [A]')
        #axs[i,0].sharey(axs[0, 0])
        axs[i,0].set_yscale('log')
        #axs[i,0].yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axs[i,k-1].set_yscale('log')
        if show_add == 'Id/W':
            axs[i,k-1].set_ylabel(rf'$I_D/W$ [A/$\mu$m]')
            #axs[i,k-1].sharey(axs[0,k-1])
            #axs[i,k-1].yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        elif show_add == 'Id_L/W':
            axs[i,k-1].set_ylabel(rf'$I_D \cdot L/W$ [A]')
        axs[i,0].set_xlabel(rf'$V_G$ [V]')
        if show_add:
            axs[i,k-1].set_xlabel(rf'$V_G$ [V]')
        group_map[group_elements] =i
        i+=1
    if fit:
        for group_elements, subset_fit in df_IdVgfit_subset.groupby(grouping):
            i = group_map[group_elements]
            for group_showing, subset_showing in subset_fit.groupby(showing):
                for group_sample, subset_sample in subset_showing.groupby(['batch','dut','sample']):
                    Vg_fit = subset_sample['Vg_fit'].to_numpy()
                    Id_fit = subset_sample['Id_fit'].to_numpy()
                    fit_type = subset_sample['fit'].iloc[0]
                    color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
                    axs[i,0].plot(Vg_fit, Id_fit, linestyle='-', color=color)
                    if show_add == 'Id/W':
                        W = subset_sample['width'].to_numpy()
                        axs[i,k-1].plot(Vg_fit, Id_fit, linestyle='-', color=color)
                    if texts == 'fit':
                        axs[i,0].text(Vg_fit[len(Vg_fit) // 2],Id_fit[len(Vg_fit) // 2], f'{fit_labels[fit_type]}', horizontalalignment='center', 
                                verticalalignment='top', color= color)
    
    # Combine both legends
    all_handles = marker_handles
    axs[0,k-1].legend(handles=all_handles, ncol=1,loc = 'upper left',bbox_to_anchor=(1.05, 1), handlelength=1.0, framealpha=1)
    fig.text(0.5, 1 - (title_space/2)/H, f'{name}: {dataset}' if name else f'{dataset}', ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()
    return

def plot_IdVg_Vth_vs_var_duts(dataset, varying, df=None,grouping=None, showing=None, multiple=None, filter_plot={}, legend = ['dut','sample'], texts=None, name=None, fit=False, across=None, mul_markers=True, title=True, show_add=False, transparent=True, plot_format='pdf',overwrite=True):     
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'tStress'        : r'$t_{str}$ [s]',
        'VgStress'      : r'$V_{G,str}$ [V]',
        'VgRemain'      : r'$V_{G,rec}$ [V]',
        'tRec'           : r'$t_{rec}$ [s]',
        'Vd'            :r'$V_D$ [V]',
        'temp'          : r'T',
        'Eod_str'          : r'$E_{od,str}$ [MV/cm]',
        'Vth_initial' : r'$V_{th,initial}$ [V]',
        'width' : r'W [$\mu m$]',
        'area': r'A [$\mu m^2$]'
    }
    if show_add:
        k = 2
    else:
        k = 1
    plot_file = f'IdVg_Vth_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'IdVg')
    if grouping:
        plot_file += '_grouping'
        for grouping_var in grouping:
            plot_file += f'_{grouping_var}'
            if grouping_var in filter_plot:
                plot_file += f'_{filter_plot[grouping_var]}'
    if showing:
        plot_file += '_showing'
        for showing_var in showing:
            plot_file += f'_{showing_var}'
            if showing_var in filter_plot:
                plot_file += f'_{filter_plot[showing_var]}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    if df is None:
        if name is None:
            df_Vth_meas = pd.read_csv(os.path.join(script_dir,'data', dataset, f'IdVg_{dataset}.csv'))
        else:
            df_Vth_meas = pd.read_csv(os.path.join(script_dir,'data', dataset, f'IdVg_{dataset}_{name}.csv'))
    else:
        df_Vth_meas = df
    for variable,plot in filter_plot.items():
        df_Vth_meas = df_Vth_meas[df_Vth_meas[variable].isin(plot)]
    df_Vth_subset = df_Vth_meas
    groups = (df_Vth_subset.groupby(grouping) if grouping else [(None, df_Vth_subset)])
    if across:
        n_rows = sum(subset.groupby(across).ngroups > 1
        for _, subset in groups)
        df_Vth_subset = df_Vth_subset.groupby(grouping).filter(
                lambda subset: subset.groupby(across).ngroups > 1
            )
    else:
        n_rows = df_Vth_subset.groupby(grouping).ngroups if grouping else 1
    if n_rows == 0:
        print(f'No sufficient data to plot Vth for multiple {across} under the chosen conditions.')
        return
    ylabel_space = 1.6
    if legend:
        right_additional_space = 4
    else:
        right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.35
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width = 1.5
    fig, axs = plt.subplots(
        n_rows, k,
        figsize=(k*axis_width + ylabel_space + right_additional_space + (k-1)*inter_space_width, 
                 axis_height*n_rows + xlabel_space + title_space + (n_rows-1)*inter_space_height), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    plt.subplots_adjust(wspace=inter_space_width/axis_width, hspace=inter_space_height/axis_height)
    axs = np.array(axs).reshape(n_rows, k)
    for i in range(n_rows):
        axs[i,0].sharex(axs[0,0])
        axs[i,0].sharey(axs[0,0])
        # if i != n_rows-1:
        #     axs[i,0].tick_params(axis='x', labelbottom=False)
        #if show_add:
            #axs[i,k-1].sharex(axs[0,k-1])
            #axs[i,k-1].sharey(axs[0,k-1])
            # if i != n_rows-1:
            #     axs[i,k-1].tick_params(axis='x', labelbottom=False)
    if mul_markers:
        markers = ['o','v','^','<','>','s','p','*','h','H','+','x','D','d','|','_','1','2','3','4','.',',']
    else:
        markers = ['s']
    all_samples = df_Vth_subset[['batch','dut','sample']].drop_duplicates()
    all_samples = [tuple(x) for x in all_samples.values]
    global_marker_map = {
        combo: markers[i % len(markers)]
        for i, combo in enumerate(all_samples)
    }
    if showing:
        all_showed = df_Vth_subset[showing].drop_duplicates()#.sort_values(by=showing[0])
        if len(showing)==1:
            all_showed = all_showed.sort_values(by=showing[0])
        all_showed = list(map(tuple, all_showed.to_numpy()))
        colors = plt.cm.viridis(np.linspace(0.1,0.9,len(all_showed)))
        global_color_map = {
            combo: colors[i % len(colors)]
            for i, combo in enumerate(all_showed)
        }
    else:
        colors = plt.cm.tab10(np.linspace(0,1,10))
    i=0
    marker_handles = []
    group_map = {}
    for group_elements, subset in groups:
        group_text = ''
        if grouping:
            for j,elem in enumerate(grouping):
                if elem in var_labels.keys():
                    group_text += var_labels[elem] + f'= {subset[elem].iloc[0]}; '
                else:
                    group_text += f'{subset[elem].iloc[0]}'
        # axs[i,0].text(0.5,0.2, group_text,
        #         horizontalalignment='center', verticalalignment='center', transform=axs[i,0].transAxes, color ='k')
        if len(groups)>1:
            fig.text((ylabel_space+(k*(axis_width)+(k-1)*inter_space_width)*0.5)/W,1-(title_space + i*(axis_height+inter_space_height))/H,group_text, 
                    horizontalalignment='center', verticalalignment='bottom', color ='k')
        sublabel_handles = []
        groups_showing = [(None, subset)] if showing is None else subset.groupby(showing)
        for group_showing, subset_showing in groups_showing:
            for (batch,dut,sample), subset_sample in subset_showing.groupby(['batch','dut','sample']):
                marker = global_marker_map[(batch, dut, sample)]
                label = ''
                for j,elem in enumerate(legend):
                    if elem in var_labels.keys():
                        label += var_labels[elem] + f'= {subset_sample[elem].iloc[0]} '
                    else:
                        label += f'{subset_sample[elem].iloc[0]} '
                if (showing is not None):
                    color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
                else:
                    color = colors[i%len(colors)]
                for index,row in subset_sample.iterrows():
                    var_plot = np.array(row[varying])
                    Vth = np.array(row['Vth'])
                    axs[i,0].plot(var_plot, Vth, marker=marker, color = color, alpha=1)
                    if show_add == 'Id/W':
                        width = row['width']
                        axs[i,k-1].plot(var_plot, Vth/width, marker=marker, color = color, alpha=1)
                    if texts:
                        sublabel = ''
                        for j,elem in enumerate(texts):
                            if elem in var_labels.keys():
                                sublabel += var_labels[elem] + f'= {subset_sample[elem].iloc[0]:.2f}'
                            else:
                                sublabel += f'{subset_sample[elem].iloc[0]}'
                        sublabel_handles.append(Line2D([], [],marker=marker,linestyle='None',markerfacecolor=color,label=sublabel))
                        #axs[i,0].text(var_plot[len(var_plot) // 2], DeltaVth[len(var_plot) // 2], text, horizontalalignment='center', 
                        #    verticalalignment='top', color= color)
                if not any(h.get_label() == label for h in marker_handles):
                    marker_handles.append(Line2D([], [], marker=marker, linestyle='None', markerfacecolor=color, alpha=1,label=label))
        if texts:
            leg = axs[i,k-1].legend(handles=sublabel_handles, ncol=1,loc = 'best', handlelength=0.5, framealpha=0.1, fontsize=18)
            axs[i,k-1].add_artist(leg) 
        axs[i,0].set_ylabel(r'$V_{th}$ [V]')
        #axs[i,0].sharey(axs[0, 0])
        #axs[i,0].set_yscale('log')
        #axs[i,0].yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        if show_add == 'Id/W':
            axs[i,k-1].set_ylabel(rf'$I_D/W$ [A/$\mu$m]')
            #ymin, ymax = axs[i,k-1].get_ylim()
            #axs[i,k-1].set_ylim(ymin, ymax)
            axs[i,k-1].axhline(0, linestyle='--', color = 'k')
            #axs[i,k-1].sharey(axs[0,k-1])
            axs[i,k-1].set_yscale('log')
            #axs[i,k-1].yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axs[i,0].set_xlabel(var_labels[varying])
        if show_add:
            axs[i,k-1].set_xlabel(var_labels[varying])
        group_map[group_elements] =i
        i+=1
    if fit:
        for group_elements, subset_fit in df_Vth_subset.groupby(grouping):
            i = group_map[group_elements]
            for group_showing, subset_showing in subset_fit.groupby(showing):
                for group_sample, subset_sample in subset_showing.groupby(['batch','dut','sample']):
                    Vg_fit = subset_sample['Vg_fit'].to_numpy()
                    Id_fit = subset_sample['Id_fit'].to_numpy()
                    fit_type = subset_sample['fit'].iloc[0]
                    color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
                    axs[i,0].plot(Vg_fit, Id_fit, linestyle='-', color=color)
                    if show_add == 'Id/W':
                        W = subset_sample['width'].to_numpy()
                        axs[i,k-1].plot(Vg_fit, Id_fit, linestyle='-', color=color)
                    if texts == 'fit':
                        axs[i,0].text(Vg_fit[len(Vg_fit) // 2],Id_fit[len(Vg_fit) // 2], f'{fit_labels[fit_type]}', horizontalalignment='center', 
                                verticalalignment='top', color= color)
    
    # Combine both legends
    all_handles = marker_handles
    if legend:
        axs[0,k-1].legend(handles=all_handles, ncol=1,loc = 'upper left',bbox_to_anchor=(1.05, 1), handlelength=1.0, framealpha=1)
    if title:
        fig.text(0.5, 1 - (title_space/2)/H, f'{name}: {dataset}' if name else f'{dataset}', ha='center', va='center')
    if name:
        folder_path = os.path.join(script_dir,'plots', dataset, name)
    else:
        folder_path = os.path.join(script_dir,'plots', dataset, 'IdVg')
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()
    return

def plot_BTI_DeltaVth(dataset, meas=None, df=None, t_plot='all', name = False, vs_cyclevar=False, overwrite=True, plot_format='pdf', transparent=True, fit=True, title=True):
    k = 1
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'tStress'        : r'$t_{str}$ [s]',
        'VgStress'      : r'$V_{G,str}$ [V]',
        'VgRemain'      : r'$V_{G,rec}$ [V]',
        'tRec'           : r'$t_{rec}$ [s]',
        'temp'          : r'T',
        'Eox_str'          : r'$E_{ox,str}$ [MV/cm]',
        'Vth_initial' : r'$V_{th,initial}$ [V]',
    }
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['BTI'].values()})
    total_duts = len(duts)
    n_meas = np.sum([len(batch_meas['meas']['BTI']) for _, batch_meas in meas.items()])
    plot_file = f'BTI_DeltaVth_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'BTI')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['BTI'].keys())[0]
            sample = meas[batch]['meas']['BTI'][key]['sample']
            T = meas[batch]['meas']['BTI'][key]['temp']
            plot_file += f'_{sample}_{T}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    if vs_cyclevar:
        k = k + 1
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width = 0.75
    fig, axs = plt.subplots(n_meas,k,
    figsize=(k*axis_width + ylabel_space + right_additional_space + (k-1)*inter_space_width, 
                 n_meas*axis_height + xlabel_space + title_space + (n_meas-1)*inter_space_height),
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    plt.subplots_adjust(wspace=inter_space_width/axis_width, hspace=inter_space_height/axis_height)
    axs = np.array(axs).reshape(n_meas, k)
    column_rec = 0
    column_str = 1
    for row in range(n_meas):
        axs[row, column_rec].sharex(axs[0, column_rec])
        #axs[row, column_rec].sharey(axs[0, column_rec])
        if vs_cyclevar:
            axs[row, column_str].sharey(axs[row, column_rec])
            axs[row, column_str].tick_params(axis='y', labelleft=False)
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}.csv'))
    row = 0
    for batch,batch_meas in meas.items():
        for m in batch_meas['meas']['BTI'].values():
            dut = m['dut']['name']
            sample = m['sample']
            T = m['temp']
            meas_type = m['meas_type']
            varying = m['varying']
            cycles = m['cycles']
            acc_cycle = m['acc_cycle']
            if 'vth_extract' in m and 'method' in m['vth_extract']:
                y_var = 'Vth'
                y_var_fit = 'Vth_fit'
                y_var_ref = 'Vth_ref'
                ylabel = r'$\Delta V_\mathsf{th}$ [V]'
            else:
                y_var = 'I'
                y_var_ref = 'I_ref'
                y_var_fit = 'I_fit'
                ylabel = r'$\Delta I_\mathsf{D}$ [A]'
            stress = m['stress']['type']
            df_meas = df[
                    (df['dut'] == dut) &
                    (df['temp'] == T) &
                    (df['batch'] == batch) &
                    (df['meas_type'] == meas_type) &
                    (df['sample'] == sample)
                ]
            if meas_type == 'MSM':
                t_var = 'tRec'
                t_label = r'$t_{rec}$ [s]'
            elif meas_type == 'OTF':
                t_var = 'tStress'
                t_label = r'$t_{str}$ [s]'
            xlabel = var_labels[varying]
            Stress_arr = [df_meas[(df_meas['cycle']==cycle)][varying].iloc[0] for cycle in sorted(cycles)]
            if acc_cycle:
                max_Stress = max([stress_val for stress_val in Stress_arr if stress_val != 0])
                min_Stress = min([stress_val for stress_val in Stress_arr if stress_val != 0])
            else:
                max_Stress = max(Stress_arr)
                min_Stress = min(Stress_arr)
            if meas_type == 'MSM':
                non_var = ['VgStress','VgRemain','tStress'].copy()
                non_var.remove(varying)
            elif meas_type == 'OTF':
                non_var = []
            for cycle, stress_val in zip(cycles,Stress_arr):
                df_cycle = df_meas[(df_meas['cycle'] == cycle)]
                df_cycle = df_cycle[(df_cycle[t_var] != 0.0) & (df_cycle[t_var].notna())]
                t = df_cycle[t_var].to_numpy(dtype=float)
                Delta = df_cycle[y_var].to_numpy(dtype=float) - df_cycle[y_var_ref].to_numpy(dtype=float)
                idx = np.argsort(t)
                t = t[idx]
                Delta = Delta[idx]
                if acc_cycle and cycle == 0:
                    pass
                else:
                    axs[row,column_rec].plot(t, Delta, marker='o', linestyle='None', 
                        color = plt.cm.viridis(np.log10(stress_val)/np.log10(max_Stress)) if varying=='tStress' else plt.cm.viridis((stress_val-min_Stress)/(max_Stress-min_Stress)))
                    if fit and f'{y_var}_fit' in df_cycle.columns:
                        Delta_fit = df_cycle[f'{y_var}_fit'].to_numpy(dtype=float) - df_cycle[y_var_ref].to_numpy(dtype=float)
                        Delta_fit = Delta_fit[idx]
                    else:
                        Delta_fit = Delta
                    axs[row,column_rec].plot(t, Delta_fit, linestyle='-', 
                        color = plt.cm.viridis(np.log10(stress_val)/np.log10(max_Stress)) if varying=='tStress' else plt.cm.viridis((stress_val-min_Stress)/(max_Stress-min_Stress)))
            t = df_meas[(df_meas[t_var] != 0.0) & (df_meas[t_var].notna())][t_var].unique().tolist()
            min_t = min(t)
            max_t = max(t)
            axs[row,column_rec].axhline(0, linestyle='--', color = 'k')
            axs[row,column_rec].set_xscale('log')
            axs[row,column_rec].set_xlabel(t_label)
            axs[row,column_rec].xaxis.set_major_locator(plt.LogLocator(base=10, numticks=100))
            axs[row,column_rec].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            t_axis = np.logspace(np.log10(min_t), np.log10(max_t),500)
            t_colors = plt.cm.plasma(np.log10(t_axis) / np.log10(max(t_axis)))
            axs[row,column_rec].spines['bottom'].set_visible(False)
            points = np.array([np.linspace(0, 1, 500), np.full(500, 0)]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                linewidth=10,
                colors=t_colors[:-1],   # one color per segment
                transform=axs[row,column_rec].transAxes     # IMPORTANT: draw in axis coordinates
            )
            label = f'{dut} {sample} {T}'
            for l,elem in enumerate(non_var):
                if elem == 'tStress':
                    label += '\n' + rf' $t_{{str}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} s'
                elif elem == 'VgStress':
                    label += '\n' + rf' $V_{{TG,str}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} V'
                elif elem == 'VtgRemain':
                    label += '\n' + rf' $V_{{TG,rec}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} V'
                elif elem == 'tRec':
                    label += '\n' + rf' $t_{{rec}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} s'
            axs[row,column_rec].add_collection(lc)
            axs[row,column_rec].set_ylabel(ylabel)
            # fig.text((ylabel_space+(k*axis_width + (k-1)*inter_space_width)*0.5)/W,1-(title_space + row*(axis_height+inter_space_height))/H,label, 
            #     horizontalalignment='center', verticalalignment='bottom', color ='k')
            axs[row,column_rec].text(0.95,0.95,label, transform=axs[row, column_rec].transAxes, horizontalalignment='right', verticalalignment='center')
            if vs_cyclevar:
                if t_plot == 'first':
                    t = [min(t)]
                elif t_plot == 'first_end':
                    t = [min(t),max(t)]
                elif t_plot == 'all':
                    t = t
                else:
                    t = t_plot
                if acc_cycle:
                    Stress_arr = [stress_val for stress_val in Stress_arr if stress_val != 0]
                for t_i in t:
                    df_t = df_meas[(np.isclose(df_meas[t_var], t_i)) & ~(df_meas['cycle']==0)]
                    cycle_var = np.array(df_t[varying].tolist())
                    Delta_cycle = np.array(df_t[y_var].tolist()) - np.array(df_t[y_var_ref].tolist())
                    if False and (varying == 'tStress') and len(cycle_var)>=3:
                        cycle_var_fit = np.array(df_t[varying].tolist())
                        Delta_fit = np.array(df_t[y_var_fit].tolist()) - np.array(df_t[y_var_ref].tolist())
                        fit_used = df_t['fit_time'].iloc[0]
                    else:
                        cycle_var_fit = cycle_var
                        Delta_fit = Delta_cycle
                    axs[row,column_str].plot(cycle_var, Delta_cycle, marker='o',linestyle='None', color = plt.cm.plasma(np.log10(t_i)/np.log10(max_t)))
                    axs[row,column_str].plot(cycle_var_fit, Delta_fit, linestyle='-', color = plt.cm.plasma(np.log10(t_i)/np.log10(max_t)))
                    if False and (varying == 'tStress'):
                        axs[row,column_str].text(cycle_var[len(cycle_var) // 2], Delta_fit[len(cycle_var) // 2], f'{fit_labels[fit_used]}', horizontalalignment='center', 
                            verticalalignment='top', color= plt.cm.plasma(np.log10(t_i)/np.log10(max_t)))
                    axs[row,column_str].axhline(0, linestyle='--', color = 'k')
                    axs[row,column_str].set_xlabel(xlabel)
                    if varying == 'tStress':
                        axs[row,column_str].set_xscale('log')
                        axs[row,column_str].set_xticks(Stress_arr)
                        axs[row,column_str].set_xticklabels([f'$10^{{{int(np.log10(s))}}}$' for s in Stress_arr])
                        #axs[row,column_str].xaxis.set_major_locator(plt.LogLocator(base=10, numticks=100))
                    else:
                        axs[row,column_str].set_xticks(Stress_arr)
                        axs[row,column_str].set_xticklabels(Stress_arr)
                    for j, tick in enumerate(axs[row,column_str].get_xticklabels()):
                        color = plt.cm.viridis(np.log10(Stress_arr[j])/np.log10(max_Stress)) if varying=='tStress' else plt.cm.viridis((Stress_arr[j]-min_Stress)/(max_Stress - min_Stress))
                        tick.set_color(color)
            row += 1
    fig.text(0.5, 1 - (title_space/3)/H, rf'{stress} varying {xlabel}: {batch}', ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_BTI_vars(dataset, vars, meas = None, df=None, name = None, show_sweep=False, show_IdVg_lin = False, overwrite=True, title=True, plot_format = 'pdf',transparent=False, fit=True, vars_fit=None):
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'tStress'        : r'$t_\mathsf{str}$ [s]',
        'VgStress'      : r'$V_\mathsf{G,str}$ [V]',
        'VgRemain'      : r'$V_\mathsf{G,rec}$ [V]',
        'tRec'           : r'$t_\mathsf{rec}$ [s]',
        'temp'          : r'T',
        'Eod_str'          : r'$E_\mathsf{od,str}$ [MV/cm]',
        'Vth' : r'$V_\mathsf{th}$ [V]',
        'Vth_initial' : r'$V_\mathsf{th,initial}$ [V]',
        'DeltaVth' : r'$\Delta V_\mathsf{th}$ [V]',
        'I_max' : r'$I_\mathsf{max}$ [A]',
        'Vg' : r'$V_\mathsf{G}$ [V]',
        'Id': r'$I_\mathsf{D}$ [A]',
        'Vinput': r'$V_\mathsf{input}$ [V]',
        'Voutput'   : r'$V_\mathsf{output}$ [V]',
        'Vm' : r'$V_\mathsf{M}$ [V]',
        'DeltaVm' : r'$\Delta V_\mathsf{M}$ [V]',
        'SS' : r'$SS$ [mV/dec]',
    }
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    k = sum([show_sweep,len(vars)])
    if vars_fit is None:
        vars_fit = [None]*len(vars)
    row = 0
    shows = ''
    if show_sweep:
        initial_sweep_row = row
        row += 1
        shows += 'sweep_'
    for var in vars:
        # initial_Vth_row = row
        row += 1
        shows += var + '_'
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_height_idvg = 1.5
    inter_space_width = 2
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['BTI'].values()})
    total_duts = len(duts)
    n_meas = np.sum([len(batch_meas['meas']['BTI']) for batch, batch_meas in meas.items()])
    plot_file = f'BTI_{shows}{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'BTI')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['BTI'].keys())[0]
            sample = meas[batch]['meas']['BTI'][key]['sample']
            T = meas[batch]['meas']['BTI'][key]['temp']
            plot_file += f'_{sample}_{T}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    total_cycles = []
    row_widths = []
    for batch,dut in duts:
        dut_cycles = [len(m['cycles']) for m in meas[batch]['meas']['BTI'].values() if m['dut']['name'] == dut]
        meas_width = [len(m['cycles'])*axis_width for m in meas[batch]['meas']['BTI'].values() if m['dut']['name'] == dut]
        total_cycles.append(np.sum(dut_cycles))
        row_widths.append(np.sum(meas_width) + (len(meas_width)-1)*inter_space_width)
    max_cycles = np.max(total_cycles)
    max_row_width = np.max(row_widths)
    fig, axs = plt.subplots(
        k*total_duts, max_cycles,
        figsize=(max_row_width + ylabel_space + right_additional_space, 
                 k*total_duts*axis_height + xlabel_space + title_space + (total_duts - 1)*inter_space_height + (k-1)*total_duts*inter_space_height_idvg), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(k*total_duts, max_cycles)
    if show_sweep:
        if show_IdVg_lin:
            axs2 = np.empty_like(axs, dtype=object)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['BTI'].items() if m['dut']['name'] == dut}
        main_row = k*i
        if show_sweep:
            sweep_row = k*i + initial_sweep_row
        column = 0
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + l*inter_space_width
            bottom = title_space + axis_height + (i)*k*(axis_height) + (i)*inter_space_height + (i)*(k-1)*inter_space_height_idvg
            for n in range(k):
                axs[main_row + n, column].set_position([left/W, 1 - (bottom + n*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
            for j,_ in enumerate(m['cycles']):
                #axs[Vth_row, column].sharex(axs[Vth_row, 0])
                for n in range(k):
                    if j>0:
                        axs[main_row + n, column].set_position([(left + j*axis_width)/W, 1 - (bottom + n*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                        axs[main_row + n, column].tick_params(axis='y', labelleft=False)
                    axs[main_row + n, column].sharey(axs[main_row + n, 0])
                    axs[main_row + n, column].sharex(axs[main_row + n, 0])
                    # if column > 0:
                    #     axs[main_row + n, column].tick_params(axis='y', labelleft=False)
                    if show_sweep:
                        if show_IdVg_lin:
                            axs2[sweep_row, column] = axs[sweep_row,column].twinx()
                            axs2[sweep_row, column].set_position(axs[sweep_row, column].get_position())
                            axs2[sweep_row, column].sharey(axs2[sweep_row, 0])
                            axs2[sweep_row, column].tick_params(axis='y', right=False, labelright=False)
                    # if j>0:
                    #     axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                    # if column > 0:
                    #     axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                # else:
                #     if i < (len(duts) - 1):
                #         axs[Vth_row, column].tick_params(axis='x', labelbottom=False)
                column += 1
            
        for non_col in range(column, max_cycles):
            for n in range(k):
                axs[main_row + n, non_col].set_visible(False)
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}.csv'))
    for c in ['Id','Vg','Id_fit','Vg_fit','Vinput','Voutput']:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: np.array(json5.loads(x))
                if isinstance(x, str) and x.strip() != ''
                else x
            )
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['BTI'].items() if m['dut']['name'] == dut}
        dut_cycles = np.sum([len(m['cycles']) for m in meas_dut.values()])
        main_row = k*i
        if show_sweep:
            sweep_row = k*i + initial_sweep_row
        column = 0
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + l*inter_space_width
            bottom = title_space + axis_height + (i)*k*(axis_height) + (i)*inter_space_height + (i)*show_sweep*inter_space_height_idvg
            sample = m['sample']
            cycles = m['cycles']
            T = m['temp']
            varying = m['varying']
            acc_cycle = m['acc_cycle']
            meas_type = m['meas_type']
            stress = m['stress']
            length = parse_length(m['dut']['length'], target_unit="um")
            width = parse_length(m['dut']['width'], target_unit="um")
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['meas_type'] == meas_type) &
                    (df['temp'] == T) &
                    (df['sample'] == sample)
                ]
            if meas_type == 'MSM':
                t_var = 'tRec'
                t_label = r'$t_\mathsf{rec}$ [s]'
            elif meas_type == 'OTF':
                t_var = 'tStress'
                t_label = r'$t_\mathsf{str}$ [s]'
            if meas_type == 'MSM':
                non_var = ['VgStress','VgRemain','tStress'].copy()
                non_var.remove(varying)
            elif meas_type == 'OTF':
                non_var = []
            max_tMeas = df_meas[df_meas[t_var].notna()][t_var].max()
            min_tMeas = df_meas[df_meas[t_var].notna()][t_var].min()
            for j,(cycle) in enumerate(sorted(cycles)):
                df_meas_cycle = df_meas[(df_meas['cycle'] == cycle)]
                if show_sweep:
                    if 'type' not in m['dut'] or m['dut']['type'] in ['nMOS', 'pMOS']:
                        xsweep_var = 'Vg'
                        ysweep_var = 'Id'
                        extract_level = df_meas_cycle['Ith'].iloc[0]
                    elif m['dut']['type'] == 'inverter':
                        xsweep_var = 'Vinput'
                        ysweep_var = 'Voutput'
                        extract_level = df_meas_cycle['Vdd'].iloc[0]/2
                    # Extra initial sweep
                    df_extra = df_meas_cycle[df_meas_cycle['extra'] == True]
                    if not df_extra.empty:
                        row = df_extra.iloc[0]
                        axs[sweep_row,column].plot(row[xsweep_var],row[ysweep_var], '.-', label='extra',color = 'g')
                        if show_IdVg_lin:
                            axs2[sweep_row,column].plot(row[xsweep_var], row[ysweep_var], '-', label='extra',color = 'g')
                    # Initial sweep
                    row = df_meas_cycle[df_meas_cycle['initial'] == True].iloc[0]
                    axs[sweep_row,column].plot(row[xsweep_var],row[ysweep_var], '-', label='initial',color = 'r')
                    if show_IdVg_lin:
                        axs2[sweep_row,column].plot(row[xsweep_var], row[ysweep_var], '-', label='initial',color = 'r')
                    t_values = df_meas_cycle[(df_meas_cycle[t_var] != 0.0) & (df_meas_cycle[t_var].notna())][t_var].unique().tolist()
                    for t_val in sorted(t_values):
                        row = df_meas_cycle[df_meas_cycle[t_var] == t_val].iloc[0]
                        axs[sweep_row,column].plot(row[xsweep_var], row[ysweep_var], '-', label=rf'$t_\mathsf{{rec}}$ = {int(t_val)} s',color = plt.cm.plasma((np.log10(t_val))/(np.log10(max_tMeas))))
                        axs[sweep_row,column].margins(x=0.10)
                        if ysweep_var == 'Id':
                            axs[sweep_row,column].set_yscale('log')
                        if show_IdVg_lin:
                            axs2[sweep_row,column].plot(row[xsweep_var], row[ysweep_var], '-', label=rf'$t_\mathsf{{rec}}$ = {int(t_val)} s',color = plt.cm.plasma((np.log10(t_val))/(np.log10(max_tMeas))))
                    # Extra sweep at the end
                    df_end = df_meas_cycle[df_meas_cycle['end'] == True]
                    if not df_end.empty:
                        row = df_end.iloc[0]
                        axs[sweep_row,column].plot(row[xsweep_var],row[ysweep_var], '--', label='extra',color = 'g')
                        if show_IdVg_lin:
                            axs2[sweep_row,column].plot(row[xsweep_var], row[ysweep_var], '-', label='extra',color = 'g')
                    axs[sweep_row,column].axhline(extract_level, linestyle='--', color = 'k')
                    axs[sweep_row,column].set_xlabel(var_labels[xsweep_var])
                for v,var in enumerate(vars):
                    # Initial value
                    df_initial = df_meas_cycle[df_meas_cycle['initial'] == True]
                    if not df_initial.empty and var in df_initial.columns:
                        y_value_initial = df_initial[var].to_numpy(dtype=float)[0]
                        color = 'r' if acc_cycle and cycle == 0 else plt.cm.viridis((cycle-1)/np.max(total_cycles))
                        axs[main_row + v + show_sweep,column].axhline(y_value_initial, linestyle='--', color = color)
                    # Extra value
                    df_extra = df_meas_cycle[df_meas_cycle['extra'] == True]
                    if not df_extra.empty and var in df_extra.columns:
                        y_value_extra = df_extra[var].to_numpy(dtype=float)[0]
                        axs[main_row + v + show_sweep,column].axhline(y_value_extra, linestyle='--', label='extra',color = 'g')
                    # End value
                    df_end = df_meas_cycle[df_meas_cycle['end'] == True]
                    if not df_end.empty and var in df_end.columns:
                        y_value_end = df_end[var].to_numpy(dtype=float)[0]
                        axs[main_row + v + show_sweep,column].axhline(y_value_end, linestyle='--', label='end',color = 'g')
                    df_meas_cycle_t = df_meas_cycle[(df_meas_cycle[t_var] != 0.0) & (df_meas_cycle[t_var].notna())].sort_values(by=t_var)
                    t = df_meas_cycle_t[t_var].to_numpy(dtype=float)
                    y_value = df_meas_cycle_t[var].to_numpy(dtype=float)
                    if acc_cycle and cycle == 0:
                        axs[main_row + v + show_sweep,column].plot(t, y_value, marker='o', linestyle='None', color = 'r')
                        axs[main_row + v + show_sweep,column].plot(t, y_value, linestyle='-', color = 'r')
                    else:
                        #axs[main_row + v + show_sweep,column].text(0.5,0.2,  var_labels[varying] + f'={df_meas_cycle[varying].iloc[0]}' , horizontalalignment='center', 
                        #verticalalignment='center', transform=axs[main_row + v + show_sweep,column].transAxes, color =plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        axs[main_row + v + show_sweep,column].plot(t, y_value, marker='o', linestyle='None', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        if fit and var in vars_fit:
                            fit_used = df_meas_cycle_t['fit_time'].iloc[0]
                            y_value_fit = df_meas_cycle_t[var+'_fit'].to_numpy(dtype=float)
                            y_value_fit = y_value_fit
                            #axs[Vth_row,column].text(t_fit[len(t_fit) // 2], Vth_fit[len(t_fit) // 2], f'{fit_labels[fit_used]}', horizontalalignment='center', 
                            #        verticalalignment='top', color= plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                            axs[main_row + v + show_sweep,column].plot(t, y_value_fit, linestyle='-', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        else:
                            axs[main_row + v + show_sweep,column].plot(t, y_value, linestyle='-', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                    axs[main_row + v + show_sweep,column].set_xscale('log')
                    axs[main_row + v + show_sweep,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                    t_axis = np.logspace(np.log10(min_tMeas), np.log10(max_tMeas),500)
                    t_colors = plt.cm.plasma(np.log10(t_axis) / np.log10(max(t_axis)))
                    axs[main_row + v + show_sweep,column].spines['bottom'].set_visible(False)
                    points = np.array([np.linspace(0, 1, 500), np.full(500, 0)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, linewidth=10, colors=t_colors[:-1], transform=axs[main_row + v + show_sweep,column].transAxes)
                    axs[main_row + v + show_sweep,column].add_collection(lc)
                    axs[main_row + v + show_sweep,column].set_xlabel(t_label)
                label = f'{dut} {sample} {T}'
                for l,elem in enumerate(non_var):
                    if elem == 'tStress':
                        label += '\n' + rf' $t_\mathsf{{str}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} s'
                    elif elem == 'VgStress':
                        label +=  '\n' + rf' $V_\mathsf{{TG,str}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} V'
                    elif elem == 'VgRemain':
                        label +='\n' + rf' $V_\mathsf{{TG,rec}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} V'
                    elif elem == 'tRec':
                        label += '\n' + rf' $t_\mathsf{{rec}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} s'
                if j == 0:
                    # fig.text(left/W,1 - (bottom - axis_height)/H,label, 
                    # horizontalalignment='left', verticalalignment='bottom', color ='k')
                    axs[main_row,column].text(0.05,0.95,label, 
                    horizontalalignment='left',transform=axs[main_row,column].transAxes ,verticalalignment='top', color ='k')
                if acc_cycle and cycle == 0:
                    axs[main_row,column].text(0.5,0.2,  'Accomodation cycle:\n' + var_labels[varying] + f'={df_meas_cycle[varying].iloc[0]}' , horizontalalignment='center', 
                    verticalalignment='center', transform=axs[main_row,column].transAxes, color ='k')
                else:
                    axs[main_row,column].text(0.5,0.2,  var_labels[varying] + f'={df_meas_cycle[varying].iloc[0]}' , horizontalalignment='center', 
                            verticalalignment='center', transform=axs[main_row,column].transAxes, color ='k')
                column += 1
            if show_sweep:
                axs[sweep_row,0].set_ylabel(var_labels[ysweep_var])
            for v, var in enumerate(vars):
                axs[main_row + v + show_sweep,0].set_ylabel(var_labels[var])
    if title:
        if name:
            fig.text(0.5, 1 - (title_space/3)/H, rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5, 1 - (title_space/3)/H, rf'BTI: {dataset}', ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_BTI_Vth(dataset, meas = None, df=None, name = None, show_IdVg=False, show_IdVg_lin = False,show_Vth=True, overwrite=True, title=True, plot_format = 'pdf',transparent=False,fit=True):
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'tStress'        : r'$t_\mathsf{str}$ [s]',
        'VgStress'      : r'$V_\mathsf{G,str}$ [V]',
        'VgRemain'      : r'$V_\mathsf{G,rec}$ [V]',
        'tRec'           : r'$t_\mathsf{rec}$ [s]',
        'temp'          : r'T',
        'Eod_str'          : r'$E_\mathsf{od,str}$ [MV/cm]',
        'Vth_initial' : r'$V_\mathsf{th,initial}$ [V]'
    }
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    k = sum([show_IdVg,show_Vth])
    row = 0
    shows = ''
    if show_IdVg:
        initial_idvg_row = row
        row += 1
        shows += 'IdVg_'
    if show_Vth:
        initial_Vth_row = row
        row += 1
        shows += 'Vth_'
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 2
    inter_space_height_idvg = 1.5
    inter_space_width = 1.5
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['BTI'].values()})
    total_duts = len(duts)
    n_meas = np.sum([len(batch_meas['meas']['BTI']) for batch, batch_meas in meas.items()])
    plot_file = f'BTI_{shows}{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'BTI')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['BTI'].keys())[0]
            sample = meas[batch]['meas']['BTI'][key]['sample']
            T = meas[batch]['meas']['BTI'][key]['temp']
            plot_file += f'_{sample}_{T}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    total_cycles = []
    row_widths = []
    for batch,dut in duts:
        dut_cycles = [len(m['cycles']) for m in meas[batch]['meas']['BTI'].values() if m['dut']['name'] == dut]
        meas_width = [len(m['cycles'])*axis_width for m in meas[batch]['meas']['BTI'].values() if m['dut']['name'] == dut]
        total_cycles.append(np.sum(dut_cycles))
        row_widths.append(np.sum(meas_width) + (len(meas_width)-1)*inter_space_width)
    max_cycles = np.max(total_cycles)
    max_row_width = np.max(row_widths)
    fig, axs = plt.subplots(
        k*total_duts, max_cycles,
        figsize=(max_row_width + ylabel_space + right_additional_space, 
                 k*total_duts*axis_height + xlabel_space + title_space + (total_duts - 1)*inter_space_height + (k-1)*total_duts*inter_space_height_idvg), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(k*total_duts, max_cycles)
    if show_IdVg:
        if show_IdVg_lin:
            axs2 = np.empty_like(axs, dtype=object)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['BTI'].items() if m['dut']['name'] == dut}
        main_row = k*i
        if show_Vth:
            Vth_row = k*i + initial_Vth_row
        if show_IdVg:
            idvg_row = k*i + initial_idvg_row
        column = 0
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + l*inter_space_width
            bottom = title_space + axis_height + (i)*k*(axis_height) + (i)*inter_space_height + (i)*show_IdVg*inter_space_height_idvg
            for n in range(k):
                if show_IdVg and n == 1:
                    axs[main_row + n, column].set_position([left/W, 1 - (bottom + n*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                else:
                    axs[main_row + n, column].set_position([left/W, 1 - (bottom + n*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
            for j,_ in enumerate(m['cycles']):
                #axs[Vth_row, column].sharex(axs[Vth_row, 0])
                if j>0:
                    if show_IdVg:
                        axs[idvg_row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + idvg_row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                    elif show_Vth:
                        axs[Vth_row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + Vth_row*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
                    for n in range(k):
                        if show_IdVg and n == 1:
                            axs[main_row + n, column].set_position([(left + j*axis_width)/W, 1 - (bottom + n*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                        else:
                            axs[main_row + n, column].set_position([(left + j*axis_width)/W, 1 - (bottom + n*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
                if show_Vth:
                    axs[Vth_row, column].sharey(axs[Vth_row, 0])
                    if column > 0:
                        axs[Vth_row, column].tick_params(axis='y', labelleft=False)
                if show_IdVg:
                    axs[idvg_row, column].sharex(axs[idvg_row, 0])
                    axs[idvg_row, column].sharey(axs[idvg_row, 0])
                    if show_IdVg_lin:
                        axs2[idvg_row, column] = axs[idvg_row,column].twinx()
                        axs2[idvg_row, column].set_position(axs[idvg_row, column].get_position())
                        axs2[idvg_row, column].sharey(axs2[idvg_row, 0])
                        axs2[idvg_row, column].tick_params(axis='y', right=False, labelright=False)
                    if j>0:
                        axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                    if column > 0:
                        axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                # else:
                #     if i < (len(duts) - 1):
                #         axs[Vth_row, column].tick_params(axis='x', labelbottom=False)
                column += 1
            
        for non_col in range(column, max_cycles):
            if show_Vth:
                axs[Vth_row, non_col].set_visible(False)
            if show_IdVg:
                axs[idvg_row, non_col].set_visible(False)
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'BTI_{dataset}.csv'))
    for c in ['Id','Vg','Id_fit','Vg_fit']:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: np.array(json5.loads(x))
                if isinstance(x, str) and x.strip() != ''
                else x
            )
    #df.loc[~df['tMeas'].isin(['initial', 'extra','end']), 'tMeas'] = df.loc[~df['tMeas'].isin(['initial', 'extra','end']),'tMeas'].astype(float)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['BTI'].items() if m['dut']['name'] == dut}
        dut_cycles = np.sum([len(m['cycles']) for m in meas_dut.values()])
        main_row = k*i
        if show_Vth:
            Vth_row = k*i + initial_Vth_row
        if show_IdVg:
            idvg_row = k*i + initial_idvg_row
        column = 0
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + l*inter_space_width
            bottom = title_space + axis_height + (i)*k*(axis_height) + (i)*inter_space_height + (i)*show_IdVg*inter_space_height_idvg
            sample = m['sample']
            if 'vth_extract' in m and 'method' in m['vth_extract']:
                vth_extract = m['vth_extract']['method']
                y_var = 'Vth'
                y_var_fit = 'Vth_fit'
                y_var_initial = 'Vth_initial'
                ylabel = r'$V_\mathsf{th}$ [V]'
                if vth_extract == 'constant_current' or 'constant_current_L/W':
                    current_level = m['vth_extract']['current_level']
                else:
                    current_level = None
            else:
                vth_extract = None
                y_var = 'I'
                y_var_fit = 'I_fit'
                y_var_initial = 'I_initial'
                ylabel = r'$I_\mathsf{D}$ [A]'
            cycles = m['cycles']
            T = m['temp']
            varying = m['varying']
            acc_cycle = m['acc_cycle']
            meas_type = m['meas_type']
            stress = m['stress']
            length = parse_length(m['dut']['length'], target_unit="um")
            width = parse_length(m['dut']['width'], target_unit="um")
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['meas_type'] == meas_type) &
                    (df['temp'] == T) &
                    (df['sample'] == sample)
                ]
            if meas_type == 'MSM':
                t_var = 'tRec'
                t_label = r'$t_\mathsf{rec}$ [s]'
            elif meas_type == 'OTF':
                t_var = 'tStress'
                t_label = r'$t_\mathsf{str}$ [s]'
            if meas_type == 'MSM':
                non_var = ['VgStress','VgRemain','tStress'].copy()
                non_var.remove(varying)
            elif meas_type == 'OTF':
                non_var = []
            max_tMeas = df_meas[df_meas[t_var].notna()][t_var].max()
            min_tMeas = df_meas[df_meas[t_var].notna()][t_var].min()
            for j,(cycle) in enumerate(sorted(cycles)):
                if show_IdVg:
                    subset = df_meas[
                        (df_meas['cycle'] == cycle)
                    ]
                    row = subset[subset['initial'] == True].iloc[0]
                    axs[idvg_row,column].plot(row['Vg'],row['Id'], '.-', label='initial',color = 'r')
                    if show_IdVg_lin:
                        axs2[idvg_row,column].plot(row['Vg'], row['Id'], '.-', label='initial',color = 'r')
                    if vth_extract:
                        Vth,Ith,extractV,extractI = Vth_extraction(row['Vg'],row['Id'],row['Vd'],vth_extract=vth_extract,current_level=current_level,width=width,length=length)
                        axs[idvg_row,column].plot(Vth, Ith, marker='x',linestyle='None',color='k')
                        if show_IdVg_lin:
                            axs2[idvg_row,column].plot(Vth, Ith, marker='x', linestyle='None', color='k')
                        if vth_extract == 'constant_current':
                            axs[idvg_row,column].axhline(current_level, linestyle='--', color = 'k')
                            if show_IdVg_lin:
                                axs2[idvg_row,column].axhline(current_level, linestyle='--', color = 'k')
                        elif vth_extract == 'constant_current_L/W':
                            axs[idvg_row,column].axhline(Ith, linestyle='--', color = 'k')
                            if show_IdVg_lin:
                                axs2[idvg_row,column].plot(extractV, extractI, linestyle='--', color = 'k')
                        elif vth_extract == 'linear_extrapolation':
                            axs[idvg_row,column].axhline(Ith, linestyle='--', color = 'k')
                            if show_IdVg_lin:
                                axs2[idvg_row,column].plot(extractV, extractI, linestyle='--', color = 'k')
                    t_values = subset[(subset[t_var] != 0.0) & (subset[t_var].notna())][t_var].unique().tolist()
                    for t_val in sorted(t_values):
                        row = subset[subset[t_var] == t_val].iloc[0]
                        axs[idvg_row,column].plot(row['Vg'], row['Id'], '.-', label=rf'$t_\mathsf{{rec}}$ = {int(t_val)} s',color = plt.cm.plasma((np.log10(t_val))/(np.log10(max_tMeas))))
                        axs[idvg_row,column].margins(x=0.10)
                        axs[idvg_row,column].set_yscale('log')
                        if show_IdVg_lin:
                            axs2[idvg_row,column].plot(row['Vg'], row['Id'], '.-', label=rf'$t_\mathsf{{rec}}$ = {int(t_val)} s',color = plt.cm.plasma((np.log10(t_val))/(np.log10(max_tMeas))))
                        if vth_extract:
                            Vth,Ith,extractV,extractI = Vth_extraction(row['Vg'],row['Id'],row['Vd'],vth_extract='constant_current',current_level=Ith)
                            # axs[idvg_row,column].plot(Vth, Ith, marker='x', linestyle='None', color=plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                            # axs2[idvg_row,column].plot(Vth, Ith, marker='x', linestyle='None', color=plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                            # if vth_extract == 'linear_extrapolation':
                            #     axs2[idvg_row,column].plot(extractV, extractI, linestyle='--', color = plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                    axs[idvg_row,column].set_xlabel(r'$V_\mathsf{G}$ [V]')
                if show_Vth:
                    df_t = df_meas[(df_meas['cycle'] == cycle)]
                    df_t = df_t[(df_t[t_var] != 0.0) & (df_t[t_var].notna())]
                    t = df_t[t_var].to_numpy(dtype=float)
                    y_value = df_t[y_var].to_numpy(dtype=float)
                    idx = np.argsort(t)
                    t = t[idx]
                    y_value = y_value[idx]
                    y_value_initial = df_t[y_var_initial].iloc[0]
                    if acc_cycle and cycle == 0:
                        axs[Vth_row,column].plot(t, y_value, marker='o', linestyle='None', color = 'r')
                        axs[Vth_row,column].plot(t, y_value, linestyle='-', color = 'r')
                        if meas_type == 'MSM':
                            Vgrem = df_t['VgRemain'].iloc[0]
                            axs[Vth_row,column].text(0.5,0.2,  'Accomodation cycle:\n' + rf'$V_\mathsf{{G,rec}}$ = {Vgrem} V' , horizontalalignment='center', 
                            verticalalignment='center', transform=axs[Vth_row,column].transAxes, color ='r')
                        elif meas_type == 'OTF':
                            Vgstr = df_t['VgStress'].iloc[0]
                            axs[Vth_row,column].text(0.5,0.2,  'Accomodation cycle:\n' + rf'$V_\mathsf{{G,str}}$ = {Vgstr} V' , horizontalalignment='center', 
                            verticalalignment='center', transform=axs[Vth_row,column].transAxes, color ='r')
                        axs[Vth_row,column].axhline(y_value_initial, linestyle='--', color = 'r')
                    else:
                        axs[Vth_row,column].text(0.5,0.2,  var_labels[varying] + f'={df_t[varying].iloc[0]}' , horizontalalignment='center', 
                        verticalalignment='center', transform=axs[Vth_row,column].transAxes, color =plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        axs[Vth_row,column].plot(t, y_value, marker='o', linestyle='None', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        axs[Vth_row,column].axhline(y_value_initial, linestyle='--', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        if fit and y_var_fit in df_t.columns:
                            fit_used = df_t['fit_time'].iloc[0]
                            y_value_fit = df_t[y_var_fit].to_numpy(dtype=float)
                            y_value_fit = y_value_fit[idx]
                            #axs[Vth_row,column].text(t_fit[len(t_fit) // 2], Vth_fit[len(t_fit) // 2], f'{fit_labels[fit_used]}', horizontalalignment='center', 
                            #        verticalalignment='top', color= plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                            axs[Vth_row,column].plot(t, y_value_fit, linestyle='-', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                        else:
                            axs[Vth_row,column].plot(t, y_value, linestyle='-', color = plt.cm.viridis((cycle-1)/np.max(total_cycles)))
                    axs[Vth_row,column].set_xscale('log')
                    # if y_var == 'I_t': # and meas_type == 'OTF'
                    #     axs[Vth_row,column].set_yscale('log')
                    axs[Vth_row,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                    #axs[Vth_row,column].set_xticks([10,1000,100000])
                    #axs[Vth_row,column].set_xticklabels([r'$10^{1}$',r'$10^{3}$',r'$10^{5}$'])
                    t_axis = np.logspace(np.log10(min_tMeas), np.log10(max_tMeas),500)
                    t_colors = plt.cm.plasma(np.log10(t_axis) / np.log10(max(t_axis)))
                    axs[Vth_row,column].spines['bottom'].set_visible(False)
                    points = np.array([np.linspace(0, 1, 500), np.full(500, 0)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments,
                        linewidth=10,
                        colors=t_colors[:-1],
                        transform=axs[Vth_row,column].transAxes
                    )
                    axs[Vth_row,column].add_collection(lc)
                    label = f'{dut} {sample} {T}'
                    for l,elem in enumerate(non_var):
                        if elem == 'tStress':
                            label += '\n' + rf' $t_\mathsf{{str}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} s'
                        elif elem == 'VgStress':
                            label +=  '\n' + rf' $V_\mathsf{{TG,str}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} V'
                        elif elem == 'VtgRemain':
                            label +='\n' + rf' $V_\mathsf{{TG,rec}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} V'
                        elif elem == 'tRec':
                            label += '\n' + rf' $t_\mathsf{{rec}}$ = {df_meas[(df_meas['cycle'] == 1)][non_var[l]].iloc[0]} s'
                    if j == 0:
                        # fig.text(left/W,1 - (bottom - axis_height)/H,label, 
                        # horizontalalignment='left', verticalalignment='bottom', color ='k')
                        axs[main_row,column].text(0.05,0.95,label, 
                        horizontalalignment='left',transform=axs[main_row,column].transAxes ,verticalalignment='top', color ='k')
                    axs[Vth_row,column].set_xlabel(t_label)
                column += 1
            if show_Vth:
                axs[Vth_row,0].set_ylabel(ylabel)
            if show_IdVg:
                axs[idvg_row,0].set_ylabel(r'$I_\mathsf{D}$ [A]')
    if title:
        if name:
            fig.text(0.5,1 - (title_space/3)/H,rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5,1 - (title_space/3)/H,rf'BTI: {dataset}',ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_BTI_CETmap(df_Vth, manu, geo, batch, meas, name = None, plot_DeltaVth = False, title=True):
    if plot_DeltaVth == True:
        k = 2
    else:
        k = 1
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.35
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width = 1.5
    fig, axs = plt.subplots(len(meas),k,
    figsize=(k*axis_width + ylabel_space + right_additional_space + (k-1)*inter_space_width, 
                 len(meas)*axis_height + xlabel_space + title_space + (len(meas)-1)*inter_space_height),
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    plt.subplots_adjust(wspace=inter_space_width, hspace=inter_space_height)
    axs = np.array(axs).reshape(len(meas), k)
    column_CET = 0
    column_DeltaVth = 1
    for i in range(len(meas)):
        axs[i, column_CET].sharex(axs[0, column_CET])
        axs[i, column_CET].sharey(axs[0, column_CET])
        if i != len(meas)-1:
            axs[i, column_CET].tick_params(axis='x', labelbottom=False)
        if plot_DeltaVth:
            axs[i, column_DeltaVth].sharex(axs[i, column_CET])
            axs[i, column_DeltaVth].sharey(axs[i, column_CET])
            axs[i, column_DeltaVth].tick_params(axis='y', labelleft=False)
            if i != len(meas)-1:
                axs[i, column_DeltaVth].tick_params(axis='x', labelbottom=False)
    for i,m in enumerate(meas.values()):
        dut = m['dut']
        sample = m['sample']
        T = m['temp']
        current_level = m['current_level']
        df_Vth_meas = df_Vth[
                (df_Vth['dut'] == dut) &
                (df_Vth['temp'] == T) &
                (df_Vth['sample'] == sample) &
                (df_Vth['cycle'] != 0)
            ]
        tStress_unique = np.array(sorted(df_Vth_meas['tStress'].unique()))
        tRecs_unique = np.array(sorted(df_Vth_meas['tRec'].unique()))
        DeltaVth = np.full((len(tStress_unique),len(tRecs_unique)),np.nan)
        for j, tstr in enumerate(tStress_unique):
            for l, trec in enumerate(tRecs_unique):
                mask = (
                    (df_Vth_meas['tStress'] == tstr) &
                    (df_Vth_meas['tRec'] == trec)
                )
                if mask.any():
                    row = df_Vth_meas[mask].iloc[0]
                    DeltaVth[j,l] = row['Vth_t'] - row['Vth_ref']
        valid_rows = ~np.isnan(DeltaVth).any(axis=1)
        valid_cols = ~np.isnan(DeltaVth).any(axis=0)
        DeltaVth = DeltaVth[valid_rows][:, valid_cols]
        tRecs    = tRecs_unique[valid_cols]
        tStress  = tStress_unique[valid_rows]
        TREC,TSTR = np.meshgrid(tRecs, tStress)
        spline = interpolate.RectBivariateSpline(
            tStress, tRecs, DeltaVth, kx=3, ky=3, s=0
        )
        d2DeltaVth = TREC*TSTR*spline.ev(TREC, TSTR, dx=1, dy=1)
        CET = axs[i,column_CET].contourf(
             TREC, TSTR, d2DeltaVth,
            levels=15
        )
        axs[i,column_CET].set_ylabel(r'$t_{str}$ (s)')
        fig.colorbar(CET, ax=axs[i,column_CET])
        axs[i,column_CET].set_xscale('log')
        axs[i,column_CET].set_yscale('log')
        axs[i,column_CET].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axs[i,column_CET].yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axs[i,column_CET].text(0.5,0.2,  rf'{dut} {sample} {T}' , horizontalalignment='center', 
                verticalalignment='center', transform=axs[i,column_CET].transAxes, color ='k')
        if i == len(meas) - 1:
            axs[i,column_CET].set_xlabel(r'$t_{rec}$ (s)')
        if plot_DeltaVth == True:
            DeltaVth_contour = axs[i,column_DeltaVth].contourf(
                TREC, TSTR, spline.ev(TSTR, TREC),
                levels=15
            )
            axs[i,column_DeltaVth].set_xscale('log')
            axs[i,column_DeltaVth].set_yscale('log')
            axs[i,column_DeltaVth].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            axs[i,column_DeltaVth].yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            fig.colorbar(DeltaVth_contour, ax=axs[i,column_DeltaVth])
            if i == len(meas) - 1:
                axs[i,column_DeltaVth].set_xlabel(r'$t_{rec}$ (s)')
    fig.text(
                0.5,
                1 - (title_space/2)/H,
                rf'{name}: {manu} {geo} {batch}',
                ha='center', va='center'
            )
    if len(meas) == 1:
        plt.savefig(os.path.join(script_dir,'plots', manu, geo, batch,'BTI',
                                    f'BTI_CETmap_{manu}_{geo}_{batch}_{T}_{dut}_{sample}_{name}.pdf'),bbox_inches=None)
    else:
        plt.savefig(os.path.join(script_dir,'plots', manu, geo, batch,'BTI',
                                    f'BTI_CETmap_{manu}_{geo}_{batch}_{name}.pdf'),bbox_inches=None)
    plt.close()

def plot_BTI_DeltaVth_vs_var_duts(dataset, meas_type, grouping, varying, df=None, showing=None, multiple=None, filter_plot={}, legend = ['dut','sample'], texts=None, name=None, fit=False, across=None, show_add=False, plot_format='pdf',transparent=True,overwrite=True,title=True):
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'tStress'        : r'$t_{str}$ [s]',
        'VgStress'      : r'$V_{G,str}$ [V]',
        'VgRemain'      : r'$V_{G,rec}$ [V]',
        'tRec'           : r'$t_{rec}$ [s]',
        'temp'          : r'T',
        'Eod_str'          : r'$E_{od,str}$ [MV/cm]',
        'Vth_initial' : r'$V_{th,initial}$ [V]',
    }
    if show_add:
        k = 2
    else:
        k = 1
    plot_file = f'{meas_type}_DeltaVth_{dataset}_vs_{varying}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'BTI')
    if grouping:
        plot_file += '_grouping'
        for grouping_var in grouping:
            plot_file += f'_{grouping_var}'
            if grouping_var in filter_plot:
                plot_file += f'_{filter_plot[grouping_var]}'
    if showing:
        plot_file += '_showing'
        for showing_var in showing:
            plot_file += f'_{showing_var}'
            if showing_var in filter_plot:
                plot_file += f'_{filter_plot[showing_var]}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    if df is None:
        if name is None:
            df = pd.read_csv(os.path.join(script_dir,'data', dataset, f'{meas_type}_{dataset}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data', dataset, f'{meas_type}_{dataset}_{name}.csv'))
    for variable,plot in filter_plot.items():
        df = df[df[variable].isin(plot)]
    df_subset = df
    groups = (df_subset.groupby(grouping) if grouping else [(None, df_subset)])
    if across:
        n_rows = sum(subset.groupby(across).ngroups > 1
        for _, subset in groups)
        df_subset = df_subset.groupby(grouping).filter(
                lambda subset: subset.groupby(across).ngroups > 1
            )
    else:
        n_rows = df_subset.groupby(grouping).ngroups
    if n_rows == 0:
        print(f'No sufficient data to plot DeltaVth vs {varying} for multiple {across} under the chosen conditions.')
        return
    ylabel_space = 1.6
    if legend:
        right_additional_space = 10
    else:
        right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.5
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width = 1.5
    fig, axs = plt.subplots(
        n_rows, k,
        figsize=(k*axis_width + ylabel_space + right_additional_space + (k-1)*inter_space_width, 
                 axis_height*n_rows + xlabel_space + title_space + (n_rows-1)*inter_space_height), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    plt.subplots_adjust(wspace=inter_space_width/axis_width, hspace=inter_space_height/axis_height)
    axs = np.array(axs).reshape(n_rows, k)
    for i in range(n_rows):
        axs[i,0].sharex(axs[0,0])
        axs[i,0].sharey(axs[0,0])
        # if i != n_rows-1:
        #     axs[i,0].tick_params(axis='x', labelbottom=False)
        if show_add:
            axs[i,k-1].sharex(axs[0,k-1])
            axs[i,k-1].sharey(axs[0,k-1])
            # if i != n_rows-1:
            #     axs[i,k-1].tick_params(axis='x', labelbottom=False)
    markers = ['o','v','^','<','>','s','p','*','h','H','+','x','D','d','|','_','1','2','3','4','.',',']
    all_samples = df_subset[['batch','dut','sample']].drop_duplicates()
    all_samples = [tuple(x) for x in all_samples.values]
    global_marker_map = {
        combo: markers[i % len(markers)]
        for i, combo in enumerate(all_samples)
    }
    if showing:
        all_showed = df_subset[showing].drop_duplicates()#.sort_values(by=showing[0])
        all_showed = list(map(tuple, all_showed.to_numpy()))
        colors = plt.cm.viridis(np.linspace(0.1,0.9,len(all_showed)))
        global_color_map = {
            combo: colors[i % len(colors)]
            for i, combo in enumerate(all_showed)
        }
    else:
        colors = plt.cm.tab10(np.linspace(0,1,10))
    edge_colors = colors
    i=0
    marker_handles = []
    group_map = {}
    for group_elements, subset in df_subset.groupby(grouping):
        group_text = ''
        for j,elem in enumerate(grouping):
            if elem in var_labels.keys():
                group_text += var_labels[elem] + f'= {subset[elem].iloc[0]} \n'
            else:
                group_text += f'{subset[elem].iloc[0]} \n'
        axs[i,0].text(0.05,0.95, group_text,
                horizontalalignment='left', verticalalignment='top', transform=axs[i,0].transAxes, color ='k')
        # fig.text((ylabel_space+(k*(axis_width)+(k-1)*inter_space_width)*0.5)/W,1-(title_space + i*(axis_height+inter_space_height))/H,group_text, 
        #         horizontalalignment='center', verticalalignment='bottom', color ='k')
        sublabel_handles = []
        groups_showing = [(None, subset)] if showing is None else subset.groupby(showing)
        for group_showing, subset_showing in groups_showing:
            for (batch,dut,sample), subset_sample in subset_showing.groupby(['batch', 'dut','sample']):
                var_plot = subset_sample[varying]
                marker = global_marker_map[(batch, dut, sample)]
                label = ''
                for j,elem in enumerate(legend):
                    if elem in var_labels.keys():
                        label += var_labels[elem] + f'= {subset_sample[elem].iloc[0]} '
                    else:
                        label += f'{subset_sample[elem].iloc[0]} '
                if (showing is not None):
                    color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
                else:
                    color = colors[i%len(colors)]
                if (multiple is not None) and (subset_sample.groupby(multiple).ngroups > 1):
                    k = 0
                    for mul,subset_mul in subset_sample.groupby(multiple):
                        label1 = label + f' {var_labels[multiple]}={mul}'
                        var_plot = subset_mul[varying].to_numpy()
                        DeltaVth = subset_mul['Vth'].to_numpy() - subset_mul['Vth_ref'].iloc[0]
                        idx = np.argsort(var_plot)
                        var_plot = var_plot[idx]
                        DeltaVth = DeltaVth[idx]
                        axs[i,0].plot(var_plot, DeltaVth, marker=marker, linestyle = 'None', markerfacecolor = color, markeredgecolor=edge_colors[k], alpha=1)
                        if fit and varying in ['tStress','tRec']:
                            DeltaVth_fit = subset_sample['Vth_fit'].to_numpy() - subset_sample['Vth_ref'].iloc[0]
                            axs[i,0].plot(var_plot, DeltaVth_fit, linestyle='-', color=color)
                        if show_add == 'DeltaVth/tox':
                            tox = subset_mul['tox'].to_numpy()
                            axs[i,k-1].plot(var_plot, DeltaVth/tox, marker=marker, linestyle = 'None', markerfacecolor = color, markeredgecolor=edge_colors[k], alpha=1)
                            if fit and varying in ['tStress','tRec']:
                                axs[i,k-1].plot(var_plot, DeltaVth_fit/tox, linestyle='-', color=color)
                        if not any(h.get_label() == label for h in marker_handles):
                            marker_handles.append(Line2D([], [],marker=marker,linestyle='None',markerfacecolor=color,markeredgecolor=edge_colors[k],label=label1))
                            k += 1
                else:
                    var_plot = subset_sample[varying].to_numpy()
                    DeltaVth = subset_sample['Vth'].to_numpy() - subset_sample['Vth_ref'].iloc[0]
                    idx = np.argsort(var_plot)
                    var_plot = var_plot[idx]
                    DeltaVth = DeltaVth[idx]
                    axs[i,0].plot(var_plot, DeltaVth, marker=marker, linestyle = 'None', markerfacecolor = color, alpha=1)
                    if fit and varying in ['tStress','tRec']:
                        DeltaVth_fit = subset_sample['Vth_fit'].to_numpy() - subset_sample['Vth_ref'].iloc[0]
                        DeltaVth_fit = DeltaVth_fit[idx]
                        axs[i,0].plot(var_plot, DeltaVth_fit, linestyle='-', color=color)
                        fit_used = subset_sample['fit_time'].iloc[0]
                        axs[i,0].text(var_plot[len(var_plot) // 2], DeltaVth_fit[len(var_plot) // 2], f'{fit_labels[fit_used]}', horizontalalignment='center', 
                        verticalalignment='top', color= color)
                    if show_add == 'DeltaVth/tox':
                        tox = subset_sample['tox'].to_numpy()*10**7 # nm to cm
                        axs[i,k-1].plot(var_plot, DeltaVth*1e-6/tox, marker=marker, linestyle = 'None', markerfacecolor = color, alpha=1)
                        if fit and varying in ['tStress','tRec']:
                            axs[i,k-1].plot(var_plot, DeltaVth_fit*1e-6/tox, linestyle='-', color=color)
                if not any(h.get_label() == label for h in marker_handles):
                    marker_handles.append(Line2D([], [], marker=marker, linestyle = 'None', markerfacecolor = color, alpha=1,label=label))
                if texts:
                    sublabel = ''
                    for j,elem in enumerate(texts):
                        if elem in var_labels.keys():
                            sublabel += var_labels[elem] + f'= {subset_sample[elem].iloc[0]:.2f}'
                        else:
                            sublabel += f'{subset_sample[elem].iloc[0]}'
                    sublabel_handles.append(Line2D([], [],marker=marker,linestyle='None',markerfacecolor=color,label=sublabel))
                    #axs[i,0].text(var_plot[len(var_plot) // 2], DeltaVth[len(var_plot) // 2], text, horizontalalignment='center', 
                    #    verticalalignment='top', color= color)
        leg = axs[i,0].legend(handles=sublabel_handles, ncol=1,loc = 'best', handlelength=0.5, framealpha=0.1, fontsize=18)
        axs[i,0].add_artist(leg) 
        axs[i,0].axhline(0, linestyle='--', color = 'k')
        axs[i,0].set_ylabel(rf'$\Delta V_{{th}}$ [V]')
        #ymin, ymax = axs[i,0].get_ylim()
        #axs[i,0].set_ylim(ymin, ymax)
        axs[i, 0].sharey(axs[0, 0])
        if show_add == 'DeltaVth/tox':
            axs[i,k-1].set_ylabel(rf'$\Delta V_{{th}}/t_{{ox}}$ [MV/cm]')
            #ymin, ymax = axs[i,k-1].get_ylim()
            #axs[i,k-1].set_ylim(ymin, ymax)
            axs[i,k-1].axhline(0, linestyle='--', color = 'k')
            axs[i,k-1].sharey(axs[0,k-1])
            leg1 = axs[i,k-1].legend(handles=sublabel_handles, ncol=1,loc = 'best', handlelength=0.5, framealpha=0.1, fontsize=18)
            axs[i,k-1].add_artist(leg1)
        if varying == 'tStress' or varying == 'tRec':
            axs[i,0].set_xscale('log')
            axs[i,0].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            if show_add:
                axs[i,k-1].set_xscale('log')
                axs[i,k-1].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axs[i,0].set_xlabel(var_labels[varying])
        if show_add:
            axs[i,k-1].set_xlabel(var_labels[varying])
        group_map[group_elements] =i
        i+=1
    # if fit:
    #     for group_elements, subset_fit in df_DeltaVthfit_subset.groupby(grouping):
    #         i = group_map[group_elements]
    #         for group_showing, subset_showing in subset_fit.groupby(showing):
    #             for group_sample, subset_sample in subset_showing.groupby(['batch','dut','sample']):
    #                 var_plot_fit = subset_sample[varying].to_numpy()
    #                 DeltaVth_fit = subset_sample['DeltaVth_fit'].to_numpy()
    #                 idx = np.argsort(var_plot_fit)
    #                 DeltaVth_fit = DeltaVth_fit[idx]
    #                 var_plot_fit = var_plot_fit[idx]
    #                 fit_type = subset_sample['fit'].iloc[0]
    #                 color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
    #                 axs[i,0].plot(var_plot_fit, DeltaVth_fit, linestyle='-', color=color)
    #                 if show_add == 'DeltaVth/tox':
    #                     tox = subset_sample['tox'].to_numpy()
    #                     axs[i,k-1].plot(var_plot_fit, DeltaVth_fit/tox, linestyle='-', color=color)
    #                 if texts == 'fit':
    #                     axs[i,0].text(var_plot_fit[len(var_plot_fit) // 2], DeltaVth_fit[len(var_plot_fit) // 2], f'{fit_labels[fit_type]}', horizontalalignment='center', 
    #                             verticalalignment='top', color= color)
    
    # Combine both legends
    all_handles = marker_handles
    axs[0,k-1].legend(handles=all_handles, ncol=1,loc = 'upper left',bbox_to_anchor=(1.05, 1), handlelength=1.0, framealpha=1)
    fig.text(0.5, 1 - (title_space/2)/H, f'{name}: {dataset}' if name else f'{dataset}', ha='center', va='center')
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_hyst_Vth_freq(dataset, meas = None, df=None, name = None, show_IdVg=False, show_IdVg_lin=False,show_Vth=False, show_DeltaVth=True,show_precond=False, overwrite=True,title=True,fit=True, plot_format='pdf',transparent=True):
    k = sum([show_IdVg, show_Vth, show_DeltaVth])
    row = 0
    shows = ''
    if show_IdVg:
        idvg_row = row
        row += 1
        shows += 'IdVg_'
    if show_Vth:
        Vth_row = row
        row += 1
        shows += 'Vth_'
    if show_DeltaVth:
        DeltaVth_row = row
        row += 1
        shows += 'DeltaVth_'
    n_meas = np.sum([len(batch_meas['meas']['hyst']) for batch, batch_meas in meas.items()])
    n_columns = np.sum([sample['cycles'] for batch, batch_meas in meas.items() for _,sample in batch_meas['meas']['hyst'].items()])
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['hyst'].values()})
    total_duts = len(duts)
    plot_file = f'hyst_{shows}{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'hyst')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['hyst'].keys())[0]
            sample = meas[batch]['meas']['hyst'][key]['sample']
            T = meas[batch]['meas']['hyst'][key]['temp']
            plot_file += f'_{sample}_{T}'
            if name:
                plot_file += f'_{name}'
                plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.35
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_height_idvg = 1.0
    inter_space_width = 1.5
    fig, axs = plt.subplots(
        k, n_columns,
        figsize=(n_columns*axis_width + ylabel_space + right_additional_space + (n_meas-1)*inter_space_width, 
                 (k)*axis_height + xlabel_space + title_space + inter_space_height + show_IdVg*inter_space_height_idvg), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(k, n_columns)
    if show_IdVg:
        if show_IdVg_lin:
            axs2 = np.atleast_2d(np.empty_like(axs[idvg_row,:], dtype=object))
    column = 0
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['hyst'].items() if m['dut'] == dut}
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + i*inter_space_width
            bottom = title_space + axis_height
            for row in range(k):
                if show_IdVg and row == 1:
                    axs[row, column].set_position([left/W, 1 - (bottom +row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                else:
                    axs[row, column].set_position([left/W, 1 - (bottom +row*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
                axs[row, column].sharex(axs[row, 0])
                axs[row, column].sharey(axs[row, 0])
            # if show_DeltaVth:
            #     axs[DeltaVth_row, column].set_position([left/W, 1 - (bottom + axis_height + inter_space_height)/H, axis_width/W, axis_height/H])
            #     axs[DeltaVth_row, column].sharex(axs[DeltaVth_row, 0])
            #     axs[DeltaVth_row, column].sharey(axs[DeltaVth_row, 0])
            # if show_IdVg:
            #     axs[idvg_row, column].set_position([left/W, 1 - (bottom - axis_height - inter_space_height_idvg)/H, axis_width/W, axis_height/H])
            for j in range(m['cycles']):
                if show_Vth:
                    axs[Vth_row, column].sharex(axs[Vth_row, 0])
                    axs[Vth_row, column].sharey(axs[Vth_row, 0])
                    if show_DeltaVth:
                        axs[Vth_row, column].tick_params(axis='x', labelbottom=False)
                if show_Vth:
                    axs[DeltaVth_row, column].sharex(axs[DeltaVth_row, 0])
                    axs[DeltaVth_row, column].sharey(axs[DeltaVth_row, 0])
                if show_Vth:
                    if j>0:
                        axs[Vth_row, column].set_position([(left + j*axis_width) /W, 1 - bottom/H, axis_width/W, axis_height/H])
                if column > 0:
                    if show_Vth:
                        axs[Vth_row, column].tick_params(axis='y', labelleft=False)
                    if show_DeltaVth:
                        axs[DeltaVth_row, column].tick_params(axis='y', labelleft=False)
                if show_IdVg:
                    #axs[idvg_row, column].sharex(axs[idvg_row, 0])
                    axs[idvg_row, column].sharey(axs[idvg_row, 0])
                    if show_IdVg_lin:
                        axs2[0, column] = axs[idvg_row,column].twinx()
                        axs2[0, column].set_position(axs[idvg_row, column].get_position())
                        axs2[0, column].sharey(axs2[0, 0])
                        axs2[0, column].tick_params(axis='y', right=False, labelright=False)
                    if j>0:
                        axs[idvg_row, column].set_position([(left + j*axis_width) /W, 1 - (bottom - axis_height - inter_space_height_idvg)/H, axis_width/W, axis_height/H])
                    if column > 0:
                        axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                column += 1
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}.csv'))
        for c in ['Id', 'Vg']:
            if c in df.columns:
                df[c] = df[c].map(json5.loads)
    if meas == None:
        with open(f"hyst_meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    column = 0
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['hyst'].items() if m['dut']['name'] == dut}
        dut_cycles = np.sum([m['cycles'] for m in meas_dut.values()])
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + i*inter_space_width
            bottom = title_space + k*axis_height + (k-1)*inter_space_height_idvg + inter_space_height
            sample = m['sample']
            total_cycle = m['cycles']
            T = m['temp']
            precondition = m['precondition']
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['temp'] == T) &
                    (df['sample'] == sample)
                ]
            cycles = np.arange(1,total_cycle+1)
            for j,cycle in enumerate(cycles):
                cycle_color =plt.cm.viridis(j/np.max(cycles))
                df_meas_cycle = df_meas[df_meas['cycle'] == cycle]
                if precondition and show_precond:
                    offset = df_meas_cycle.loc[df_meas_cycle['precondition'] == True,'sweep_index'].max()
                    df_meas_cycle.loc[df_meas_cycle['precondition'] == False,'sweep_index'] += offset + 1
                    df_meas_hyst = df_meas_cycle[df_meas_cycle['precondition'] == False]
                    df_meas_precond = df_meas_cycle[df_meas_cycle['precondition'] == True]
                else:
                    df_meas_cycle = df_meas_cycle[df_meas_cycle['precondition'] == False]
                    df_meas_hyst = df_meas_cycle
                freqs = df_meas_cycle[df_meas_cycle['freq'] > 0]['freq'].tolist()
                hyst_indexes = df_meas_cycle['sweep_index'].unique().tolist()
                if show_IdVg==True:
                    max_freq = max(freqs)
                    axs[idvg_row,column].set_xlabel(r'$V_\mathsf{G}$ [V]')
                    for l,sweep_index in enumerate(hyst_indexes):
                        row = df_meas_cycle[df_meas_cycle['sweep_index'] == sweep_index].iloc[0]
                        if l== 0:
                            axs[idvg_row,column].axhline(row['Ith'], linestyle='--', color = 'k')
                        freq = row['freq']
                        color_IdVg = plt.cm.plasma((np.log10(freq) - np.log10(min(freqs)))/(np.log10(max_freq) - np.log10(min(freqs)))) if row['precondition'] == False else 'r'
                        axs[idvg_row,column].plot(row['Vg'], row['Id'], '.-', label=rf'$f$ = {freq} Hz',color = color_IdVg)
                        #axs[idvg_row,column].set_xticks([np.min(row['Vg']),np.max(row['Vg'])])
                        axs[idvg_row,column].margins(x=0.10)
                        axs[idvg_row,column].set_yscale('log')
                        # Vth,Ith,extractV,extractI = Vth_extraction(row['Vtg'],row['Id'],row['Vd'],vth_extract='constant_current',current_level=Ith)
                        # axs[idvg_row,column].plot(Vth, Ith, marker='x',linestyle = 'None', color=plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                        if show_IdVg_lin:
                            axs2[0,column].plot(row['Vg'], row['Id'], '.-', label=rf'$f$ = {freq} Hz',color = plt.cm.plasma((np.log10(freq) - np.log10(min(freqs)))/(np.log10(max_freq) - np.log10(min(freqs)))))
                            # axs2[idvg_row,column].plot(Vth, Ith, marker='x', linestyle='None', color=plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                            # if vth_extract == 'linear_extrapolation':
                            #     axs2[idvg_row,column].plot(extractV, extractI, linestyle='--', color = plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                if show_Vth or show_DeltaVth:
                    freq_axis = np.logspace(np.log10(np.min(freqs)), np.log10(np.max(freqs)),500)
                    freq_colors = plt.cm.plasma(np.log10(freq_axis) / np.log10(max(freq_axis)))
                    points = np.array([np.linspace(0, 1, 500), np.full(500, 0)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    if show_Vth:
                        axs[Vth_row,column].plot(df_meas_hyst['freq'], df_meas_hyst['Vth_up'], marker='^',linestyle='None',color = 'C0', label='up sweep')
                        axs[Vth_row,column].plot(df_meas_hyst['freq'], df_meas_hyst['Vth_down'], marker='v',linestyle='None',color = 'C1', label='down sweep')
                        if fit:
                            axs[Vth_row,column].plot(df_meas_hyst['freq'], df_meas_hyst['Vth_up_fit'], linestyle='-', color = 'C0')
                            axs[Vth_row,column].plot(df_meas_hyst['freq'], df_meas_hyst['Vth_down_fit'], linestyle='-', color = 'C1')
                        if precondition and show_precond:
                            # axs[Vth_row,column].plot(df_meas_precond['freq'], df_meas_precond['Vth_up'], marker='^',linestyle='None',color = 'r', alpha = 0.5,label='precond up sweep')
                            # axs[Vth_row,column].plot(df_meas_precond['freq'], df_meas_precond['Vth_down'], marker='v',linestyle='None',color = 'r', alpha = 0.5, label='precond down sweep')
                            decades = 0.2
                            axs[Vth_row, column].boxplot(
                                df_meas_precond['Vth_up'],
                                positions=[df_meas_precond['freq'].iloc[0]],
                                widths=[df_meas_precond['freq'].iloc[0] * (10**(decades / 2) - 1/10**(decades / 2))],  # scale for log axis
                                patch_artist=True,
                                boxprops=dict(facecolor='none', color='C0',linewidth=6),
                                medianprops=dict(color='r', linewidth=6),
                                whiskerprops=dict(color='C0', linewidth=6),
                                capprops=dict(color='C0', linewidth=6),
                                flierprops=dict(marker='v', markersize=20, markerfacecolor='C1', markeredgecolor='k', markeredgewidth=3, alpha=0.5)
                            )
                            axs[Vth_row, column].boxplot(
                                df_meas_precond['Vth_down'],
                                positions=[df_meas_precond['freq'].iloc[0]],
                                widths=[df_meas_precond['freq'].iloc[0] * (10**(decades / 2) - 1/10**(decades / 2))],  # scale for log axis
                                patch_artist=True,
                                boxprops=dict(facecolor='none', color='C1',linewidth=6),
                                medianprops=dict(color='r', linewidth=6),
                                whiskerprops=dict(color='C1', linewidth=6),
                                capprops=dict(color='C1', linewidth=6),
                                flierprops=dict(marker='v', markersize=20, markerfacecolor='C1', markeredgecolor='k', markeredgewidth=3, alpha=0.5)
                            )
                        if column == 0:
                            axs[Vth_row,column].legend(loc = 'best', framealpha=0.0)
                        axs[Vth_row,column].spines['bottom'].set_visible(False)
                        lc = LineCollection(segments,colors=freq_colors[:-1],transform=axs[Vth_row,column].transAxes,linewidth=10)
                        axs[Vth_row,column].add_collection(lc)
                        axs[Vth_row,column].set_xscale('log')
                        axs[Vth_row,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                        axs[Vth_row,column].set_xlabel(r'$f$ [Hz]')
                        axs[DeltaVth_row,column].set_xlim(min(freqs)*0.8, max(freqs)*1.2)
                    if show_DeltaVth:
                        axs[DeltaVth_row,column].plot(df_meas_hyst['freq'], df_meas_hyst['DeltaVth'], marker='o',linestyle='None',label='HW',color = cycle_color)
                        axs[DeltaVth_row,column ].axhline(0, linestyle='--', color = 'k')
                        if fit:
                            axs[DeltaVth_row,column].plot(df_meas_hyst['freq'], df_meas_hyst['DeltaVth_fit'], linestyle='-', color = cycle_color)
                        if precondition and show_precond:
                            decades = 0.2
                            # axs[DeltaVth_row,column].plot(df_meas_precond['freq'], df_meas_precond['DeltaVth'], marker='o',linestyle='None',color = 'r', alpha = 0.5,label='precond HW')
                            axs[DeltaVth_row, column].boxplot(
                                df_meas_precond['DeltaVth'],
                                positions=[df_meas_precond['freq'].iloc[0]],
                                widths=[df_meas_precond['freq'].iloc[0] * (10**(decades / 2) - 1/10**(decades / 2))],  # scale for log axis
                                patch_artist=True,
                                boxprops=dict(facecolor='none', color=cycle_color, linewidth=6),
                                medianprops=dict(color='r', linewidth=6),
                                whiskerprops=dict(color=cycle_color, linewidth=6),
                                capprops=dict(color=cycle_color, linewidth=6),
                                flierprops=dict(marker='o', markersize=20, markerfacecolor=cycle_color, markeredgecolor='k', markeredgewidth=3, alpha=0.5)
                            )
                        axs[DeltaVth_row,column].spines['bottom'].set_visible(False)
                        lc2 = LineCollection(segments,colors=freq_colors[:-1],transform=axs[DeltaVth_row,column].transAxes,linewidth=10)
                        axs[DeltaVth_row,column].add_collection(lc2)
                        axs[DeltaVth_row,column].set_xscale('log')
                        axs[DeltaVth_row,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                        axs[DeltaVth_row,column].set_xlabel(r'$f$ [Hz]')
                        axs[DeltaVth_row,column].set_xlim(min(freqs)*0.5, max(freqs)*2)
                if j == 0:
                    label = f'{dut} {sample} {T}'
                    if 'Vd' in df_meas_hyst.columns:
                        label += '\n' + rf'$V_\mathsf{{D}}$ = {df_meas_hyst["Vd"].iloc[0]} V'
                    if 'Vmin' in df_meas_hyst.columns and 'Vmax' in df_meas_hyst.columns:
                        label += '\n' + rf'$V_\mathsf{{G,range}}$ = [{df_meas_hyst["Vmin"].iloc[0]:.2f}, {df_meas_hyst["Vmax"].iloc[0]:.2f}] V'
                    axs[0,column].text(0.05,0.95, label, horizontalalignment='left', transform=axs[0,column].transAxes, verticalalignment='top', color ='k')
                column +=1
            if show_Vth:
                axs[Vth_row,0].set_ylabel(r'$V_\mathsf{th}$ [V]')
            if show_DeltaVth:
                axs[DeltaVth_row,0].set_ylabel(r'$V_\mathsf{h}$ [V]')
            # if show_DeltaVth or show_Vth:
            #     axs[DeltaVth_row,0].set_xlabel(r'$f$ [Hz]')
            if show_IdVg==True:
                axs[idvg_row,0].set_ylabel(r'$I_\mathsf{D}$ [A]')
    if title:
        if name:
            fig.text(0.5,1 - (title_space/3)/H,rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5,1 - (title_space/3)/H,rf'hyst: {dataset}',ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_hyst_var_freq(dataset, vars, meas = None, df=None, name = None, show_sweep=False, show_precond=False, overwrite=True,title=True,vars_fit=None, plot_format='pdf',transparent=True):
    var_labels = {
        'temp'          : r'T',
        'Eod'          : r'$E_\mathsf{od}$ [MV/cm]',
        'Vth' : r'$V_\mathsf{th}$ [V]',
        'DeltaVth' : r'$\Delta V_\mathsf{th}$ [V]',
        'HW' : r'$V_\mathsf{h}$ [V]',
        'I_max' : r'$I_\mathsf{max}$ [A]',
        'Vg' : r'$V_\mathsf{G}$ [V]',
        'Id': r'$I_\mathsf{D}$ [A]',
        'Vinput': r'$V_\mathsf{input}$ [V]',
        'Voutput'   : r'$V_\mathsf{output}$ [V]',
        'Vm' : r'$V_\mathsf{M}$ [V]',
        'DeltaVm' : r'$\Delta V_\mathsf{M}$ [V]',
    }
    k = sum([show_sweep,len(vars)])
    if vars_fit is None:
        vars_fit = [None]*len(vars)
    row = 0
    shows = ''
    if show_sweep:
        initial_sweep_row = row
        row += 1
        shows += 'sweep_'
    for var in vars:
        # initial_Vth_row = row
        row += 1
        shows += var + '_'
    n_meas = np.sum([len(batch_meas['meas']['hyst']) for batch, batch_meas in meas.items()])
    n_columns = np.sum([sample['cycles'] for batch, batch_meas in meas.items() for _,sample in batch_meas['meas']['hyst'].items()])
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['hyst'].values()})
    total_duts = len(duts)
    plot_file = f'hyst_{shows}{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'hyst')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['hyst'].keys())[0]
            sample = meas[batch]['meas']['hyst'][key]['sample']
            T = meas[batch]['meas']['hyst'][key]['temp']
            plot_file += f'_{sample}_{T}'
            if name:
                plot_file += f'_{name}'
                plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.35
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_height_idvg = 1.0
    inter_space_width = 1.5
    fig, axs = plt.subplots(
        k, n_columns,
        figsize=(n_columns*axis_width + ylabel_space + right_additional_space + (n_meas-1)*inter_space_width, 
                 (k)*axis_height + xlabel_space + title_space + inter_space_height + show_sweep*inter_space_height_idvg), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(k, n_columns)
    column = 0
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['hyst'].items() if m['dut'] == dut}
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + i*inter_space_width
            bottom = title_space + axis_height
            for row in range(k):
                if show_sweep and row == 1:
                    axs[row, column].set_position([left/W, 1 - (bottom +row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                else:
                    axs[row, column].set_position([left/W, 1 - (bottom +row*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
                axs[row, column].sharex(axs[row, 0])
                axs[row, column].sharey(axs[row, 0])
            # if show_DeltaVth:
            #     axs[DeltaVth_row, column].set_position([left/W, 1 - (bottom + axis_height + inter_space_height)/H, axis_width/W, axis_height/H])
            #     axs[DeltaVth_row, column].sharex(axs[DeltaVth_row, 0])
            #     axs[DeltaVth_row, column].sharey(axs[DeltaVth_row, 0])
            # if show_sweep:
            #     axs[idvg_row, column].set_position([left/W, 1 - (bottom - axis_height - inter_space_height_idvg)/H, axis_width/W, axis_height/H])
            for j in range(m['cycles']):
                for row in range(k):
                    if j>0:
                        axs[row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                    axs[row, column].sharey(axs[row, 0])
                    axs[row, column].sharex(axs[row, 0])
                    if column > 0:
                        axs[row, column].tick_params(axis='y', labelleft=False)
                column += 1
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}.csv'))
    for c in ['Id', 'Vg', 'Vinput', 'Voutput']:
        if c in df.columns:
            df[c] = df[c].map(safe_json_load)
    if meas == None:
        with open(f"hyst_meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    column = 0
    sweep_row = initial_sweep_row if show_sweep else None
    main_row = 0
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['hyst'].items() if m['dut']['name'] == dut}
        dut_cycles = np.sum([m['cycles'] for m in meas_dut.values()])
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + i*inter_space_width
            bottom = title_space + k*axis_height + (k-1)*inter_space_height_idvg + inter_space_height
            sample = m['sample']
            total_cycle = m['cycles']
            T = m['temp']
            precondition = m['precondition']
            fit_freq = m['fit_freq'] if 'fit_freq' in m else False
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['temp'] == T) &
                    (df['sample'] == sample)
                ]
            cycles = np.arange(1,total_cycle+1)
            for j,cycle in enumerate(cycles):
                df_meas_cycle = df_meas[df_meas['cycle'] == cycle]
                df_meas_hyst = df_meas_cycle[df_meas_cycle['precondition'] == False]
                df_meas_hyst = df_meas_hyst.sort_values(by='freq')
                freqs = df_meas_hyst[df_meas_hyst['freq'] > 0]['freq'].unique().tolist()
                hyst_indexes = df_meas_hyst['sweep_index'].unique().tolist()
                if show_sweep==True:
                    if 'type' not in m['dut'] or m['dut']['type'] in ['nMOS', 'pMOS']:
                        xsweep_var = 'Vg'
                        ysweep_var = 'Id'
                    elif m['dut']['type'] == 'inverter':
                        xsweep_var = 'Vinput'
                        ysweep_var = 'Voutput'
                    max_freq = max(freqs)
                    axs[sweep_row,column].set_xlabel(r'$V_\mathsf{G}$ [V]')
                    for l,sweep_index in enumerate(hyst_indexes):
                        row = df_meas_hyst[df_meas_hyst['sweep_index'] == sweep_index].iloc[0]
                        freq = row['freq']
                        axs[sweep_row,column].plot(row[xsweep_var], row[ysweep_var], '.-', label=rf'$f$ = {freq} Hz',color = plt.cm.plasma((np.log10(freq) - np.log10(min(freqs)))/(np.log10(max_freq) - np.log10(min(freqs)))))
                        #axs[sweep_row,column].set_xticks([np.min(row['Vg']),np.max(row['Vg'])])
                        axs[sweep_row,column].margins(x=0.10)
                    if ysweep_var == 'Id':
                        axs[sweep_row,column].set_yscale('log')
                    axs[sweep_row,column].set_xlabel(var_labels[xsweep_var])

                for v,var in enumerate(vars):
                    freq_axis = np.logspace(np.log10(np.min(freqs)), np.log10(np.max(freqs)),500)
                    freq_colors = plt.cm.plasma(np.log10(freq_axis) / np.log10(max(freq_axis)))
                    points = np.array([np.linspace(0, 1, 500), np.full(500, 0)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    axs[main_row+v+show_sweep,column].spines['bottom'].set_visible(False)
                    lc = LineCollection(segments,colors=freq_colors[:-1],transform=axs[main_row+v+show_sweep,column].transAxes,linewidth=10)
                    axs[main_row+v+show_sweep,column].add_collection(lc)
                    axs[main_row+v+show_sweep,column].set_xscale('log')
                    axs[main_row+v+show_sweep,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                    axs[main_row+v+show_sweep,column].set_xlabel(r'$f$ [Hz]')
                    if var in ['Vth','Vm']:
                        axs[main_row+v+show_sweep,column].plot(df_meas_hyst['freq'], df_meas_hyst[var + '_up'], marker='^',linestyle='None',color = 'C0', label='up sweep')
                        axs[main_row+v+show_sweep,column].plot(df_meas_hyst['freq'], df_meas_hyst[var + '_down'], marker='v',linestyle='None',color = 'C1', label='down sweep')
                        if column == 0:
                            axs[main_row+v+show_sweep,column].legend(loc = 'best')
                        if fit_freq and var in vars_fit:
                            axs[main_row+v+show_sweep,column].plot(df_meas_hyst['freq'], df_meas_hyst[var + '_up_fit'], linestyle='-', color = 'C0')
                            axs[main_row+v+show_sweep,column].plot(df_meas_hyst['freq'], df_meas_hyst[var + '_down_fit'], linestyle='-', color = 'C1')
                    else:
                        axs[main_row + v+show_sweep,column].plot(df_meas_hyst['freq'], df_meas_hyst[var], marker='o',linestyle='None',label='HW',color =plt.cm.viridis(j/np.max(cycles)))
                        axs[main_row + v+show_sweep,column ].axhline(0, linestyle='--', color = 'k')
                        if fit_freq and var in vars_fit:
                            axs[main_row + v+show_sweep,column].plot(df_meas_hyst['freq'], df_meas_hyst[var + '_fit'], linestyle='-', color =plt.cm.viridis(j/np.max(cycles)))
                if j == 0:
                    label = f'{dut} {sample} {T}'
                    if 'Vd' in df_meas_hyst.columns:
                        label += '\n' + rf'$V_\mathsf{{D}}$ = {df_meas_hyst["Vd"].iloc[0]} V'
                    if 'Vmin' in df_meas_hyst.columns and 'Vmax' in df_meas_hyst.columns:
                        label += '\n' + rf'$V_\mathsf{{G,range}}$ = [{df_meas_hyst["Vmin"].iloc[0]:.2f}, {df_meas_hyst["Vmax"].iloc[0]:.2f}] V'
                    axs[0,column].text(0.05,0.95, label, horizontalalignment='left', transform=axs[0,column].transAxes, verticalalignment='top', color ='k')
                if show_sweep:
                    axs[sweep_row,0].set_ylabel(var_labels[ysweep_var])
                for v, var in enumerate(vars):
                    axs[main_row + v + show_sweep,0].set_ylabel(var_labels[var])
                column +=1
    if title:
        if name:
            fig.text(0.5, 1 - (title_space/3)/H,rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5,1 - (title_space/3)/H,rf'hyst: {dataset}',ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_hyst_Vth_cycle(dataset, meas = None, df=None, name = None, show_IdVg=False, show_IdVg_lin=False,show_Vth=False, show_DeltaVth=True,show_hyst=False, overwrite=True,title=True,fit=False, plot_format='png',transparent=True):
    k = sum([show_IdVg, show_Vth, show_DeltaVth])
    row = 0
    shows = ''
    if show_IdVg:
        idvg_row = row
        row += 1
        shows += 'IdVg_'
    if show_Vth:
        Vth_row = row
        row += 1
        shows += 'Vth_'
    if show_DeltaVth:
        DeltaVth_row = row
        row += 1
        shows += 'DeltaVth_'
    n_meas = np.sum([len(batch_meas['meas']['hyst']) for batch, batch_meas in meas.items()])
    n_columns = np.sum([sample['cycles'] for batch, batch_meas in meas.items() for _,sample in batch_meas['meas']['hyst'].items()])
    duts = list({(batch, m['dut']['name']) for batch, batch_meas in meas.items() for m in batch_meas['meas']['hyst'].values()})
    total_duts = len(duts)
    plot_file = f'hyst_{shows}cycle_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'hyst')
    if total_duts == 1:
        batch = duts[0][0]
        dut = duts[0][1]
        plot_file += f'_{dut}'
        if n_meas == 1:
            key = list(meas[batch]['meas']['hyst'].keys())[0]
            sample = meas[batch]['meas']['hyst'][key]['sample']
            T = meas[batch]['meas']['hyst'][key]['temp']
            plot_file += f'_{sample}_{T}'
            if name:
                plot_file += f'_{name}'
                plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    ylabel_space = 1.6
    right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.35
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_height_idvg = 1.0
    inter_space_width = 1.5
    fig, axs = plt.subplots(
        k, n_columns,
        figsize=(n_columns*axis_width + ylabel_space + right_additional_space + (n_meas-1)*inter_space_width, 
                 (k)*axis_height + xlabel_space + title_space + inter_space_height + show_IdVg*inter_space_height_idvg), # 
        sharey=False,
        sharex=False,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    axs = np.array(axs).reshape(k, n_columns)
    if show_IdVg:
        if show_IdVg_lin:
            axs2 = np.atleast_2d(np.empty_like(axs[idvg_row,:], dtype=object))
    column = 0
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['hyst'].items() if m['dut'] == dut}
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + i*inter_space_width
            bottom = title_space + axis_height
            for row in range(k):
                if show_IdVg and row == 1:
                    axs[row, column].set_position([left/W, 1 - (bottom +row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                else:
                    axs[row, column].set_position([left/W, 1 - (bottom +row*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
            for j,_ in enumerate(m['cycles']):
                #axs[Vth_row, column].sharex(axs[Vth_row, 0])
                if j>0:
                    if show_IdVg:
                        axs[idvg_row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + idvg_row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                    elif show_Vth:
                        axs[Vth_row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + Vth_row*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
                    for row in range(k):
                        if show_IdVg and row == 1:
                            axs[row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + row*(axis_height+inter_space_height_idvg))/H, axis_width/W, axis_height/H])
                        else:
                            axs[row, column].set_position([(left + j*axis_width)/W, 1 - (bottom + row*(axis_height+inter_space_height))/H, axis_width/W, axis_height/H])
                if show_Vth:
                    axs[Vth_row, column].sharey(axs[Vth_row, 0])
                    if column > 0:
                        axs[Vth_row, column].tick_params(axis='y', labelleft=False)
                if show_IdVg:
                    axs[idvg_row, column].sharex(axs[idvg_row, 0])
                    axs[idvg_row, column].sharey(axs[idvg_row, 0])
                    if show_IdVg_lin:
                        axs2[idvg_row, column] = axs[idvg_row,column].twinx()
                        axs2[idvg_row, column].set_position(axs[idvg_row, column].get_position())
                        axs2[idvg_row, column].sharey(axs2[idvg_row, 0])
                        axs2[idvg_row, column].tick_params(axis='y', right=False, labelright=False)
                    if j>0:
                        axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                    if column > 0:
                        axs[idvg_row, column].tick_params(axis='y', labelleft=False)
                column += 1
    column = 0
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}.csv'))
        for c in ['Id', 'Vg']:
            if c in df.columns:
                df[c] = df[c].map(json5.loads)
    if meas == None:
        with open(f"meas_{dataset}.jsonc", "r") as f:
            meas = json5.load(f)
    for i,(batch,dut) in enumerate(duts):
        meas_dut = {key: m for key, m in meas[batch]['meas']['hyst'].items() if m['dut']['name'] == dut}
        dut_cycles = np.sum([m['cycles'] for m in meas_dut.values()])
        for l,m in enumerate(meas_dut.values()):
            left = ylabel_space + column*axis_width + i*inter_space_width
            bottom = title_space + k*axis_height + (k-1)*inter_space_height_idvg + inter_space_height
            sample = m['sample']
            total_cycle = m['cycles']
            T = m['temp']
            precondition = m['precondition']
            if precondition==False and show_hyst==False:
                print(f"Measurement {batch} {dut} {sample} {T} has no precondition data to show.")
                if len(duts) == 1:
                    if len(meas_dut.values()) == 1:
                        plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
                        plt.close()
                        return
                else:
                    continue
            df_meas = df[
                    (df['batch'] == batch) &
                    (df['dut'] == dut) &
                    (df['temp'] == T) &
                    (df['sample'] == sample)
                ]
            cycles = np.arange(1,total_cycle+1)
            for j,cycle in enumerate(cycles):
                df_meas_cycle = df_meas[df_meas['cycle'] == cycle]
                if precondition and show_hyst:
                    offset = df_meas_cycle.loc[df_meas_cycle['precondition'] == True,'sweep_index'].max()
                    df_meas_cycle.loc[df_meas_cycle['precondition'] == False,'sweep_index'] += offset + 1
                elif precondition==False and show_hyst == True:
                    df_meas_cycle = df_meas_cycle[df_meas_cycle['precondition'] == False]
                else:
                    df_meas_cycle = df_meas_cycle[df_meas_cycle['precondition'] == True]
                max_sweep_index = df_meas_cycle['sweep_index'].max()
                # min_sweep_index = df_meas_cycle['sweep_index'].min()
                # for precond in preconditions:
                # df_meas_hyst = df_meas_cycle[df_meas_cycle['precondition'] == precond]
                # freqs = df_meas_hyst[df_meas_hyst['freq'] > 0]['freq'].unique().tolist()
                hyst_indexes = df_meas_cycle['sweep_index'].unique().tolist()
                if show_IdVg==True:
                    for l,sweep_index in enumerate(hyst_indexes):
                        row = df_meas_cycle[df_meas_cycle['sweep_index'] == sweep_index].iloc[0]
                        if l== 0:
                            axs[idvg_row,column].axhline(row['Ith'], linestyle='--', color = 'k')
                        freq = row['freq']
                        axs[idvg_row,column].plot(row['Vg'], row['Id'], '.-', label=rf'$f$ = {freq} Hz',color = plt.cm.plasma(sweep_index/max_sweep_index))
                        # axs[idvg_row,column].plot(Vth, Ith, marker='x',linestyle = 'None', color=plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                        if show_IdVg_lin:
                            axs2[0,column].plot(row['Vg'], row['Id'], '.-', label=rf'$f$ = {freq} Hz',color = plt.cm.plasma(sweep_index/max_sweep_index))
                            # axs2[idvg_row,column].plot(Vth, Ith, marker='x', linestyle='None', color=plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                            # if vth_extract == 'linear_extrapolation':
                            #     axs2[idvg_row,column].plot(extractV, extractI, linestyle='--', color = plt.cm.plasma(np.log10(tRec)/np.log10(max(tRecs))))
                    axs[idvg_row,column].margins(x=0.10)
                    axs[idvg_row,column].set_yscale('log')
                    axs[idvg_row,column].set_xlabel(r'$V_{G}$ [V]')
                if show_Vth or show_DeltaVth:
                    cycle_axis = np.linspace(0, max_sweep_index,500)
                    cycle_colors = plt.cm.plasma(np.log10(cycle_axis) / np.log10(max(cycle_axis)))
                    points = np.array([np.linspace(0, 1, 500), np.full(500, 0)]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    if show_Vth:
                        axs[Vth_row,column].hexbin(df_meas_cycle['sweep_index'], 0.5*(df_meas_cycle['Vth_down'] + df_meas_cycle['Vth_up']), gridsize=25, bins='log', cmap='inferno')
                        #axs[Vth_row,column].plot(df_meas_hyst['sweep_index'], df_meas_hyst['Vth_down'], marker='v',linestyle='None',color = 'C1', label='down sweep')
                        axs[Vth_row,column].spines['bottom'].set_visible(False)
                        lc = LineCollection(segments,colors=cycle_colors[:-1],transform=axs[Vth_row,column].transAxes,linewidth=10)
                        axs[Vth_row,column].add_collection(lc)
                        axs[Vth_row,column].set_xlabel(r'Number of cycles')
                        #axs[Vth_row,column].set_xscale('log')
                        #axs[Vth_row,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                        #if column == 0:
                        #    axs[Vth_row,column].legend(loc = 'best')
                        if fit:
                            axs[Vth_row,column].plot(df_meas_cycle['sweep_index'], df_meas_cycle['Vth_up_fit_cycle'], linestyle='-', color = 'C0')
                            axs[Vth_row,column].plot(df_meas_cycle['sweep_index'], df_meas_cycle['Vth_down_fit_cycle'], linestyle='-', color = 'C1')
                    if show_DeltaVth:
                        axs[DeltaVth_row,column].hexbin(df_meas_cycle['sweep_index'], df_meas_cycle['DeltaVth'], gridsize=25, bins='log', cmap='inferno')
                        axs[DeltaVth_row,column ].axhline(0, linestyle='--', color = 'k')
                        axs[DeltaVth_row,column].spines['bottom'].set_visible(False)
                        lc2 = LineCollection(segments,colors=cycle_colors[:-1],transform=axs[DeltaVth_row,column].transAxes,linewidth=10)
                        axs[DeltaVth_row,column].add_collection(lc2)
                        axs[DeltaVth_row,column].set_xlabel(r'Number of cycles')
                        #axs[DeltaVth_row,column].set_xscale('log')
                        #axs[DeltaVth_row,column].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                        if fit:
                            axs[DeltaVth_row,column].plot(df_meas_cycle['sweep_index'], df_meas_cycle['DeltaVth_fit_cycle'], linestyle='-', color =plt.cm.viridis(j/np.max(cycles)))
                if j == 0:
                    label = f'{dut} {sample} {T}'
                    if 'Vd' in df_meas_cycle.columns:
                        label += '\n' + rf'$V_\mathsf{{D}}$ = {df_meas_cycle["Vd"].iloc[0]} V'
                    if 'Vmin' in df_meas_cycle.columns and 'Vmax' in df_meas_cycle.columns:
                        label += '\n' + rf'$V_\mathsf{{G,range}}$ = [{df_meas_cycle["Vmin"].iloc[0]:.2f}, {df_meas_cycle["Vmax"].iloc[0]:.2f}] V'
                    axs[0,column].text(0.05,0.95, label, horizontalalignment='left', transform=axs[0,column].transAxes, verticalalignment='top', color ='k')
                    #fig.text(left/W,1 - (bottom - k*axis_height - (k-1)*inter_space_height_idvg)/H,label, 
                    #horizontalalignment='left', verticalalignment='bottom', color ='k')
                column +=1
            if show_Vth:
                axs[Vth_row,0].set_ylabel(r'$<V_\mathsf{{th}}> = \frac{V_\mathsf{{th,up}} + V_\mathsf{{th,down}}}{2}$ [V]')
            if show_DeltaVth:
                axs[DeltaVth_row,0].set_ylabel(rf'$V_{{h}}$ [V]')
            if show_IdVg==True:
                axs[idvg_row,0].set_ylabel(r'$I_D$ [A]')
    if title:
        if name:
            fig.text(0.5,1 - (title_space/3)/H,rf'{name}: {dataset}',ha='center', va='center')
        else:
            fig.text(0.5,1 - (title_space/3)/H,rf'hyst: {dataset}',ha='center', va='center')
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def plot_hyst_DeltaVth_vs_var_duts(dataset, varying, grouping=None, df=None, showing=None, multiple=None, filter_plot={}, legend = ['dut','sample'], texts=None, name=None, fit=False, across=None, show_add = False, plot_format='pdf', transparent=True, overwrite=True, title=True):
    fit_labels = {
        'linear'        : r'$\sim a\,t + b$',
        'log'           : r'$\sim A\,\ln\!\left(\dfrac{t}{t_0}\right) + C$',
        'powerlaw'      : r'$\sim A\,t^{n} + C$',
        'stretched_exp' : r'$\sim A\!\left[1 - \exp\!\left(-\left(\dfrac{t}{\tau}\right)^{\beta}\right)\right] + C$'
    }
    var_labels = {
        'Vd'        : r'$V_d$ [V]',
        'temp'          : r'T',
        'Eod'          : r'$E_{od}$ [MV/cm]',
        'Vth_initial' : r'$V_{th,initial}$ [V]',
        'freq' : r'$f$ [Hz]'
    }
    if show_add:
        k = 2
    else:
        k = 1
    plot_file = f'hyst_DeltaVth_{dataset}'
    plot_folder = os.path.join(script_dir,'plots', dataset, 'hyst')
    if grouping:
        plot_file += '_grouping'
        for grouping_var in grouping:
            plot_file += f'_{grouping_var}'
            if grouping_var in filter_plot:
                plot_file += f'_{filter_plot[grouping_var]}'
    if showing:
        plot_file += '_showing'
        for showing_var in showing:
            plot_file += f'_{showing_var}'
            if showing_var in filter_plot:
                plot_file += f'_{filter_plot[showing_var]}'
    if name:
        plot_file += f'_{name}'
        plot_folder = os.path.join(plot_folder, name)
    if overwrite == False and os.path.exists(os.path.join(plot_folder, f'{plot_file}.{plot_format}')):
        print(f"Plot {plot_file}.{plot_format} already exists. Skipping...")
        return
    if df is None:
        if name is None:
            df_Vth_meas = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}.csv'))
        else:
            df_Vth_meas = pd.read_csv(os.path.join(script_dir,'data',dataset,f'hyst_{dataset}_{name}.csv'))
    else:
        df_Vth_meas = df
    for variable,plot in filter_plot.items():
        df_Vth_meas = df_Vth_meas[df_Vth_meas[variable].isin(plot)]
    df_Vth_subset = df_Vth_meas
    groups = (df_Vth_subset.groupby(grouping) if grouping else [(None, df_Vth_subset)])
    if across:
        n_rows = sum(subset.groupby(across).ngroups > 1 for _, subset in groups)
        if grouping:
            df_Vth_subset = df_Vth_subset.groupby(grouping).filter(
                    lambda subset: subset.groupby(across).ngroups > 1
                )
    else:
        n_rows = df_Vth_subset.groupby(grouping).ngroups if grouping else 1
    if n_rows == 0:
        print(f'No sufficient data to plot DeltaVth vs {varying} for multiple {across} under the chosen conditions.')
        return
    ylabel_space = 1.6
    if legend:
        right_additional_space = 10
    else:
        right_additional_space = 0.1
    axis_width = 8.5
    axis_height = 8.5
    if title:
        title_space = 0.35
    else:
        title_space = 0.1
    xlabel_space = 1
    inter_space_height = 1.5
    inter_space_width = 1.5
    fig, axs = plt.subplots(
        n_rows, k,
        figsize=(k*axis_width + ylabel_space + right_additional_space + (k-1)*inter_space_width, 
                 axis_height*n_rows + xlabel_space + title_space + (n_rows-1)*inter_space_height), # 
        sharey=False,
        sharex=True,
        constrained_layout=False
    )
    W, H = fig.get_size_inches()
    plt.subplots_adjust(top = 1 - title_space/H, bottom=xlabel_space/H, left=ylabel_space/W, right=1 - right_additional_space/W)
    plt.subplots_adjust(wspace=inter_space_width/axis_width, hspace=inter_space_height/axis_height)
    axs = np.array(axs).reshape(n_rows, k)
    markers = ['o','v','^','<','>','s','p','*','h','H','+','x','D','d','|','_','1','2','3','4','.',',']
    all_samples = df_Vth_subset[['batch','dut','sample']].drop_duplicates()
    all_samples = [tuple(x) for x in all_samples.values]
    global_marker_map = {
        combo: markers[i % len(markers)]
        for i, combo in enumerate(all_samples)
    }
    if showing:
        all_showed = df_Vth_subset[showing].drop_duplicates()#.sort_values(by=showing[0])
        all_showed = list(map(tuple, all_showed.to_numpy()))
        colors = plt.cm.viridis(np.linspace(0.1,0.9,len(all_showed)))
        global_color_map = {
            combo: colors[i % len(colors)]
            for i, combo in enumerate(all_showed)
        }
    else:
        colors = plt.cm.tab10(np.linspace(0,1,10))
    edge_colors = colors
    i=0
    marker_handles = []
    group_map = {}
    for group_elements, subset in groups:
        group_text = ''
        for elem in grouping or []:
            if elem in var_labels.keys():
                group_text += var_labels[elem] + f'= {subset[elem].iloc[0]}; '
            else:
                group_text += f'{subset[elem].iloc[0]}'
        axs[i,0].text(0.05,0.95, group_text,
                horizontalalignment='center', verticalalignment='center', transform=axs[i,0].transAxes, color ='k')
        # fig.text((ylabel_space+(k*(axis_width)+(k-1)*inter_space_width)*0.5)/W,1-(title_space + i*(axis_height+inter_space_height))/H,group_text, 
        #         horizontalalignment='center', verticalalignment='bottom', color ='k')
        sublabel_handles = []
        groups_showing = [(None, subset)] if showing is None else subset.groupby(showing)
        for group_showing, subset_showing in groups_showing:
            for (batch,dut,sample), subset_sample in subset_showing.groupby(['batch', 'dut','sample']):
                var_plot = subset_sample[varying]
                marker = global_marker_map[(batch, dut, sample)]
                label = ''
                for j,elem in enumerate(legend):
                    if elem in var_labels.keys():
                        label += var_labels[elem] + f'= {subset_sample[elem].iloc[0]} '
                    else:
                        label += f'{subset_sample[elem].iloc[0]} '
                if (showing is not None):
                    color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
                else:
                    color = colors[i%len(colors)]
                if (multiple is not None) and (subset_sample.groupby(multiple).ngroups > 1):
                    l = 0
                    for mul,subset_mul in subset_sample.groupby(multiple):
                        label1 = label + f' {var_labels[multiple]}={mul}'
                        var_plot = subset_mul[varying].to_numpy()
                        DeltaVth = subset_mul['DeltaVth'].to_numpy()
                        idx = np.argsort(var_plot)
                        var_plot = var_plot[idx]
                        DeltaVth = DeltaVth[idx]
                        axs[i,0].plot(var_plot, DeltaVth, marker=marker, linestyle='None', color=color, label=label1)
                        if fit:
                            DeltaVth_fit = subset_mul['DeltaVth_fit'].to_numpy()
                            DeltaVth_fit = DeltaVth_fit[idx]
                            axs[i,0].plot(var_plot, DeltaVth_fit, linestyle='-', color=color)
                        if show_add == 'DeltaVth/tox':
                            tox = subset_mul['tox'].iloc[0]
                            axs[i,k-1].plot(var_plot, DeltaVth/tox, marker=marker, linestyle='None', color=color)
                            if fit:
                                axs[i,k-1].plot(var_plot, DeltaVth_fit/tox, linestyle='-', color=color)
                        if not any(h.get_label() == label for h in marker_handles):
                            marker_handles.append(Line2D([], [],marker=marker,linestyle='None',facecolors=color,edgecolors=edge_colors[l],label=label1))
                            l += 1
                else:
                    var_plot = subset_sample[varying].to_numpy()
                    DeltaVth = subset_sample['DeltaVth'].to_numpy()
                    idx = np.argsort(var_plot)
                    var_plot = var_plot[idx]
                    DeltaVth = DeltaVth[idx]
                    axs[i,0].plot(var_plot, DeltaVth, marker=marker, linestyle='None', markerfacecolor=color, alpha=1)
                    if show_add == 'DeltaVth/tox':
                        tox = subset_sample['tox'].iloc[0]
                        axs[i,k-1].plot(var_plot, DeltaVth/tox, marker=marker, linestyle='None', markerfacecolor = color, alpha=1)
                        if fit:
                            DeltaVth_fit = subset_sample['DeltaVth_fit'].to_numpy()
                            axs[i,k-1].plot(var_plot, DeltaVth_fit/tox, linestyle='-', color=color)
                    if texts:
                        sublabel = ''
                        for j,elem in enumerate(texts):
                            if elem in var_labels.keys():
                                sublabel += var_labels[elem] + f'= {subset_sample[elem].iloc[0]:.2f}'
                            else:
                                sublabel += f'{subset_sample[elem].iloc[0]}'
                        sublabel_handles.append(Line2D([], [],marker=marker,linestyle='None',markerfacecolor=color,label=sublabel))
                        #axs[i,0].text(var_plot[len(var_plot) // 2], DeltaVth[len(var_plot) // 2], text, horizontalalignment='center', 
                        #    verticalalignment='top', color= color)
                    if not any(h.get_label() == label for h in marker_handles):
                        marker_handles.append(Line2D([], [], marker=marker, linestyle='None', markerfacecolor=color, alpha=1,label=label))
        if texts:
            leg = axs[i,0].legend(handles=sublabel_handles, ncol=1,loc = 'lower left', handlelength=0.5, framealpha=0.1, fontsize=18)
            axs[i,0].add_artist(leg)
        axs[i,0].axhline(0, linestyle='--', color = 'k')
        axs[i,0].set_ylabel(rf'$\Delta V_{{th}}$ [V]')
        ymin, ymax = axs[i,0].get_ylim()
        axs[i,0].set_ylim(ymin, ymax)
        axs[i,0].sharey(axs[0, 0])
        if show_add == 'DeltaVth/tox':
            axs[i,k-1].axhline(0, linestyle='--', color = 'k')
            axs[i,k-1].set_ylabel(rf'$\Delta V_{{th}}/t_{{ox}}$ [V/nm]')
            ymin, ymax = axs[i,k-1].get_ylim()
            axs[i,k-1].set_ylim(ymin, ymax)
            axs[i, k-1].sharey(axs[0,k-1])
            leg1 = axs[i,k-1].legend(handles=sublabel_handles, ncol=1,loc = 'lower left', handlelength=0.5, framealpha=0.1)
            axs[i,k-1].add_artist(leg1) 
        #axs[i,0].sharex(axs[0,0])
        #axs[i,0].sharey(axs[0,0])
        if varying == 'freq':
            axs[i,0].set_xscale('log')
            axs[i,0].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
            if show_add:
                axs[i,k-1].set_xscale('log')
                axs[i,k-1].xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        axs[i,0].set_xlabel(var_labels[varying])
        if show_add:
            axs[i,k-1].set_xlabel(var_labels[varying])
        group_map[group_elements] = i
        i+=1
    # if fit:
    #     for group_elements, subset_fit in groups_fit:
    #         i = group_map[group_elements]
    #         for group_showing, subset_showing in subset_fit.groupby(showing):
    #             for group_sample, subset_sample in subset_showing.groupby(['batch','dut','sample']):
    #                 var_plot_fit = subset_sample[varying].to_numpy()
    #                 DeltaVth_fit = subset_sample['DeltaVth_fit'].to_numpy()
    #                 idx = np.argsort(var_plot_fit)
    #                 DeltaVth_fit = DeltaVth_fit[idx]
    #                 var_plot_fit = var_plot_fit[idx]
    #                 fit_type = subset_sample['fit'].iloc[0]
    #                 color = global_color_map[group_showing if isinstance(group_showing, tuple) else (group_showing,)]
    #                 axs[i,0].plot(var_plot_fit, DeltaVth_fit, linestyle='-', color=color)
    #                 if texts == 'fit':
    #                     axs[i,0].text(var_plot_fit[len(var_plot_fit) // 2], DeltaVth_fit[len(var_plot_fit) // 2], f'{fit_labels[fit_type]}', horizontalalignment='center', 
    #                             verticalalignment='top', color= color)
    
    # Combine both legends
    all_handles = marker_handles
    axs[0,k-1].legend(handles=all_handles, ncol=1,loc = 'upper left',bbox_to_anchor=(1.05, 1), handlelength=1.0, framealpha=1)
    fig.text(0.5, 1 - (title_space/2)/H, f'{name}: {dataset}' if name else f'{dataset}', ha='center', va='center')
    plt.savefig(os.path.join(plot_folder,f'{plot_file}.{plot_format}'), bbox_inches=None, transparent=transparent)
    plt.close()

def fit_data_time(time,data,time_fit,fit):
    time = np.array(time, dtype=float)
    data = np.array(data, dtype=float)
    idx = np.argsort(time)
    time_sorted = time[idx]
    data_sorted = data[idx]
    mask = (~np.isnan(data_sorted)) & (time_sorted >= 1)
    time_clean = time_sorted[mask]
    data_clean = data_sorted[mask]
    if fit == 'linear':
        popt, pcov = optimize.curve_fit(linear_fit, time_clean, data_clean)
        data_fit = linear_fit(time_fit, *popt)
    elif fit == 'powerlaw':
        popt, pcov = optimize.curve_fit(powerlaw_fit, time_clean, data_clean,
                                p0=powerlaw_initial_guess(time_clean, data_clean),
                                bounds=([-np.inf, -1, -np.inf],[np.inf, 1, np.inf]),
                                maxfev=20000)
        data_fit = powerlaw_fit(time_fit, *popt)
    elif fit == 'log':
        popt, pcov = optimize.curve_fit(log_fit, time_clean, data_clean)
        data_fit = log_fit(time_fit, *popt)
    elif fit == 'stretched_exp':
        popt, pcov = optimize.curve_fit(stretched_exp_fit, time_clean, data_clean, 
                    p0=stretched_exp_initial_guess(time_clean, data_clean), 
                    bounds=([-np.inf, 0, 0.1, -np.inf],[np.inf, np.inf, 1.0, np.inf]),
                    maxfev=20000
        )
        data_fit = stretched_exp_fit(time_fit, *popt)
    elif fit == 'spline':
        spline_fit = interpolate.UnivariateSpline(time_clean, data_clean, s=0)
        data_fit = spline_fit(time_fit)
    elif fit == 'best':
        fits = ['linear', 'powerlaw', 'log', 'stretched_exp']
        fit = None
        best_residual = np.inf

        for fit_type in fits:
            try:
                if fit_type == 'linear':
                    popt, pcov = optimize.curve_fit(linear_fit, time_clean, data_clean)
                    data_fit_temp = linear_fit(time, *popt)
                elif fit_type == 'powerlaw':
                    popt, pcov = optimize.curve_fit(powerlaw_fit, time_clean, data_clean,
                                p0=powerlaw_initial_guess(time_clean, data_clean),
                                bounds=([-np.inf, -1, -np.inf],[np.inf, 1, np.inf]),
                                maxfev=20000)
                    data_fit_temp = powerlaw_fit(time, *popt)
                elif fit_type == 'log':
                    popt, pcov = optimize.curve_fit(log_fit, time_clean, data_clean)
                    data_fit_temp = log_fit(time, *popt)
                elif fit_type == 'stretched_exp':
                    popt, pcov = optimize.curve_fit(stretched_exp_fit, time_clean, data_clean,
                                p0=stretched_exp_initial_guess(time_clean, data_clean),
                                bounds=([-np.inf, 0, 0.1, -np.inf],[np.inf, np.inf, 1.0, np.inf]),
                                maxfev=20000
                    )
                    data_fit_temp = stretched_exp_fit(time, *popt)
                elif fit_type == 'spline':
                    spline_fit_temp = interpolate.UnivariateSpline(time_clean, data_clean, s=0)
                    data_fit_temp = spline_fit_temp(time)
                
                # Calculate residual
                residual = np.sum((data_clean - np.interp(time_clean, time, data_fit_temp))**2)
                
                if residual < best_residual:
                    best_residual = residual
                    fit = fit_type
            except:
                continue
        
        data_fit,_ = fit_data_time(time,data,time_fit,fit)

    return data_fit,fit

def fit_data_freq(freq,data,freq_fit,fit):
    freq = np.asarray(freq)
    data = np.asarray(data)
    idx = np.argsort(freq)
    freq = freq[idx]
    data = data[idx]
    mask = ~np.isnan(data) & (freq >= 0)
    freq_clean = freq[mask]
    data_clean = data[mask]
    if fit == 'linear':
        popt, pcov = optimize.curve_fit(linear_fit, freq_clean, data_clean)
        data_fit = linear_fit(freq_fit, *popt)
    elif fit == 'powerlaw':
        popt, pcov = optimize.curve_fit(powerlaw_fit, freq_clean, data_clean,
                                p0=powerlaw_initial_guess(freq_clean, data_clean),
                                bounds=([-np.inf, -1, -np.inf],[np.inf, 1, np.inf]),
                                maxfev=20000)
        data_fit = powerlaw_fit(freq_fit, *popt)
    elif fit == 'log':
        popt, pcov = optimize.curve_fit(log_fit, freq_clean, data_clean)
        data_fit = log_fit(freq_fit, *popt)
    elif fit == 'stretched_exp':
        popt, pcov = optimize.curve_fit(stretched_exp_fit, freq_clean, data_clean, 
                    p0=stretched_exp_initial_guess(freq_clean, data_clean), 
                    bounds=([-np.inf, 0, 0.1, -np.inf],[np.inf, np.inf, 1.0, np.inf]),
                    maxfev=20000
        )
        data_fit = stretched_exp_fit(freq_fit, *popt)
    elif fit == 'spline':
        polyfit = np.polyfit(np.log10(freq_clean), data_clean, 3)
        func_polyfit= np.poly1d(polyfit)
        data_fit = func_polyfit(np.log10(freq_fit))
    elif fit == 'best':
        fits = ['linear', 'powerlaw', 'log', 'stretched_exp']
        fit = None
        best_residual = np.inf
        best_params = None
        best_data_fit = None
        
        for fit_type in fits:
            try:
                if fit_type == 'linear':
                    popt, pcov = optimize.curve_fit(linear_fit, freq_clean, data_clean)
                    data_fit_temp = linear_fit(freq_fit, *popt)
                elif fit_type == 'powerlaw':
                    popt, pcov = optimize.curve_fit(powerlaw_fit, freq_clean, data_clean,
                                p0=powerlaw_initial_guess(freq_clean, data_clean),
                                bounds=([-np.inf, -1, -np.inf],[np.inf, 1, np.inf]),
                                maxfev=20000)
                    data_fit_temp = powerlaw_fit(freq_fit, *popt)
                elif fit_type == 'log':
                    popt, pcov = optimize.curve_fit(log_fit, freq_clean, data_clean)
                    data_fit_temp = log_fit(freq_fit, *popt)
                elif fit_type == 'stretched_exp':
                    popt, pcov = optimize.curve_fit(stretched_exp_fit, freq_clean, data_clean,
                                p0=stretched_exp_initial_guess(freq_clean, data_clean),
                                bounds=([-np.inf, 0, 0.1, -np.inf],[np.inf, np.inf, 1.0, np.inf]),
                                maxfev=20000
                    )
                    data_fit_temp = stretched_exp_fit(freq_fit, *popt)
                elif fit_type == 'spline':
                    spline_fit_temp = interpolate.UnivariateSpline(freq_clean, data_clean, s=0)
                    data_fit_temp = spline_fit_temp(freq_fit)
                
                # Calculate residual
                residual = np.sum((data_clean - np.interp(freq_clean, freq_fit, data_fit_temp))**2)
                
                if residual < best_residual:
                    best_residual = residual
                    fit = fit_type
                    best_params = popt if fit_type != 'spline' else None
                    best_data_fit = data_fit_temp
            except:
                continue
        
        data_fit,_ = fit_data_freq(freq_clean,data,freq_fit,fit)

    return data_fit,fit

def fit_data_IdVg(Id,Vg,Id_fit_in,Vg_fit_in,fit_IdVg,noise_level=1e-12,temp=300):
    Vg = np.array(Vg)
    Vg_fit_in = np.array(Vg_fit_in)
    Id_fit_in = np.array(Id_fit_in)
    Id = np.array(Id)
    # Remove data below noise level
    mask = Id >= noise_level
    Vg = Vg[mask]
    Id = Id[mask]
    mask = Id_fit_in >= noise_level
    Id_fit_in = Id_fit_in[mask]
    if fit_IdVg == 'lambert':

        k = 1.380649e-23
        q = 1.602176634e-19
        Vt = k*temp/q   # thermal voltage

        # --- Initial guesses ---
        I0_guess = np.min(Id)
        theta_guess = 0.01
        K_guess = 1e-6
        n_guess = 1.5

        p0 = [I0_guess, theta_guess, K_guess, n_guess, Vt]

        lambert = lambda Id, I0, theta, K, n, Vt: Vg_model(Id, n, I0, K, theta, Vt)

        popt, _ = optimize.curve_fit(
            lambert,
            Vg,
            Id,
            p0=p0,
            maxfev=40000
        )

        Vg_fit = lambert(Id_fit_in, *popt)
        Id_fit = Id_fit_in
    return Id_fit,Vg_fit, fit_IdVg, popt

def fit_data_invtransfer(Vinput,Voutput,Vinput_fit,fit_invtransfer,Vdd=1.0,noise_level=1e-12):
    Vinput = np.array(Vinput)
    Voutput = np.array(Voutput)
    Vinput_fit = np.array(Vinput_fit)
    # Remove data below noise level
    mask = Voutput >= noise_level
    Vinput = Vinput[mask]
    Voutput = Voutput[mask]
    if fit_invtransfer == 'polynomial':
        degree = 3
        popt = np.polyfit(Vinput, Voutput, degree)
        poly_fit = np.poly1d(popt)
        Voutput_fit = poly_fit(Vinput_fit)
    elif fit_invtransfer == 'logistic':
        p0 = [np.percentile(Voutput,5), np.percentile(Voutput,95), np.median(Vinput), 1]
        popt, _ = optimize.curve_fit(logistic_fit, Vinput, Voutput, p0=p0, maxfev=20000)
        Voutput_fit = logistic_fit(Vinput_fit, *popt)
    elif fit_invtransfer == 'spline':
        if len(Vinput) > 3:
            spline_fit = interpolate.UnivariateSpline(
                Vinput, Voutput, s=0.001*len(Vinput)
            )
            Voutput_fit = spline_fit(Vinput_fit)
        else:
            spline_fit = None    
            Voutput_fit = None
        popt = None
    elif fit_invtransfer == 'invtransfer_model':
        def model_fit(Vin, Vth, Vx):
            return loadinv_model(Vin, Vth, Vx, Vdd)
        p0 = [0.3*Vdd, 0.1*Vdd]  # Initial guess for Vth and Vx
        popt, _ = optimize.curve_fit(model_fit, Vinput, Voutput, p0=p0, bounds=([1e-6, 1e-6], [Vdd-1e-6, 2*Vdd-1e-6]),maxfev=20000)
        Voutput_fit = loadinv_model(Vinput_fit, *popt, Vdd)
    return Vinput_fit,Voutput_fit, fit_invtransfer,popt

def Vth_extraction(Vg,Id,Vd,vth_extract,current_level,width=1,length=1, noise_level=1e-12):
    Vg = np.array(Vg)
    Id = np.array(Id)
    Vd = np.array(Vd)
    # Remove data below noise level
    mask = Id >= noise_level
    Vg = Vg[mask]
    Id = Id[mask]
    if vth_extract == 'constant_current':
        intersection_x, intersection_y, Interpolator = InterpolationHelper.get_abscissa(
            Vg, Id, current_level,ymin=current_level/10, ymax=current_level*10, scale="lin-log", interpolator_function=None)
        Vth = float(intersection_x)
        Idth = float(intersection_y)
        extract_curve_Vg = Vg
        extract_curve_Id = current_level * np.ones_like(Vg)
    elif vth_extract == 'linear_extrapolation':
        dId_dVg = np.gradient(Id, Vg)
        max_slope_index = np.argmax(dId_dVg)
        slope = dId_dVg[max_slope_index]
        abcisas = Id[max_slope_index] - slope * Vg[max_slope_index]
        V_zerocurrent = -abcisas / slope
        Vth = V_zerocurrent - np.mean(Vd)/2
        intersection_x, intersection_y, Interpolator = InterpolationHelper.get_ordinate(
            Vg, Id, Vth, scale="lin-lin", interpolator_function=None)
        Idth = float(intersection_y)
        extract_curve_Vg = Vg
        extract_curve_Id = slope * (Vg - V_zerocurrent)
        mask = extract_curve_Id > 0
        extract_curve_Vg = extract_curve_Vg[mask]
        extract_curve_Id = extract_curve_Id[mask]
    elif vth_extract == 'SS_deviation':
        log_Id = np.log10(np.abs(Id) + 1e-12)
        dId_dVg = np.gradient(log_Id, Vg)
        SS = 1 / dId_dVg
        SS_ideal = 60e-3  # Ideal subthreshold swing in V/decade
        deviation_index = np.where(SS > SS_ideal)[0][0]
        Vth = Vg[deviation_index]
    elif vth_extract == 'maxcurrent/10':
        max_Id = np.max(np.abs(Id))
        target_current = max_Id / 10
        intersection_x, intersection_y, Interpolator = InterpolationHelper.get_abscissa(
            Vg, Id, target_current,ymin=target_current/10, ymax=target_current*10, scale="lin-log", interpolator_function=None)
        Vth = float(intersection_x)
        Idth = float(intersection_y)
        extract_curve_Vg = Vg
        extract_curve_Id = target_current * np.ones_like(Vg)
    elif vth_extract == 'constant_current_L/W':
        intersection_x, intersection_y, Interpolator = InterpolationHelper.get_abscissa(
            Vg, Id*length/width, current_level,ymin=current_level/10, ymax=current_level*10, scale="lin-log", interpolator_function=None)
        Vth = float(intersection_x)
        Idth = current_level*width/length
        extract_curve_Vg = Vg
        extract_curve_Id = current_level * np.ones_like(Vg)
    else:
        raise ValueError(f"Unknown Vth extraction method: {vth_extract}")
    return Vth,Idth,extract_curve_Vg,extract_curve_Id

def SS_extraction(Vg,Id,Vd,ss_extract,noise_level,order,width=1,length=1):
    Vg = np.array(Vg)
    Id = np.array(Id)
    # Remove data below noise level
    mask = Id >= noise_level
    Vg = Vg[mask]
    Id = Id[mask]
    Vd = np.array(Vd)
    if ss_extract == 'max_derivative':
        logId = np.log10(Id)
        dlogId_dVg = np.gradient(logId, Vg)
        max_slope_index = np.argmax(dlogId_dVg)
        slope = dlogId_dVg[max_slope_index]
        SS = 1/slope
        abcisas = logId[max_slope_index] - slope * Vg[max_slope_index]
        V_zerocurrent = -abcisas / slope
        extract_curve_Vg = Vg
        extract_curve_Id = 10**(slope * (Vg - V_zerocurrent))
        mask = extract_curve_Id > 0
        extract_curve_Vg = extract_curve_Vg[mask]
        extract_curve_Id = extract_curve_Id[mask]
    elif ss_extract == 'orders_above_noise':
        logId = np.log10(Id)
        # Define current range: from noise_level to 2 orders of magnitude above
        current_upper = noise_level * 10**order  # 2 orders of magnitude above noise level
        mask_range = (Id >= noise_level) & (Id <= current_upper)
        
        if np.sum(mask_range) < 2:
            print("Insufficient data points in the specified current range for SS extraction")
            return None, None, None, None
        
        Vg_fit = Vg[mask_range]
        logId_fit = logId[mask_range]
        
        # Linear fit in log-lin space: log(Id) vs Vg
        popt, _ = optimize.curve_fit(linear_fit, Vg_fit, logId_fit)
        slope = popt[0]
        intercept = popt[1]
        
        # SS = dVg/d(logId) = 1/slope [V/decade]
        SS = 1.0 / slope
        
        # Extract the line equation to find zero current intercept
        # log(Id) = slope * Vg + intercept
        # Id = 0 when log(Id) -> -inf, so we use the linear extrapolation
        V_zerocurrent = -intercept / slope
        
        # Generate extraction curve
        extract_curve_Vg = Vg
        extract_curve_Id = 10**(slope * Vg + intercept)
        mask = extract_curve_Id > 0
        extract_curve_Vg = extract_curve_Vg[mask]
        extract_curve_Id = extract_curve_Id[mask]
    else:
        raise ValueError(f"Unknown SS extraction method: {ss_extract}")
    return SS,V_zerocurrent,extract_curve_Vg,extract_curve_Id

## Miscellaneuous functions

def filter_meas(meas, meas_type=None, dut_filters=None, meas_filters=None, keys=None):
    """
    Filter measurement dictionary.

    Supports:
    - equality: {"param": value}
    - list membership: {"param": [v1, v2]}
    - negative filters: {"param": {"not": value}}
    - existence: {"param": {"exists": True/False}}
    - nested filters (recursive)
    """

    def match(val, cond):
        # negative filter
        if isinstance(cond, dict) and "not" in cond:
            nv = cond["not"]
            if isinstance(nv, (list, tuple, set)):
                return val not in nv
            return val != nv

        # list filter
        if isinstance(cond, (list, tuple, set)):
            return val in cond

        # equality
        return val == cond

    def match_dict(data, cond):
        """
        Recursively match nested dictionaries with existence support.
        """
        for k, v in cond.items():

            # --- EXISTS filter ---
            if isinstance(v, dict) and "exists" in v:
                exists = v["exists"]

                if exists and k not in data:
                    return False
                if not exists and k in data:
                    return False

                continue

            # --- nested dictionary ---
            if isinstance(v, dict) and "not" not in v:
                sub_data = data.get(k)
                if not isinstance(sub_data, dict):
                    return False
                if not match_dict(sub_data, v):
                    return False

            # --- standard match ---
            else:
                if not match(data.get(k), v):
                    return False

        return True

    meas_f = copy.deepcopy(meas)

    if isinstance(meas_type, str):
        meas_type = [meas_type]

    if isinstance(keys, str):
        keys = [keys]

    for batch in meas_f.values():

        meas_all = batch.get("meas", {})

        for mt in list(meas_all.keys()):

            # --- filter by measurement type ---
            if meas_type is not None and mt not in meas_type:
                del meas_all[mt]
                continue

            meas_dict = meas_all[mt]
            filtered = {}

            for k, v in meas_dict.items():

                if keys and k not in keys:
                    continue

                keep = True

                # --- DUT filters ---
                if dut_filters:
                    dut = v.get("dut", {})
                    if not match_dict(dut, dut_filters):
                        keep = False

                # --- Measurement filters ---
                if keep and meas_filters:
                    if not match_dict(v, meas_filters):
                        keep = False

                if keep:
                    filtered[k] = v

            meas_all[mt] = filtered

    return meas_f

def data_average(average_vars, average_over, dataset, meas_type, fit_data_t, fit_var_t, df=None, name=None):
    '''Averages data over specified dimensions and optionally fits vs time.
        - average_vars: list of variables to average (e.g. ["DeltaVth"])
        - average_over: list of dimensions to average over (e.g. ["batch", "dut", "sample"])
        - dataset: dataset name (used for loading/saving)
        - meas_type: measurement type (used for loading/saving)
        - fit_data_t: list of variables to fit vs tRec (e.g. ["DeltaVth"])
        - df: optional dataframe to use instead of loading from file
        - name: optional name for loading/saving (e.g. "fit_t")
    '''
    if df is None:
        if name:
            df = pd.read_csv(os.path.join(script_dir, 'data', dataset, f'{meas_type}_{dataset}_{name}.csv'))
        else:
            df = pd.read_csv(os.path.join(script_dir, 'data', dataset, f'{meas_type}_{dataset}.csv'))

    for var in average_vars:
        if var not in df.columns:
            raise ValueError(f"{var} not found in dataframe.")

    for col in average_over:
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe.")

    # Define grouping columns (everything except averaged dimensions & averaged vars)
    group_cols = [col for col in df.columns if col not in average_over and col not in average_vars]

    agg_dict = {}
    for var in average_vars:
        agg_dict[f"{var}_mean"] = (var, "mean")
        agg_dict[f"{var}_std"]  = (var, "std")

    agg_dict["n_samples"] = (average_vars[0], "count")

    # Perform aggregation
    df_average = (
        df
        .groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )

    # Save
    # if name is None:
    #     filename = f'{meas_type}_average_{dataset}.csv'
    # else:
    #     filename = f'{meas_type}_average_{dataset}_{name}.csv'

    # df_average.to_csv(
    #     os.path.join(script_dir, 'data', filename),
    #     index=False
    # )
    
    # df_fit = df_average.copy()

    if fit_data_t:   # list of variables to fit vs tRec

        if fit_var_t not in df_average.columns:
            raise ValueError(f"{fit_var_t} not found in df_average.")

        # Validate variables
        for var in fit_data_t:
            if f"{var}_mean" not in df_average.columns:
                raise ValueError(f"{var}_mean not found in df_average.")

        # Physical condition columns (exclude aggregated columns + tRec)
        fit_group_cols = [
            col for col in df_average.columns
            if col != "tRec"
            and not col.endswith("_mean")
            and not col.endswith("_std")
            and col != "n_samples"
        ]

        for keys, df_group in df_average.groupby(fit_group_cols):

            df_group = df_group.sort_values(fit_var_t)
            idx = df_group.index

            time = df_group[fit_var_t].values
            y = df_group[f"{var}_mean"].values

            if len(time) < 2:
                continue

            data_fit, fit_params = fit_data_time(time, y, time, fit='powerlaw')

            df_average.loc[idx, f"{var}_fit"] = data_fit

        # Save
        if name is None:
            filename = f'{meas_type}_average_{dataset}.csv'
            folder = os.path.join(script_dir, 'data', dataset)
        else:
            filename = f'{meas_type}_average_{dataset}_{name}.csv'
            folder = os.path.join(script_dir, 'data', dataset, name)

        df_average.to_csv(
            os.path.join(script_dir, 'data', folder, filename), index=False
        )

    return df_average

def float_to_latex(x, precision=0):
    s = f"{x:.{precision}e}"      # scientific notation, e.g. '2.30e+04'
    base, exp = s.split('e')
    exp = int(exp)                # remove + or leading zeros
    return rf"{base}\cdot 10^{{{exp}}}"

def extract_vremain(path):
    pattern = re.compile(
        r'smub_VRemain=([-+]?\d+(?:\.\d+)?e[-+]?\d+)',
        re.IGNORECASE
    )

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))

    raise ValueError("VRemain not found in file")

def extract_date_start(path):
    pattern = re.compile(
        r'dateStart=(\d{2}\.\d{2}\.\d{4}@\d{2}:\d{2}:\d{2})'
    )

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return datetime.strptime(
                    m.group(1),
                    "%d.%m.%Y@%H:%M:%S"
                )

    raise ValueError("dateStart not found in file")

def extract_data_crv(filename):
    colnames = None
    units = None
    data_start = None

    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith("#n"):
                colnames = line.split()[1:]  # drop '#n'
            elif line.startswith("#u"):
                units = line.split()[1:]     # drop '#u'
            elif not line.startswith("#") and line.strip():
                data_start = i
                break

    if colnames is None:
        raise ValueError("Missing #n column header")

    # Load numerical data
    data = np.loadtxt(filename, skiprows=data_start)

    # Force 2D even for a single row
    data = np.atleast_2d(data)

    df = pd.DataFrame(data, columns=colnames)

    # time = df['t'].values
    # Id = df['IDrain'].values
    # Vd = df['VDrain'].values
    # Vg = df['VTopGate'].values
    # if 'ITopGate' in df.columns:
    #     Ig = df['ITopGate'].values
    # else:
    #     Ig = None

    return df

def extract_data_AxelHyst(Dict,Data,precondition,batch,dut,sample,T):
    data_axel = []
    DeviceDict = Dict['DeviceDict']
    MeasDict = Dict['MeasDict']
    
    # Get values from Dict's
    AdvancedSetupDict=Dict["AdvancedSetupDict"]
    HystFreqDict=MeasDict["Hyst sweep freq.:"]
    AERCalcDict=Dict["AERCalcDict"]
        
    # replace Id=0.0A values
    #Data['Id [A]'].replace(0.0, np.nan, inplace=True)           # replace 0.0A with NaN -> can use forward fill function
    Data.replace({'Id [A]': 0.0}, np.nan, inplace=True)           # replace 0.0A with NaN -> can use forward fill function
    Data['Id [A]'] = Data['Id [A]'].ffill()               # fill NaN/0.0A values with preceding value in dataframe forward fill (ffill)
    
    # PreCond Parameters
    if precondition == True:
        NPreCondSweeps = MeasDict["NPreCond_before_Hyst"]
        #PreCondDataPoints = MeasDict["NPreCond_before_Hyst"] * MeasDict["Points_per_sweep"]
        FreqArray = np.full(NPreCondSweeps, AdvancedSetupDict["f_PreCond"])
        NSweeps =  len(FreqArray)
    else:
        FreqArray = np.fromiter(HystFreqDict.values(), dtype=float)
        NSweeps = len(FreqArray)

    for SweepIndex in range(0, NSweeps):
        SweepStartIndex = MeasDict["Points_per_sweep"]*SweepIndex
        SweepData = Data.iloc[SweepStartIndex : SweepStartIndex + MeasDict["Points_per_sweep"]+1,:]
        
        absId_Sweep=np.array(np.abs(SweepData['Id [A]'])).tolist()
        Vg_Sweep=np.array((SweepData['Vg [V]'])).tolist()
        t_Sweep=np.array((SweepData['t [s]'])).tolist()
        Vd_Sweep=np.array((SweepData['Vd [V]'])).tolist()
        tsweep_actual = SweepData['t [s]'].iloc[-1] - SweepData['t [s]'].iloc[0]
        
        data_axel.append({
            "batch": batch,
            "dut": dut,
            "sample": sample,
            "T": T,
            'meas_type': 'hyst',
            'cycle': 1,
            "precondition": precondition,
            "SweepIndex": SweepIndex,
            "nom_freq": FreqArray[SweepIndex],
            "freq": 1/tsweep_actual,
            'date': AdvancedSetupDict['MeasDateTime'],
            "Vd": MeasDict['Vd'],
            "Eod": AERCalcDict['Eod'],
            "Ioff": AERCalcDict['IOff'],
            "Vg": Vg_Sweep,
            "Id": absId_Sweep,
            "time": t_Sweep
        })

    return data_axel

def extract_data_AxelIdVg(Dict,Data,VarVd,batch,dut,sample,T):
    data_axel = []
    MeasDict = Dict['MeasDict']
    
    # Get values from Dict's
    AdvancedSetupDict=Dict["AdvancedSetupDict"]
        
    # replace Id=0.0A values
    #Data['Id [A]'].replace(0.0, np.nan, inplace=True)           # replace 0.0A with NaN -> can use forward fill function
    Data.replace({'Id [A]': 0.0}, np.nan, inplace=True)           # replace 0.0A with NaN -> can use forward fill function
    Data['Id [A]'] = Data['Id [A]'].ffill()               # fill NaN/0.0A values with preceding value in dataframe forward fill (ffill)
    
    # PreCond Parameters
    if VarVd == True:
        NSweeps = len(MeasDict["VdValuesTbl"])
    else:
        NSweeps = 1

    for SweepIndex in range(0, NSweeps):
        SweepStartIndex = MeasDict["Points_per_sweep"]*SweepIndex
        SweepData = Data.iloc[SweepStartIndex : SweepStartIndex + MeasDict["Points_per_sweep"],:]
        
        Id=np.array(np.abs(SweepData['Id [A]'])).tolist()
        Vg=np.array((SweepData['Vg [V]'])).tolist()
        time=np.array((SweepData['t [s]'])).tolist()
        Vd=np.array((SweepData['Vd_Source [V]'])).tolist()
        Ig=np.array((SweepData['Ig [A]'])).tolist()
        
        data_axel.append({
            'batch': batch,
            'dut':dut,
            'sample': sample,
            'T': T,
            'date': AdvancedSetupDict['MeasDateTime'],
            'sweep': SweepIndex,
            'Vd': Vd[0],
            'Vbg': None,
            'time': time,
            'Id': Id,
            'Vg': Vg,
            'Ig': Ig
        })

    return data_axel

def linear_fit(x, a, b):
    return a * x + b

def powerlaw_fit(x, A, n, C):
    return A * (x ** n) + C

def logistic_fit(x, ymin, ymax, x0, k):
    return ymin + (ymax - ymin)/(1 + np.exp(-k*(x-x0)))

def log_fit(x, a, b, c):
    return a + b * np.log(np.abs(x) + 1e-9) + c * (np.log(np.abs(x) + 1e-9) ** 2)

def lambert_fit(Vg, I0, theta, K, n, Vth, Vt):
    arg = K * (1 + theta*Vg) * np.exp((Vg - Vth) / (n*Vt))
    W = special.lambertw(arg)
    return np.real(I0 / (1 + theta*Vg) * W)

def Vg_model(Id, n, I0, K, theta, Vth):
    """
    Computes V_G from I_D using the provided model.

    Parameters
    ----------
    Id : array-like or float
        Drain current
    n : float
        Slope factor
    vth : float
        Thermal voltage (kT/q)
    I0 : float
        Normalization current
    K : float
        Gain parameter
    theta : float
        Mobility degradation parameter

    Returns
    -------
    Vg : array-like or float
        Gate voltage
    """

    Id = np.asarray(Id)

    numerator = (
        n * Vth * np.log(Id)
        + (n * Vth / I0) * Id
        - n * Vth * np.log(K * I0)
    )

    denominator = 1 - (n * Vth * theta / I0) * Id

    return numerator / denominator

def stretched_exp_fit(x, A, tau, beta, C):
    return A *(1 - np.exp(-(x/tau)**beta)) + C

def loadinv_model(Vin, Vth, Vx, Vdd):
    Vin = np.asarray(Vin)

    Vin_boundary = Vth + np.sqrt(2*Vdd*Vx - Vx**2) - Vx
    # Define regions
    cond1 = Vin < Vth
    cond2 = (Vin >= Vth) & (Vin < Vin_boundary)
    cond3 = (Vin >= Vin_boundary) & (Vin >= np.sqrt(2*Vdd*Vx) - Vx + Vth)
    cond4 = (Vin >= Vin_boundary) & (Vin < np.sqrt(2*Vdd*Vx) - Vx + Vth)

    Vout = np.zeros_like(Vin)

    # Region 1
    Vout[cond1] = Vdd

    # Region 2
    Vout[cond2 | cond4] = Vdd - 1/(2*Vx)*(Vin[cond2 | cond4]-Vth)**2

    # Region 3
    Vout[cond3] = (
        Vin[cond3] - Vth + Vx
        - np.sqrt((Vin[cond3] - Vth + Vx)**2 - 2*Vdd*Vx)
    )
    return Vout

def depletioninv_model(Vin, Vthd, Vthl, kd, kl, Vdd):
    Vin = np.asarray(Vin)

    # Define regions
    cond1 = Vin < Vthd
    cond2 = (Vin >= Vthd) & (Vin <((kd*Vthd + kl*(Vdd + Vthl)
    - np.sqrt(kd*kl*((Vdd + Vthl - Vthd)**2 + Vthl**2*(kl/kd - 1)))) / (kd + kl)))
    cond3 = Vin >= ((kd*Vthd + kl*(Vdd + Vthl)
    - np.sqrt(kd*kl*((Vdd + Vthl - Vthd)**2 + Vthl**2*(kl/kd - 1)))) / (kd + kl))

    Vout = np.zeros_like(Vin)

    # Region 1
    Vout[cond1] = Vdd

    # Region 2
    Vout[cond2] = Vdd + Vthl - np.sqrt(Vthl**2 - kd/kl*(Vin[cond2]-Vthd)**2)

    # sqrt_arg = (Vin[cond3] - Vth + Vx)**2 - 2*Vdd*Vx
    # sqrt_arg = np.maximum(sqrt_arg, 0)

    # Region 3
    Vout[cond3] = (Vin[cond3] - Vthd) - np.sqrt((Vin[cond3] - Vthd)**2 - kl/kd*Vthl**2) 

    return Vout

def stretched_exp_initial_guess(x, y):
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Offset ≈ early-time value
    C0 = y[0]

    # Amplitude ≈ total change
    A0 = y[-1] - y[0]

    # τ ≈ time at 63% of total change
    y_target = C0 + 0.63 * A0
    tau0 = x[np.argmin(np.abs(y - y_target))]

    # β ≈ 0.3–0.7 typical for BTI
    beta0 = 0.5

    return A0, tau0, beta0, C0

def powerlaw_initial_guess(x, y):
    idx = np.argsort(x)
    x = np.asarray(x)[idx]
    y = np.asarray(y)[idx]

    # Offset ≈ early-time value
    C0 = y[0]

    # Amplitude ≈ total change
    A0 = y[-1] - y[0]

    # Avoid zero / negative times
    mask = x > 0
    x_fit = x[mask]
    y_fit = y[mask] - C0

    # Fallback if not enough points
    if len(x_fit) < 2 or np.any(y_fit <= 0):
        n0 = 0.25
    else:
        # log–log slope estimate
        logx = np.log(x_fit)
        logy = np.log(y_fit)
        n0 = np.polyfit(logx, logy, 1)[0]
        n0 = np.clip(n0, 0.05, 1.0)

    return A0, n0, C0

def parse_length(value_str, target_unit="m"):
    """
    Parse a string containing a number + unit and return value in target_unit.
    Default target_unit is meters.
    """

    if not isinstance(value_str, str):
        return float(value_str)

    # --- Extract numeric part (supports scientific notation)
    num_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value_str)
    if not num_match:
        raise ValueError(f"No numeric value found in '{value_str}'")

    value = float(num_match.group())

    # --- Extract unit (letters or Å)
    unit_match = re.search(r'[a-zA-ZÅ]+', value_str)
    unit = unit_match.group().lower() if unit_match else target_unit

    # --- Unit conversion to meters
    unit_factors = {
        "m": 1,
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
        "a": 1e-10,
        "å": 1e-10,
    }

    if unit not in unit_factors:
        raise ValueError(f"Unsupported unit '{unit}'")

    value_m = value * unit_factors[unit]

    # --- Convert to requested unit
    if target_unit not in unit_factors:
        raise ValueError(f"Unsupported target unit '{target_unit}'")

    return value_m / unit_factors[target_unit]

def safe_json_load(x):
    if isinstance(x, str):
        try:
            x = x.replace('nan', 'null')  # JSON-safe
            out = json5.loads(x)
            return [np.nan if v is None else v for v in out]
        except Exception:
            return x   # or return None
    return x

def set_matplotlibstyle(textSize=28, textSizeLegend=24):
    plt.rcParams.update(
        {   
            # Ticks control
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
            "xtick.labelsize": textSizeLegend,
            "ytick.labelsize": textSizeLegend,
            "font.size": textSize,
            "legend.fontsize": textSizeLegend,
            "legend.framealpha": None,
            "axes.labelsize": textSize,

            #  Marker control
            "lines.markersize": 20,
            "lines.markeredgewidth": 3,
            "lines.markeredgecolor": "#13073A",
            "lines.linewidth": 6,
        }
    )

    plt.rcParams["text.usetex"] = False
