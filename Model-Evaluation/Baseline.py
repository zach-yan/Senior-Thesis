import os
import json
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from network_architectures import DirectRewardNet, ClassifierNet
from online_evals2 import create_patient_objects, WardEnvironment, RollingHorizonSimulator, PerfectForesightOracle, \
    ExpectedValueBridge, NeuroICUSchedulingModel
from solve_global_dynamic import solve_dynamic_global_oracle

plt.switch_backend('Agg')

plt.rcParams.update({
    "text.usetex": False,            # Turn off external LaTeX compiler
    "font.family": "serif",          # Use serif fonts
    "mathtext.fontset": "cm",        # Use Computer Modern for math/LaTeX text
    "font.serif": ["cmr10", "Computer Modern Roman"], # Try cmr10 first, fallback to CM Roman
    "axes.formatter.use_mathtext": True
})


# ==========================================
# 1. SETUP & UTILS
# ==========================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_scan_log(scan_log):
    scans_by_patient = defaultdict(list)
    for entry in scan_log:
        scans_by_patient[entry['patient_id']].append(entry['hour'])
    for pid in scans_by_patient:
        scans_by_patient[pid].sort()
    return dict(scans_by_patient)


def evaluate_peak_detection_with_std(oracle_log, model_log, tau=1):
    oracle_scans, model_scans = format_scan_log(oracle_log), format_scan_log(model_log)
    tp, fp, fn = 0, 0, 0
    raw_offsets = []

    all_patients = set(oracle_scans.keys()).union(set(model_scans.keys()))
    for pid in all_patients:
        o_hours = oracle_scans.get(pid, [])
        m_hours = model_scans.get(pid, []).copy()

        if not m_hours:
            fn += len(o_hours)
            continue
        if not o_hours:
            fp += len(m_hours)
            continue

        for o_h in o_hours:
            if not m_hours:
                fn += 1
                continue
            differences = [abs(m_h - o_h) for m_h in m_hours]
            min_idx = np.argmin(differences)

            if differences[min_idx] <= tau:
                tp += 1
                raw_offsets.append(m_hours.pop(min_idx) - o_h)
            else:
                fn += 1
        fp += len(m_hours)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    mean_offset = np.mean(raw_offsets) if raw_offsets else 0.0
    std_offset = np.std(raw_offsets) if raw_offsets else 0.0

    return {
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Mean Offset': mean_offset,
        'Std Offset': std_offset
    }


# ==========================================
# 2. BASELINE SIMULATION EXECUTION
# ==========================================
def run_baseline_analysis():
    print("Loading Test Set and Models...")
    dfV = pd.read_csv("HELMET_Triangular_Targets_with_V.csv")
    patientArray = create_patient_objects(dfV)

    train_patients, temp_patients = train_test_split(patientArray, test_size=0.30, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)
    T = 3

    all_targets = [p.get_targets(t) for group in [train_patients, val_patients] for p in group for t in range(p.T - 2)]
    mean_action_rewards = np.array(all_targets).mean(axis=0)

    mse_params = optuna.load_study(study_name='new_mse_study', storage='sqlite:///optuna_new_mse_study.db').best_params
    huber_params = optuna.load_study(study_name='new_huber_study',
                                     storage='sqlite:///optuna_new_huber_study.db').best_params
    class_params = optuna.load_study(study_name='new_classifier_study',
                                     storage='sqlite:///optuna_new_classifier_study.db').best_params
    spo_params = optuna.load_study(study_name='new_spo_dfl_study',
                                   storage='sqlite:///optuna_new_spo_dfl.db').best_params

    model_mse = DirectRewardNet(input_dim=46, T=T, hidden_dim=mse_params['hidden_dim']).to(device)
    model_mse.load_state_dict(torch.load('final_mse.pth'))
    model_mse.eval()

    model_huber = DirectRewardNet(input_dim=46, T=T, hidden_dim=huber_params['hidden_dim']).to(device)
    model_huber.load_state_dict(torch.load('final_huber.pth'))
    model_huber.eval()

    model_spo = DirectRewardNet(input_dim=46, T=T, hidden_dim=spo_params['hidden_dim']).to(device)
    model_spo.load_state_dict(torch.load('final_spo.pth'))
    model_spo.eval()

    model_class = ClassifierNet(input_dim=46, hidden_dim=class_params['hidden_dim']).to(device)
    model_class.load_state_dict(torch.load('final_classifier.pth'))
    model_class.eval()

    bridge_classifier = ExpectedValueBridge(model_class, mean_action_rewards, device)

    N, T, BASE_R, SIM_HOURS = 30, 3, 3, 1000
    Q, delta = 6, 3  # Set your desired scan limits and refractory periods

    # Trackers specifically for the clinical baseline
    clinical_C = np.zeros(N)
    clinical_L = np.full(N, -np.inf)

    print(f"\n{'=' * 45}\nRunning Baseline Environment\n{'=' * 45}")
    set_seeds(42)
    patient_rng = np.random.RandomState(42)

    eval_model = NeuroICUSchedulingModel(N=N, T=T, R=BASE_R)
    oracle_patient_ref = []
    model_rolling_oracle = PerfectForesightOracle(oracle_patient_ref, T, device)

    wards = [
        WardEnvironment("MSE", RollingHorizonSimulator(model_mse, eval_model, N=N, T=T, R=BASE_R, device=device)),
        WardEnvironment("Huber", RollingHorizonSimulator(model_huber, eval_model, N=N, T=T, R=BASE_R, device=device)),
        WardEnvironment("SPO+", RollingHorizonSimulator(model_spo, eval_model, N=N, T=T, R=BASE_R, device=device)),
        WardEnvironment("Classifier",
                        RollingHorizonSimulator(bridge_classifier, eval_model, N=N, T=T, R=BASE_R, device=device)),
        WardEnvironment("Rolling Oracle",
                        RollingHorizonSimulator(model_rolling_oracle, eval_model, N=N, T=T, R=BASE_R, device=device)),
        WardEnvironment("Clinical Baseline", None, is_clinical_baseline=True)
    ]

    # Add usage log to wards
    for w in wards:
        w.usage_log = []

    global_patient_trace = []

    initial_indices = patient_rng.choice(len(test_patients), N, replace=False)
    initial_pool = []
    for idx in initial_indices:
        p = copy.deepcopy(test_patients[idx])

        # 1. Select a random start time within their length of stay
        # patient_rng.randint(low, high) is exclusive of high
        start_t = patient_rng.randint(0, p.T)
        p.current_t = start_t

        # 2. Update their observation trackers relative to this new start time
        past_scans = np.where(p.scans[:start_t + 1] == 1)[0]
        if len(past_scans) > 0:
            last_scan_idx = past_scans[-1]
            p.last_observed_state = p.true_states[last_scan_idx]
            p.time_since_last_scan = start_t - last_scan_idx
        else:
            p.last_observed_state = p.features[0, 21]
            p.time_since_last_scan = start_t

        p.episode_id = f"{p.id}_0"
        global_patient_trace.append({
            "patient_id": p.episode_id,
            "arrival_hour": 0,
            "length_of_stay": p.T - start_t,  # Reflect remaining time in simulation
            "true_rewards": copy.deepcopy(p.rewards).tolist()
        })
        initial_pool.append(p)

    for w in wards:
        w.assign_initial_patients(copy.deepcopy(initial_pool))
        if w.name == "Rolling Oracle": oracle_patient_ref.extend(w.patients)

    for global_hour in range(SIM_HOURS):
        incoming_this_hour = {}
        for i in range(N):
            if not wards[0].patients[i].active:
                idx = patient_rng.choice(len(test_patients))
                new_p = copy.deepcopy(test_patients[idx])
                unique_episode_id = f"{new_p.id}_{global_hour}"
                new_p.episode_id = unique_episode_id
                incoming_this_hour[i] = new_p

                global_patient_trace.append({
                    "patient_id": unique_episode_id, "arrival_hour": global_hour, "length_of_stay": new_p.T,
                    "true_rewards": copy.deepcopy(new_p.rewards)
                })

        if incoming_this_hour:
            for w in wards:
                w.replace_discharged_patients(copy.deepcopy(incoming_this_hour))
                if w.name == "Rolling Oracle":
                    oracle_patient_ref.clear()
                    oracle_patient_ref.extend(w.patients)
                elif not w.is_clinical:
                    for bed_idx in incoming_this_hour.keys():
                        w.simulator.C[bed_idx] = 0
                        w.simulator.L[bed_idx] = -np.inf
                else:  # Reset trackers for the clinical baseline
                    for bed_idx in incoming_this_hour.keys():
                        clinical_C[bed_idx] = 0
                        clinical_L[bed_idx] = -np.inf

        for w in wards:
            if w.is_clinical:
                desired_scans = []
                for i, p in enumerate(w.patients):
                    # Check historical scan request AND the Q / Delta constraints
                    if p.scans[p.current_t] == 1:
                        if clinical_C[i] < Q and (global_hour - clinical_L[i]) >= delta:
                            desired_scans.append(i)

                if len(desired_scans) > BASE_R:
                    triage_data = [(i, w.patients[i].last_observed_state, w.patients[i].time_since_last_scan) for i in
                                   desired_scans]
                    triage_data.sort(key=lambda x: (x[1], x[2]), reverse=True)
                    scanned_indices = [data[0] for data in triage_data[:BASE_R]]
                else:
                    scanned_indices = desired_scans

                for i, p in enumerate(w.patients):
                    if i in scanned_indices:
                        p.update_after_scan(p.true_states[p.current_t])
                        # Update the clinical trackers upon successful scan
                        clinical_C[i] += 1
                        clinical_L[i] = global_hour
                    else:
                        p.update_no_scan()
            else:
                scanned_indices = w.simulator.step(global_hour, w.patients)

            for i in scanned_indices:
                w.scan_log.append({'patient_id': w.patients[i].episode_id, 'hour': global_hour})

            w.usage_log.append(len(scanned_indices))

        if (global_hour + 1) % 250 == 0:
            print(f"Completed Hour {global_hour + 1}/{SIM_HOURS}")

    print("Solving Global Static Oracle (Full Hindsight)...")
    global_log = solve_dynamic_global_oracle(global_patient_trace, lambda t: BASE_R, SIM_HOURS=SIM_HOURS)
    # Calculate Scanner Usage for Global Oracle
    global_usage_log = [0] * SIM_HOURS
    for entry in global_log:
        global_usage_log[entry['hour']] += 1

        # ==========================================
        # 3. EVALUATION & PLOTTING
        # ==========================================
        print("\nEvaluating Metrics...")

        class_data = []
        offset_data = []
        avg_scans_data = []

        # Get a list of all unique patient IDs that passed through the simulation
        all_patient_ids = [p['patient_id'] for p in global_patient_trace]

        dark_blue_palette = ['mediumblue', 'royalblue', 'darkblue', 'deepskyblue', 'steelblue', 'darkgray', '#c6dbef']
        sns.set_palette(dark_blue_palette)

        for w in wards:
            # Calculate Average Scans and Standard Deviation per Patient
            scans_by_patient = format_scan_log(w.scan_log)
            patient_scan_counts = [len(scans_by_patient.get(pid, [])) for pid in all_patient_ids]

            avg_scans = np.mean(patient_scan_counts)
            std_scans = np.std(patient_scan_counts)
            avg_scans_data.append({'Policy': w.name, 'Avg Scans per Patient': avg_scans, 'Std Scans': std_scans})

            if w.name == "Global Oracle": continue
            res = evaluate_peak_detection_with_std(global_log, w.scan_log, tau=1)

            class_data.extend([
                {'Policy': w.name, 'Metric': 'Precision', 'Score': res['Precision'] * 100},
                {'Policy': w.name, 'Metric': 'Recall', 'Score': res['Recall'] * 100},
                {'Policy': w.name, 'Metric': 'F1-Score', 'Score': res['F1-Score'] * 100}
            ])
            offset_data.append({
                'Policy': w.name, 'Mean Offset': res['Mean Offset'], 'Std Offset': res['Std Offset']
            })

        # Add Global Oracle to avg scans with standard deviation
        global_scans_by_patient = format_scan_log(global_log)
        global_patient_scan_counts = [len(global_scans_by_patient.get(pid, [])) for pid in all_patient_ids]

        avg_scans_global = np.mean(global_patient_scan_counts)
        std_scans_global = np.std(global_patient_scan_counts)
        avg_scans_data.append(
            {'Policy': 'Global Oracle', 'Avg Scans per Patient': avg_scans_global, 'Std Scans': std_scans_global})

        df_class = pd.DataFrame(class_data)
        df_offset = pd.DataFrame(offset_data)
        df_avg_scans = pd.DataFrame(avg_scans_data)

    # --- PLOT 1: Classification Metrics ---
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(data=df_class, x='Metric', y='Score', hue='Policy', edgecolor=".2")
    plt.title('Classification Metrics vs Global Oracle', fontsize=16, fontweight='bold', color='#08306b')
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold', color='#08306b')
    plt.xlabel('')
    plt.ylim(0, 110)

    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom',
                         fontsize=6, fontweight='bold', color='#08306b', xytext=(0, 4), textcoords='offset points')

    plt.legend(title='Policy', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig('baseline_classification_metrics.png', dpi=300)
    plt.close()

    # --- PLOT 2: Temporal Offset with Std Dev ---
    plt.figure(figsize=(10, 6))
    ax2 = plt.gca()
    bars = ax2.bar(df_offset['Policy'], df_offset['Mean Offset'], yerr=df_offset['Std Offset'], capsize=5,
                   color=dark_blue_palette[:len(df_offset)], edgecolor=".2")
    plt.title('Mean Temporal Offset with Standard Deviation', fontsize=16, fontweight='bold',
              color='#08306b')
    plt.ylabel('Mean Offset (Hours)', fontsize=12, color='#08306b')
    plt.xlabel('Policy', fontsize=12, fontweight='bold', color='#08306b')

    # Add a horizontal line at 0 (perfect timing)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')

    # Expand the bottom limit slightly to make enough room for the annotations at the bottom
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min - (y_max - y_min) * 0.15, y_max)

    # Annotate mean and std values uniformly at the bottom edge
    for i, bar in enumerate(bars):
        mean_val = bar.get_height()
        std_val = df_offset['Std Offset'].iloc[i]

        annotation_text = f"{mean_val:.2f}h\n($\\pm${std_val:.2f}h)"

        ax2.annotate(annotation_text,
                     xy=(bar.get_x() + bar.get_width() / 2., 0.02),
                     xycoords=('data', 'axes fraction'),
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#08306b')

    plt.tight_layout()
    plt.savefig('baseline_temporal_offset.png', dpi=300)
    plt.close()

    # --- PLOT 3: Average Scans Per Patient ---
    plt.figure(figsize=(10, 6))
    ax3 = plt.gca()

    # Use matplotlib bar directly to easily handle yerr (error bars)
    bars3 = ax3.bar(df_avg_scans['Policy'], df_avg_scans['Avg Scans per Patient'],
                    yerr=df_avg_scans['Std Scans'], capsize=5,
                    color=[dark_blue_palette[i % len(dark_blue_palette)] for i in range(len(df_avg_scans))],
                    edgecolor=".2")

    plt.title('Average Scans per Patient by Policy', fontsize=16, fontweight='bold', color='#08306b')
    plt.ylabel('Average Scans', fontsize=12, fontweight='bold', color='#08306b')
    plt.xlabel('Policy', fontsize=12, fontweight='bold', color='#08306b')
    plt.xticks(rotation=45)

    # Annotate mean and std deviation text above the error bars
    for i, p in enumerate(bars3):
        h = p.get_height()
        std_val = df_avg_scans['Std Scans'].iloc[i]
        ax3.annotate(f"{h:.2f}\n($\\pm${std_val:.2f})",
                     (p.get_x() + p.get_width() / 2., h + std_val),
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color='#08306b', xytext=(0, 4), textcoords='offset points')

    # Expand the top limit slightly to ensure the annotations don't get cut off
    y_min, y_max = ax3.get_ylim()
    ax3.set_ylim(y_min, y_max * 1.15)

    plt.tight_layout()
    plt.savefig('baseline_avg_scans_per_patient.png', dpi=300)
    plt.close()

    # --- PLOT 4: Scanner Usage Over Time (Split) ---
    # Create subplots for each ward plus the Global Oracle
    fig, axes = plt.subplots(nrows=len(wards) + 1, ncols=1, figsize=(12, 18), sharex=True)
    fig.suptitle('Scanner Usage Over Time (Baseline $R=3$)', fontsize=16, fontweight='bold', color='#08306b')

    for i, w in enumerate(wards):
        # Use plt.step to accurately reflect discrete integer states
        axes[i].step(range(SIM_HOURS), w.usage_log, where='post', color=dark_blue_palette[i % len(dark_blue_palette)])
        axes[i].set_title(f'{w.name} Usage', fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Scanners', fontsize=9)
        axes[i].set_yticks([0, 1, 2, 3])  # Force y-axis to only show whole numbers
        axes[i].set_ylim(-0.2, 3.2)
        axes[i].grid(True, alpha=0.3)

    # Plot Global Oracle on the final subplot
    axes[-1].step(range(SIM_HOURS), global_usage_log, where='post', color=dark_blue_palette[-1])
    axes[-1].set_title('Global Oracle Usage', fontsize=10, fontweight='bold')
    axes[-1].set_ylabel('Scanners', fontsize=9)
    axes[-1].set_xlabel('Simulation Hour', fontsize=12, fontweight='bold')
    axes[-1].set_yticks([0, 1, 2, 3])
    axes[-1].set_ylim(-0.2, 3.2)
    axes[-1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Adjust top to make room for suptitle
    plt.subplots_adjust(top=0.95)
    plt.savefig('baseline_scanner_usage_split.png', dpi=300)
    plt.close()
    # --- PLOT 5: Individual Patient Reward Curves & Scan Decisions ---
    print("Generating Patient-Level Scan Visualizations...")

    # Define number of patients to sample for the visualization
    num_patients_to_plot = 4

    # Safely sample patients (ensure we don't sample more than exist)
    num_to_sample = min(num_patients_to_plot, len(global_patient_trace))
    sample_patients = patient_rng.choice(global_patient_trace, num_to_sample, replace=False)

    fig, axes = plt.subplots(nrows=num_to_sample, ncols=1, figsize=(12, 3.5 * num_to_sample), sharex=False)
    if num_to_sample == 1:
        axes = [axes]

    fig.suptitle('Patient Reward Trajectories and Policy Scan Decisions', fontsize=16, fontweight='bold',
                 color='#08306b')

    # Distinct markers and colors for each policy to make them stand out on the curve
    policy_markers = {
        'MSE': ('o', 'mediumblue'),
        'Huber': ('s', 'royalblue'),
        'SPO+': ('^', 'steelblue'),
        'Classifier': ('D', 'deepskyblue'),
        'Rolling Oracle': ('v', 'darkblue'),
        'Clinical Baseline': ('X', 'darkgray'),
        'Global Oracle': ('*', '#c6dbef')
    }

    for idx, p_info in enumerate(sample_patients):
        ax = axes[idx]
        pid = p_info['patient_id']
        arr_hour = p_info['arrival_hour']
        los = p_info['length_of_stay']
        rewards = p_info['true_rewards']

        # Time steps relative to the patient's admission
        time_steps = list(range(los))

        # Plot the underlying true reward curve
        ax.plot(time_steps, rewards, label='True Expected Reward', color='black', linestyle='--', linewidth=2,
                zorder=1)

        # Plot scan decisions for each ward policy
        for w in wards:
            # Extract absolute scan hours and convert to relative hours since admission
            patient_scans = [entry['hour'] - arr_hour for entry in w.scan_log if entry['patient_id'] == pid]

            # Filter out scans that somehow occurred outside the patient's recorded LOS bounds
            patient_scans = [t for t in patient_scans if 0 <= t < los]

            if patient_scans:
                y_vals = [rewards[t] for t in patient_scans]
                marker, color = policy_markers.get(w.name, ('o', 'black'))

                # Add a slight horizontal jitter based on policy index to prevent markers from perfectly overlapping
                jitter = (wards.index(w) - len(wards) / 2) * 0.05
                jittered_scans = [t + jitter for t in patient_scans]

                ax.scatter(jittered_scans, y_vals, label=w.name, marker=marker, color=color, s=120, zorder=2,
                           edgecolors='white', alpha=0.8)

        # Plot Global Oracle decisions
        global_scans = [entry['hour'] - arr_hour for entry in global_log if entry['patient_id'] == pid]
        global_scans = [t for t in global_scans if 0 <= t < los]

        if global_scans:
            y_vals = [rewards[t] for t in global_scans]
            marker, color = policy_markers['Global Oracle']
            ax.scatter(global_scans, y_vals, label='Global Oracle', marker=marker, color=color, s=200, zorder=3,
                       edgecolors='black')

        # Formatting subplots
        ax.set_title(f'Patient ID: {pid} (Admitted at Global Hour {arr_hour})', fontsize=12, fontweight='bold',
                     loc='left')
        ax.set_xlabel('Hours Since Admission', fontsize=10)
        ax.set_ylabel('True Reward Signal', fontsize=10)

        # Ensure x-axis ticks are integers but not overcrowded
        tick_step = max(1, los // 10)  # Dynamically limit to ~10 ticks on the axis
        ax.set_xticks(range(0, los, tick_step))
        ax.grid(True, alpha=0.3, linestyle=':')

        # ==========================================
        # Global Legend Construction
        # ==========================================
        import matplotlib.lines as mlines

        # Manually create handles so every policy is guaranteed to appear in the legend
        legend_elements = [
            mlines.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Underlying True Reward')
        ]

        for policy_name, (marker, color) in policy_markers.items():
            if policy_name == 'Global Oracle':
                legend_elements.append(mlines.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                                                     markeredgecolor='black', markersize=14, label=policy_name,
                                                     linestyle='None'))
            else:
                legend_elements.append(mlines.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                                                     markeredgecolor='white', markersize=10, label=policy_name,
                                                     linestyle='None'))

        # Attach the master legend to the Figure, not an individual axis
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5),
                   title="Policies", title_fontproperties={'weight': 'bold'})

    plt.tight_layout()
    # Adjust layout to make room for the master title and the external legend
    plt.subplots_adjust(top=0.92, right=0.8)
    plt.savefig('baseline_patient_trajectories.png', dpi=300)
    plt.close()

    print("\nBaseline results saved successfully:")
    print(" - baseline_classification_metrics.png")
    print(" - baseline_temporal_offset.png")
    print(" - baseline_avg_scans_per_patient.png")
    print(" - baseline_scanner_usage_split.png")


if __name__ == "__main__":
    run_baseline_analysis()