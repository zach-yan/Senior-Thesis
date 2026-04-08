import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import gurobipy as gp
from gurobipy import GRB
import optuna
from pyepo.model.grb import optGrbModel

import torch
import torch.nn.functional as F

# Import your network architectures and data utilities
from network_architectures import DirectRewardNet, ClassifierNet

plt.switch_backend('Agg')
# ==========================================
# LATEX FONT CONFIGURATION (Built-in)
# ==========================================
plt.rcParams.update({
    "text.usetex": False,            # Turn off external LaTeX compiler
    "font.family": "serif",          # Use serif fonts
    "mathtext.fontset": "cm",        # Use Computer Modern for math/LaTeX text
    "font.serif": ["cmr10", "Computer Modern Roman"], # Try cmr10 first, fallback to CM Roman
    "axes.formatter.use_mathtext": True
})

# (Assume addFeatures is imported from your existing utilities)
from dataset_utils import addFeatures

class Patient:
    def __init__(self, patient_id, patient_rewards, delay_values, features, patient_scans, edema_states,
                 times_since_scan, T):
        self.id = patient_id
        self.features = features
        self.T = T
        self.rewards = patient_rewards
        self.delay_values = delay_values
        self.scans = patient_scans
        self.edema_states = edema_states
        self.times_since_scan = times_since_scan

    def get_context(self, t, window=2):
        """Retrieves the flattened feature context for the current and previous timesteps."""
        indices = np.arange(t - window + 1, t + 1)
        indices = np.maximum(indices, 0)
        return self.features[indices, :].flatten()

    def get_targets(self, t):
        """Retrieves the true rewards for the next 3 hours and the current delay penalty."""
        if t + 3 > self.T:
            padding = np.zeros(3 - (self.T - t))
            rewards_window = np.concatenate([self.rewards[t:], padding])
        else:
            rewards_window = self.rewards[t:t + 3]

        delay_val = np.array([self.delay_values[t]])
        return np.concatenate([rewards_window, delay_val])

    def get_clinician_action(self, t, horizon=3):
        """
        Determines the actual action taken by the clinician within the decision window.
        0: Scan at t
        1: Scan at t+1
        2: Scan at t+2
        3: Defer (No scan in the window)
        """
        max_lookahead = min(horizon, self.T - t)

        for i in range(max_lookahead):
            if self.scans[t + i] == 1:
                return i  # Return the index of the first scan found in the window

        # If no scan was found in the 3-hour window, the clinician deferred
        return 3

    def get_triage_features(self, t):
        """Returns the features used to prioritize patients if capacity is exceeded."""
        return (self.edema_states[t], self.times_since_scan[t])


def setup_gurobi_model(N, T, R):
    """
    Builds the base Gurobi MILP model for the NeuroICU scheduling evaluation.
    Variables and constraints are built once; the objective is updated per shift.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("NeuroICUScheduling_Eval", env=env)

    # Variables: N patients * (T + 1) actions.
    # Actions 0 to T-1 are scheduled scans; action T is 'Defer'.
    x = m.addVars(N * (T + 1), vtype=GRB.BINARY, name="x")

    # 1. Capacity Constraint: At most R scans per timestep t (excluding Defer)
    for t in range(T):
        m.addConstr(gp.quicksum(x[i * (T + 1) + t] for i in range(N)) <= R, name=f"Cap_{t}")

    # 2. Assignment Constraint: Exactly one decision (scan or defer) per patient
    for i in range(N):
        m.addConstr(gp.quicksum(x[i * (T + 1) + t] for t in range(T + 1)) == 1, name=f"Assign_{i}")

    m.update() # Commit variables and constraints
    return m, x

def create_patient_objects(df, horizon_T=3):
    """
    Instantiates Patient objects directly from the provided DataFrame.
    Filters out the portion of the stay before the first scan.
    """
    patient_objects = []

    for pid, group in df.groupby('ptid_idx'):
        group = group.sort_values('hour').reset_index(drop=True)
        scan_indices = group.index[group['true_scans'] == 1].tolist()

        # Skip patients who never received a scan
        if not scan_indices:
            continue

        # Start tracking from the patient's first scan
        first_scan_idx = scan_indices[0]
        remaining_stay_df = group.iloc[first_scan_idx:].reset_index(drop=True)
        total_stay = len(remaining_stay_df)

        # Only include patients whose remaining stay meets the required horizon
        if total_stay >= horizon_T:
            initial_state = remaining_stay_df['last_known_mls_class'].iloc[0]
            probs_matrix = remaining_stay_df[['proba_0', 'proba_1', 'proba_2', 'proba_3']].values
            rewards = remaining_stay_df['proxy_reward'].values
            delay_values = remaining_stay_df['scan_delay_value'].values
            scans = remaining_stay_df['true_scans'].values
            hours = remaining_stay_df['hour'].values

            # Use last_known_mls_class as the triage edema column based on CSV schema
            edema_col = 'last_known_mls_class'

            feature_list = []
            edema_list = []
            time_since_scan_list = []

            last_scan_hour = hours[0]
            last_observed_state = initial_state

            for t in range(total_stay):
                if scans[t] == 1:
                    last_scan_hour = hours[t]
                    # Update the last observed state when a true scan occurs
                    # Forward-fill logic is inherently maintained if `last_known_mls_class` is pre-processed
                    last_observed_state = remaining_stay_df['last_known_mls_class'].iloc[t]

                time_since_last_scan = hours[t] - last_scan_hour

                # Track triage features for the simulation loop
                edema_list.append(remaining_stay_df[edema_col].iloc[t])
                time_since_scan_list.append(time_since_last_scan)

                current_probs = probs_matrix[t]
                prev_probs = probs_matrix[t - 1] if t > 0 else (np.ones(4) / 4.0)

                # Construct the feature vector (Requires your custom addFeatures function)
                feat_vec = addFeatures(initial_state, hours[t], current_probs, prev_probs,
                                       last_observed_state, time_since_last_scan)
                feature_list.append(feat_vec)

            # Instantiate the finalized patient object
            p = Patient(
                patient_id=pid,
                patient_rewards=rewards,
                delay_values=delay_values,
                features=np.array(feature_list),
                patient_scans=scans,
                edema_states=np.array(edema_list),
                times_since_scan=np.array(time_since_scan_list),
                T=total_stay
            )
            patient_objects.append(p)

    return patient_objects

# ==========================================
# 1. SETUP & SEEDS
# ==========================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==========================================
# 2. EVALUATION MILP SOLVER (OPTIMIZED)
# ==========================================
class NeuroICUSchedulingModel(optGrbModel):
    def __init__(self, N, T, R):
        self.N = N
        self.T = T
        self.R = R
        super().__init__()

    def _getModel(self):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("NeuroICUScheduling_Rolling", env=env)

        x = m.addVars(self.N * (self.T + 1), vtype=GRB.BINARY, name="x")

        def get_x(i, t): return x[i * (self.T + 1) + t]
        def get_x_later(i): return x[i * (self.T + 1) + self.T]

        for t in range(self.T):
            m.addConstr(gp.quicksum(get_x(i, t) for i in range(self.N)) <= self.R)

        for i in range(self.N):
            m.addConstr(
                gp.quicksum(get_x(i, t) for t in range(self.T)) + get_x_later(i) == 1
            )
        return m, x


def solve_scheduling_milp(m, x_vars, cost_vector, N, T):
    """
    Updates the objective coefficients of the pre-compiled Gurobi model and re-solves.
    """
    # Update objective coefficients
    for j in range(N * (T + 1)):
        x_vars[j].Obj = cost_vector[j]

    # We want to minimize the cost
    m.ModelSense = GRB.MINIMIZE
    m.update()  # Ensure objective changes are pushed to the model
    m.optimize()

    # Extract the solution back into a numpy array
    sol = np.zeros(N * (T + 1))
    if m.status == GRB.OPTIMAL:
        for j in range(N * (T + 1)):
            sol[j] = x_vars[j].X
    else:
        print("Warning: Optimization did not find an optimal solution.")

    return sol


# ==========================================
# 3. TEST SET SAMPLER
# ==========================================
def get_eval_shift(test_patients, N):
    """Samples exactly N independent patients to simulate a hospital shift."""
    indices = np.random.choice(len(test_patients), N, replace=False)
    contexts, targets, clinician_actions, triage_info = [], [], [], []

    for idx in indices:
        p = test_patients[idx]
        t = np.random.randint(0, p.T - 2)

        contexts.append(p.get_context(t))
        targets.append(p.get_targets(t))
        clinician_actions.append(p.get_clinician_action(t))
        triage_info.append(p.get_triage_features(t))

    return (torch.tensor(np.array(contexts), dtype=torch.float32).to(device),
            np.array(targets),
            np.array(clinician_actions),
            triage_info)


# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def evaluate_models():
    print("Loading Test Set...")
    dfV = pd.read_csv("HELMET_Triangular_Targets_with_V.csv")
    patientArray = create_patient_objects(dfV)
    train_patients, temp_patients = train_test_split(patientArray, test_size=0.30, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)
    print(f"Test Set Size: {len(test_patients)} patients.")

    # Simulation Parameters
    N = 30
    T = 3
    R = 3
    kappa = 0.028
    K_shifts = 500

    print("Precomputing empirical mean rewards for Classifier Expected Value Bridge...")
    all_targets = []
    for p in train_patients:
        for t in range(p.T - 2):
            all_targets.append(p.get_targets(t))
    for p in val_patients:
        for t in range(p.T - 2):
            all_targets.append(p.get_targets(t))
    mean_action_rewards = np.array(all_targets).mean(axis=0)
    print(f"Empirical Mean Rewards (t=0, t=1, t=2, Defer): {mean_action_rewards}")

    print("Extracting optimal parameters from Optuna...")
    mse_params = optuna.load_study(study_name='new_mse_study', storage='sqlite:///optuna_new_mse_study.db').best_params
    huber_params = optuna.load_study(study_name='new_huber_study',
                                     storage='sqlite:///optuna_new_huber_study.db').best_params
    class_params = optuna.load_study(study_name='new_classifier_study',
                                     storage='sqlite:///optuna_new_classifier_study.db').best_params
    spo_params = optuna.load_study(study_name='new_spo_dfl_study',
                                   storage='sqlite:///optuna_new_spo_dfl.db').best_params

    print("Loading Trained Models...")

    # Initialize and load MSE
    model_mse = DirectRewardNet(input_dim=46, T=T, hidden_dim=mse_params['hidden_dim']).to(device)
    model_mse.load_state_dict(torch.load('final_mse.pth'))
    model_mse.eval()
    mse_scale = mse_params.get('scale_factor', 1.0)

    # Initialize and load Huber
    model_huber = DirectRewardNet(input_dim=46, T=T, hidden_dim=huber_params['hidden_dim']).to(device)
    model_huber.load_state_dict(torch.load('final_huber.pth'))
    model_huber.eval()
    huber_scale = huber_params.get('scale_factor', 1.0)

    # Initialize and load SPO+
    model_spo = DirectRewardNet(input_dim=46, T=T, hidden_dim=spo_params['hidden_dim']).to(device)
    model_spo.load_state_dict(torch.load('final_spo.pth'))
    model_spo.eval()
    spo_scale = spo_params.get('scale_factor', 1.0)

    # Initialize and load Classifier
    model_class = ClassifierNet(input_dim=46, hidden_dim=class_params['hidden_dim']).to(device)
    model_class.load_state_dict(torch.load('final_classifier.pth'))
    model_class.eval()

    # Tracking Dictionaries (NEW: Added Clinician)
    # Tracking Dictionaries (Updated with 'scans')
    results = {
        "Oracle": {"reward": [], "scans": []},
        "Clinician": {"reward": [], "regret": [], "scans": []},
        "MSE": {"reward": [], "regret": [], "scans": []},
        "Huber": {"reward": [], "regret": [], "scans": []},
        "Classifier": {"reward": [], "regret": [], "scans": []},
        "SPO+": {"reward": [], "regret": [], "scans": []}
    }

    print(f"Setting up Master Gurobi Environment for N={N}, T={T}, R={R}...")
    eval_model, eval_vars = setup_gurobi_model(N, T, R)

    print(f"Starting simulation of {K_shifts} NeuroICU shifts...")
    with torch.no_grad():
        worst_spo_regret = -1.0
        worst_spo_data = {}

        for k in range(K_shifts):
            if (k + 1) % 50 == 0:
                print(f"Evaluating shift {k + 1}/{K_shifts}...")

            # Extract Shift Context
            x_batch, s_batch_true, clinician_actions, triage_info = get_eval_shift(test_patients, N)

            # Construct the TRUE Reward and Cost Vectors
            true_reward_matrix = np.copy(s_batch_true)
            true_reward_matrix[:, :3] -= kappa
            true_reward_flat = true_reward_matrix.flatten()

            # Gurobi minimizes cost, so true cost is the negative true reward
            true_cost_flat = -true_reward_flat

            # ---------------------------------------------------------
            # 1. Oracle Decision
            # ---------------------------------------------------------
            oracle_sol = solve_scheduling_milp(eval_model, eval_vars, true_cost_flat, N, T)
            oracle_reward = np.dot(oracle_sol, true_reward_flat)
            results["Oracle"]["reward"].append(oracle_reward)
            # Count Oracle Scans
            oracle_scan_count = np.sum(oracle_sol.reshape(N, T + 1)[:, :3])
            results["Oracle"]["scans"].append(oracle_scan_count)

            # ---------------------------------------------------------
            # 1.5 Clinician Baseline (With Capacity Triage)
            # ---------------------------------------------------------
            for tau in range(3):
                contenders = np.where(clinician_actions == tau)[0]

                if len(contenders) > R:
                    # Triage: Create a list of (patient_index, edema_state, time_since_scan)
                    contender_features = [
                        (i, triage_info[i][0], triage_info[i][1]) for i in contenders
                    ]

                    # Sort DESCENDING: Higher edema state first, then longer time since last scan
                    contender_features.sort(key=lambda x: (x[1], x[2]), reverse=True)

                    # The top R patients keep their scan. The rest get Deferred (Action = 3)
                    losers = [x[0] for x in contender_features[R:]]
                    for loser in losers:
                        clinician_actions[loser] = 3

            clinician_sol = np.zeros(N * (T + 1))
            for i, action in enumerate(clinician_actions):
                clinician_sol[i * (T + 1) + action] = 1

            clinician_reward = np.dot(clinician_sol, true_reward_flat)
            results["Clinician"]["reward"].append(clinician_reward)
            results["Clinician"]["regret"].append(oracle_reward - clinician_reward)
            # Count Clinician Scans
            clinician_scan_count = np.sum(clinician_sol.reshape(N, T + 1)[:, :3])
            results["Clinician"]["scans"].append(clinician_scan_count)

            # ---------------------------------------------------------
            # 2. Continuous Baselines (MSE, Huber, SPO+)
            # ---------------------------------------------------------
            model_configs = [
                ("MSE", model_mse, 1.0),
                ("Huber", model_huber, 1.0),
                ("SPO+", model_spo, spo_scale)
            ]

            for name, model, scale in model_configs:
                preds = (model(x_batch) / scale).cpu().numpy()

                pred_cost_matrix = np.copy(preds)
                pred_cost_matrix[:, :3] = kappa - pred_cost_matrix[:, :3]
                pred_cost_matrix[:, 3] = -pred_cost_matrix[:, 3]

                model_sol = solve_scheduling_milp(eval_model, eval_vars, pred_cost_matrix.flatten(), N, T)
                model_reward = np.dot(model_sol, true_reward_flat)

                current_regret = oracle_reward - model_reward

                results[name]["reward"].append(model_reward)
                results[name]["regret"].append(current_regret)
                # Count Model Scans
                model_scan_count = np.sum(model_sol.reshape(N, T + 1)[:, :3])
                results[name]["scans"].append(model_scan_count)

                # --- Track the worst SPO+ Outlier ---
                if name == "SPO+" and current_regret > worst_spo_regret:
                    worst_spo_regret = current_regret
                    worst_spo_data = {
                        'shift': k,
                        'true_rewards': true_reward_matrix.copy(),
                        'spo_preds': preds.copy(),
                        'oracle_sol': oracle_sol.copy(),
                        'spo_sol': model_sol.copy()
                    }

            # ---------------------------------------------------------
            # 3. Classifier Baseline (Expected Value Bridge)
            # ---------------------------------------------------------
            logits = model_class(x_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            expected_rewards = probs * mean_action_rewards

            class_cost_matrix = np.copy(expected_rewards)
            class_cost_matrix[:, :3] = kappa - class_cost_matrix[:, :3]
            class_cost_matrix[:, 3] = -class_cost_matrix[:, 3]

            class_sol = solve_scheduling_milp(eval_model, eval_vars, class_cost_matrix.flatten(), N, T)
            class_reward = np.dot(class_sol, true_reward_flat)

            results["Classifier"]["reward"].append(class_reward)
            results["Classifier"]["regret"].append(oracle_reward - class_reward)
            # Count Classifier Scans
            class_scan_count = np.sum(class_sol.reshape(N, T + 1)[:, :3])
            results["Classifier"]["scans"].append(class_scan_count)

    print(f"\n==========================================")
    print(f" SPO+ WORST-CASE OUTLIER ANALYSIS")
    print(f"==========================================")
    print(f"Shift index: {worst_spo_data['shift']}")
    print(f"Max Regret:  {worst_spo_regret:.4f}")

    diff_indices = np.where(worst_spo_data['oracle_sol'] != worst_spo_data['spo_sol'])[0]
    print(f"Total differing binary decisions: {len(diff_indices)}\n")

    print("Breakdown of differing decisions:")
    print("Pat_Idx | Act_Idx | Oracle | SPO+ | True Reward | Pred Reward")
    print("-" * 65)

    for idx in diff_indices:
        patient_idx = idx // (T + 1)
        action_idx = idx % (T + 1)

        true_r = worst_spo_data['true_rewards'].flatten()[idx]
        pred_r = worst_spo_data['spo_preds'].flatten()[idx]

        oracle_chose = int(worst_spo_data['oracle_sol'][idx])
        spo_chose = int(worst_spo_data['spo_sol'][idx])

        print(
            f"  {patient_idx:2d}    |    {action_idx}    |   {oracle_chose}    |   {spo_chose}  |   {true_r:8.4f}  |   {pred_r:8.4f}")
    print("==========================================\n")
    print("\nSimulation Complete. Generating visual reports...")
    generate_visualizations(results, N) # UPDATED


# ==========================================
# 5. VISUALIZATION & REPORTING
# ==========================================
def generate_visualizations(results, N=30):
    # Added Clinician to model array
    models = ["MSE", "Huber", "Classifier", "SPO+", "Clinician"]
    all_models = models + ["Oracle"]
    colors = ['mediumblue', 'royalblue', 'deepskyblue', 'darkblue','darkgray']
    total_oracle_reward = np.sum(results["Oracle"]["reward"])

    pct_oracle = {}
    mean_regret = {}
    for m in models:
        total_m_reward = np.sum(results[m]["reward"])
        pct_oracle[m] = (total_m_reward / total_oracle_reward) * 100
        mean_regret[m] = np.mean(results[m]["regret"])

        print(f"--- {m} ---")
        print(f"Avg Regret per Shift: {mean_regret[m]:.5f}")
        print(f"% of Oracle Reward:   {pct_oracle[m]:.2f}%\n")

    # 2. Bar Chart: Percentage of Oracle Reward
    plt.figure(figsize=(8, 6))


    bars = plt.bar(models, [pct_oracle[m] for m in models], color=colors, edgecolor='black')

    plt.axhline(y=100, color='steelblue', linestyle='--', linewidth=1.5, label='Oracle (100%)')
    plt.title('Percentage of Optimal Oracle Reward Attained', fontsize=14, fontweight='bold', color='#08306b')
    plt.ylabel('Percentage (%)', fontsize=12)

    min_pct = min([pct_oracle[m] for m in models])
    plt.ylim(max(0, min_pct - 10), 105)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom',
                 fontweight='bold')

    plt.legend()
    plt.tight_layout()
    plt.savefig('test_pct_oracle_performance.png', dpi=300)
    plt.close()

    # 3. Boxplot: Distribution of Regret
    regret_data = pd.DataFrame({m: results[m]["regret"] for m in models})

    print("\n--- Boxplot Metrics (Regret Distribution) ---")
    for m in models:
        stats = regret_data[m].describe(percentiles=[.25, .5, .75])
        iqr = stats['75%'] - stats['25%']
        lower_whisker = max(stats['min'], stats['25%'] - 1.5 * iqr)
        upper_whisker = min(stats['max'], stats['75%'] + 1.5 * iqr)

        print(f"{m}:")
        print(f"  Median (Q2):         {stats['50%']:.4f}")
        print(f"  Lower Quartile (Q1): {stats['25%']:.4f}")
        print(f"  Upper Quartile (Q3): {stats['75%']:.4f}")
        print(f"  Lower Whisker:       {lower_whisker:.4f}")
        print(f"  Upper Whisker:       {upper_whisker:.4f}")

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=regret_data, palette=colors, showfliers=False)
    sns.stripplot(data=regret_data, color=".25", alpha=0.3, size=3)

    plt.title('Decision Regret Distribution Across Shifts (Lower is Better)', fontsize=14, fontweight='bold', color='#08306b')
    plt.ylabel('Regret (Oracle Reward - Realized Reward)', fontsize=12, color='#08306b')

    plt.yscale('symlog', linthresh=0.1)

    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    plt.savefig('test_regret_distribution_symlog.png', dpi=300)
    plt.close()

    # --- NEW: Average Scans Per Patient Visualization ---
    avg_scans = {}
    std_scans = {}

    for m in all_models:
        # Calculate the average scans per patient for EACH shift individually
        scans_per_patient_per_shift = np.array(results[m]["scans"]) / N

        # Calculate the overall mean and the standard deviation across all shifts
        avg_scans[m] = np.mean(scans_per_patient_per_shift)
        std_scans[m] = np.std(scans_per_patient_per_shift)

    plt.figure(figsize=(8, 6))

    # Adding a grey color for the Oracle to keep the dark-blue theme intact for the models
    scan_colors = ['mediumblue', 'royalblue', 'deepskyblue', 'darkblue', 'darkgray', 'steelblue']

    # Plot bars with yerr for standard deviation
    bars_scans = plt.bar(all_models,
                         [avg_scans[m] for m in all_models],
                         yerr=[std_scans[m] for m in all_models],
                         capsize=5,  # Adds the horizontal caps to the error bars
                         color=scan_colors,
                         edgecolor='black')

    plt.title('Average Resource Utilization (Scans per Patient)', fontsize=14, fontweight='bold', color='#08306b')
    plt.ylabel('Average Scans', fontsize=12, color='#08306b')

    # Add a bit more headroom so the error bars and text labels don't get cut off
    max_height = max([avg_scans[m] + std_scans[m] for m in all_models])
    plt.ylim(0, max_height * 1.25)  # Increased slightly to fit the longer text

    # Add text labels right above the top of the error bars, including the std dev
    for m, bar in zip(all_models, bars_scans):
        yval = bar.get_height()
        err = std_scans[m]
        # Position the text slightly above the error bar cap
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + err + 0.02,
                 f'{yval:.2f}\n($\\pm${err:.2f})', ha='center', va='bottom',
                 fontweight='bold', fontsize=10, color='#08306b')

    plt.tight_layout()
    plt.savefig('test_average_scans_per_patient.png', dpi=300)
    plt.close()

    # --- 4. Export Summary Metrics to CSV ---
    print("\nExporting summary statistics to CSV...")

    summary_data = []

    # Optional: Add Oracle to the CSV for baseline comparison
    summary_data.append({
        "Model": "Oracle",
        "Clinical_Efficacy_Pct": 100.00,
        "Mean_Regret": 0.0000,
        "Avg_Scans_Per_Patient": round(avg_scans["Oracle"], 4),
        "Regret_Min": 0, "Regret_Lower_Whisker": 0, "Regret_Q1": 0,
        "Regret_Median": 0, "Regret_Q3": 0, "Regret_Upper_Whisker": 0, "Regret_Max": 0
    })

    for m in models:
        stats = regret_data[m].describe(percentiles=[.25, .5, .75])
        iqr = stats['75%'] - stats['25%']
        lower_whisker = max(stats['min'], stats['25%'] - 1.5 * iqr)
        upper_whisker = min(stats['max'], stats['75%'] + 1.5 * iqr)

        summary_data.append({
            "Model": m,
            "Clinical_Efficacy_Pct": round(pct_oracle[m], 2),
            "Mean_Regret": round(mean_regret[m], 4),
            "Avg_Scans_Per_Patient": round(avg_scans[m], 4),  # ADDED
            "Regret_Min": round(stats['min'], 4),
            "Regret_Lower_Whisker": round(lower_whisker, 4),
            "Regret_Q1": round(stats['25%'], 4),
            "Regret_Median": round(stats['50%'], 4),
            "Regret_Q3": round(stats['75%'], 4),
            "Regret_Upper_Whisker": round(upper_whisker, 4),
            "Regret_Max": round(stats['max'], 4)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('simulation_summary_metrics.csv', index=False)
    print("Saved 'simulation_summary_metrics.csv'.")
    print("Saved 'test_pct_oracle_performance.png', 'test_regret_distribution_symlog.png', and "
          "'test_average_scans_per_patient.png'.")


if __name__ == "__main__":
    evaluate_models()