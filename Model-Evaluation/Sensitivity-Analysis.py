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
import gurobipy as gp
from gurobipy import GRB
from sklearn.model_selection import train_test_split
from pyepo.model.grb import optGrbModel

# Import your specific network architectures
from network_architectures import DirectRewardNet, ClassifierNet

# Use Agg backend for matplotlib if running on a headless server (like Della)
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
# 2. FEATURE ENGINEERING & DATA PREP
# ==========================================
def mode(probs): return np.argmax(probs) + 1


def predictive_entropy(probs, eps=1e-8):
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


def kl_divergence(p, q, eps=1e-8):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def addFeatures(trueState, time, patientProbs, prevPatientProbs, last_observed_state, time_since_last_scan):
    mHat = mode(patientProbs)
    mHatLast = mode(prevPatientProbs)
    uncert = predictive_entropy(patientProbs)
    prevUncert = predictive_entropy(prevPatientProbs)
    diverge = kl_divergence(patientProbs, prevPatientProbs)

    output = np.zeros(23, dtype='float')
    output[0] = trueState if not np.isnan(trueState) else 0.0
    output[1] = mHat
    output[2] = mHatLast
    output[3] = mHat - mHatLast
    output[4] = uncert
    output[5] = prevUncert
    output[6] = uncert - prevUncert
    output[7] = diverge
    output[8] = time
    for i in range(4):
        output[i + 9] = patientProbs[i]
        output[i + 13] = prevPatientProbs[i]
        output[i + 17] = patientProbs[i] - prevPatientProbs[i]

    output[21] = last_observed_state if not np.isnan(last_observed_state) else 0.0
    output[22] = time_since_last_scan
    return output


class Patient:
    def __init__(self, patient_id, patient_rewards, delay_values, features, patient_scans, true_states, T):
        self.id = patient_id
        self.features = np.copy(features)
        self.T = T
        self.rewards = patient_rewards
        self.delay_values = delay_values
        self.scans = patient_scans
        self.true_states = true_states

        # Simulation State Trackers
        self.current_t = 0
        self.last_observed_state = self.features[0, 21]
        self.time_since_last_scan = 0
        self.active = True

    def get_targets(self, t):
        # Safety: Ensure t does not exceed the data bounds
        effective_t = min(t, self.T - 1)

        if effective_t + 3 > self.T:
            padding = np.zeros(3 - (self.T - effective_t))
            rewards_window = np.concatenate([self.rewards[effective_t:], padding])
        else:
            rewards_window = self.rewards[effective_t:effective_t + 3]

        delay_val = np.array([self.delay_values[effective_t]])
        return np.concatenate([rewards_window, delay_val])

    def get_current_features(self, window=2):
        indices = np.arange(self.current_t - window + 1, self.current_t + 1)
        indices = np.maximum(indices, 0)
        context = np.copy(self.features[indices, :])
        context[-1, 21] = self.last_observed_state
        context[-1, 22] = self.time_since_last_scan
        return context.flatten()

    def get_last_clinical_scan(self, current_global_time):
        past_scans = np.where(self.scans[:self.current_t + 1] == 1)[0]
        if len(past_scans) > 0:
            local_tA = past_scans[-1]
            global_tA = current_global_time - (self.current_t - local_tA)
            return global_tA, self.true_states[local_tA]
        return current_global_time - self.current_t, self.true_states[0]

    def get_next_clinical_scan(self, current_global_time):
        future_scans = np.where(self.scans[self.current_t + 1:] == 1)[0]
        if len(future_scans) > 0:
            local_tB = future_scans[0] + self.current_t + 1
            global_tB = current_global_time + (local_tB - self.current_t)
            return global_tB, self.true_states[local_tB]
        return None, None

    def update_after_scan(self, revealed_state):
        self.last_observed_state = revealed_state
        self.time_since_last_scan = 0
        self._advance_time()

    def update_no_scan(self):
        self.time_since_last_scan += 1
        self._advance_time()

    def _advance_time(self):
        self.current_t += 1
        if self.current_t >= self.T:
            self.active = False


def create_patient_objects(df, horizon_T=3):
    patient_objects = []
    for pid, group in df.groupby('ptid_idx'):
        group = group.sort_values('hour').reset_index(drop=True)
        scan_indices = group.index[group['true_scans'] == 1].tolist()
        if not scan_indices: continue

        first_scan_idx = scan_indices[0]
        remaining_stay_df = group.iloc[first_scan_idx:].reset_index(drop=True)
        total_stay = len(remaining_stay_df)

        if total_stay >= horizon_T:
            initial_state = remaining_stay_df['last_known_mls_class'].iloc[0]
            probs_matrix = remaining_stay_df[['proba_0', 'proba_1', 'proba_2', 'proba_3']].values
            rewards = remaining_stay_df['proxy_reward'].values
            delay_values = remaining_stay_df['scan_delay_value'].values
            scans = remaining_stay_df['true_scans'].values
            hours = remaining_stay_df['hour'].values
            true_states = remaining_stay_df['last_known_mls_class'].values

            feature_list = []
            last_scan_hour = hours[0]
            last_observed_state = initial_state

            for t in range(total_stay):
                if scans[t] == 1:
                    last_scan_hour = hours[t]
                    last_observed_state = remaining_stay_df['last_known_mls_class'].iloc[t]

                time_since_last_scan = hours[t] - last_scan_hour
                current_probs = probs_matrix[t]
                prev_probs = probs_matrix[t - 1] if t > 0 else (np.ones(4) / 4.0)

                feat_vec = addFeatures(initial_state, hours[t], current_probs, prev_probs,
                                       last_observed_state, time_since_last_scan)
                feature_list.append(feat_vec)

            p = Patient(patient_id=pid, patient_rewards=rewards, delay_values=delay_values,
                        features=np.array(feature_list), patient_scans=scans,
                        true_states=true_states, T=total_stay)
            patient_objects.append(p)
    return patient_objects


# ==========================================
# 3. MODELS & WRAPPERS
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

        def get_x(i, t):
            return x[i * (self.T + 1) + t]

        def get_x_later(i):
            return x[i * (self.T + 1) + self.T]

        for t in range(self.T):
            m.addConstr(gp.quicksum(get_x(i, t) for i in range(self.N)) <= self.R)

        for i in range(self.N):
            m.addConstr(
                gp.quicksum(get_x(i, t) for t in range(self.T)) + get_x_later(i) == 1
            )
        return m, x


class ExpectedValueBridge(torch.nn.Module):
    def __init__(self, classifier_model, mean_action_rewards, device='cpu'):
        super().__init__()
        self.model = classifier_model
        self.mean_action_rewards = torch.tensor(mean_action_rewards, dtype=torch.float32).to(device)
        self.device = device

    def forward(self, x):
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        return probs * self.mean_action_rewards


class PerfectForesightOracle(torch.nn.Module):
    def __init__(self, active_patients_ref, T=3, device='cpu'):
        super().__init__()
        self.active_patients_ref = active_patients_ref
        self.T = T
        self.device = device

    def forward(self, x):
        true_targets = [p.get_targets(p.current_t) for p in self.active_patients_ref]
        return torch.tensor(np.array(true_targets), dtype=torch.float32).to(self.device)


# ==========================================
# 4. SIMULATOR & ENVIRONMENT
# ==========================================
class RollingHorizonSimulator:
    def __init__(self, reward_net, scheduling_model, N=30, T=3, R=3, Q=6, delta=3, device='cpu', debug_name=None):
        self.reward_net = reward_net
        self.reward_net.eval()
        self.opt_model = scheduling_model
        self.N, self.T, self.R, self.Q, self.delta = N, T, R, Q, delta
        self.device = device
        self.debug_name = debug_name
        self.C = np.zeros(self.N)
        self.L = np.full(self.N, -np.inf)

    def step(self, current_time, active_patients):
        current_features = np.array([p.get_current_features() for p in active_patients])
        x_tensor = torch.tensor(current_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.reward_net(x_tensor).cpu().numpy().flatten()

        # 1. Get a fresh Gurobi model and variables
        m, x_vars = self.opt_model._getModel()

        # 2. Set the objective directly on this fresh model
        m.setObjective(gp.quicksum(preds[j] * x_vars[j] for j in range(len(preds))), GRB.MAXIMIZE)

        # 3. Apply dynamic patient-specific constraints
        for i in range(self.N):
            # Constraint: Refractory Period (Delta)
            time_since_last = current_time - self.L[i]
            if time_since_last < self.delta:
                forbidden_steps = int(self.delta - time_since_last)
                for t in range(min(forbidden_steps, self.T)):
                    m.addConstr(x_vars[i * (self.T + 1) + t] == 0, name=f"refractory_p{i}_t{t}")

            # Constraint: Max Scans per Patient (Q)
            if self.C[i] >= self.Q:
                # Force all scanning actions for this patient in the planning horizon to 0
                for t in range(self.T):
                    m.addConstr(x_vars[i * (self.T + 1) + t] == 0, name=f"max_scans_p{i}_t{t}")

        # 4. Optimize our custom model (bypassing PyEPO's internal solver)
        m.setParam("OutputFlag", 0)
        m.optimize()

        # 5. Extract immediate decisions (t=0)
        immediate_decisions = []
        for i in range(self.N):
            immediate_decisions.append(x_vars[i * (self.T + 1)].X)

        immediate_decisions = np.array(immediate_decisions)
        scanned_patients = np.where(immediate_decisions > 0.5)[0].tolist()

        # Enforce ward capacity tie-breaking (R)
        if len(scanned_patients) > self.R:
            triage_data = [(i, active_patients[i].last_observed_state, active_patients[i].time_since_last_scan) for i in
                           scanned_patients]
            triage_data.sort(key=lambda x: (x[1], x[2]), reverse=True)
            scanned_patients = [data[0] for data in triage_data[:self.R]]

        # Update patient state and tracking variables
        for i in scanned_patients:
            self.C[i] += 1
            self.L[i] = current_time

        for i, p in enumerate(active_patients):
            if i in scanned_patients:
                t_A, X_A = p.get_last_clinical_scan(current_time)
                t_B, X_B = p.get_next_clinical_scan(current_time)
                if t_A == t_B or t_B is None:
                    p.update_after_scan(X_A)
                else:
                    p_ratio = (current_time - t_A) / (t_B - t_A)
                    x_cont = X_A + p_ratio * (X_B - X_A)
                    p.update_after_scan(
                        np.ceil(x_cont) if np.random.rand() < (x_cont - np.floor(x_cont)) else np.floor(x_cont))
            else:
                p.update_no_scan()

        return scanned_patients


class WardEnvironment:
    def __init__(self, name, simulator_instance, is_clinical_baseline=False):
        self.name = name
        self.simulator = simulator_instance
        self.is_clinical = is_clinical_baseline
        self.patients = []
        self.scan_log = []

    def assign_initial_patients(self, initial_patients):
        self.patients = [copy.deepcopy(p) for p in initial_patients]

    def replace_discharged_patients(self, incoming_patients_dict):
        for i, p in enumerate(self.patients):
            if not p.active:
                self.patients[i] = copy.deepcopy(incoming_patients_dict[i])


# ==========================================
# 5. EVALUATION METRICS
# ==========================================
def format_scan_log(scan_log):
    patient_scans = defaultdict(list)
    for entry in scan_log:
        patient_scans[entry['patient_id']].append(entry['hour'])
    for pid in patient_scans:
        patient_scans[pid].sort()
    return patient_scans


def evaluate_peak_detection(oracle_log, model_log, tau=1):
    oracle_scans, model_scans = format_scan_log(oracle_log), format_scan_log(model_log)
    tp, fp, fn = 0, 0, 0
    raw_offsets = []

    all_patients = set(oracle_scans.keys()).union(set(model_scans.keys()))
    for pid in all_patients:
        o_hours, m_hours = oracle_scans.get(pid, []), model_scans.get(pid, []).copy()

        if not m_hours: fn += len(o_hours); continue
        if not o_hours: fp += len(m_hours); continue

        for o_h in o_hours:
            if not m_hours: fn += 1; continue
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
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'Precision': prec, 'Recall': rec, 'F1-Score': f1,
            'Mean Raw Offset (hrs)': np.mean(raw_offsets) if raw_offsets else 0.0}


# ==========================================
# 6. SENSITIVITY ANALYSIS EXECUTION
# ==========================================
from solve_global_dynamic import solve_dynamic_global_oracle


def apply_distribution_shift(patient, noise_std=0.15, noise_rng=None):
    """
    Injects Gaussian noise using an isolated RNG to prevent state leakage.
    """
    p = copy.deepcopy(patient)
    if noise_rng is None:
        noise_rng = np.random

    noise = noise_rng.normal(0, noise_std, p.features[:, 9:17].shape)
    p.features[:, 9:17] += noise
    p.features[:, 9:17] = np.clip(p.features[:, 9:17], 0.0, 1.0)

    row_sums_curr = p.features[:, 9:13].sum(axis=1, keepdims=True)
    row_sums_curr[row_sums_curr == 0] = 1e-8
    p.features[:, 9:13] /= row_sums_curr

    row_sums_prev = p.features[:, 13:17].sum(axis=1, keepdims=True)
    row_sums_prev[row_sums_prev == 0] = 1e-8
    p.features[:, 13:17] /= row_sums_prev
    return p


def run_sensitivity_analysis():
    print("Loading Test Set and Models...")
    dfV = pd.read_csv("HELMET_Triangular_Targets_with_V.csv")
    patientArray = create_patient_objects(dfV)

    train_patients, temp_patients = train_test_split(patientArray, test_size=0.30, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)
    T = 3

    print("Precomputing empirical mean rewards...")
    all_targets = [p.get_targets(t) for group in [train_patients, val_patients] for p in group for t in range(p.T - 2)]
    mean_action_rewards = np.array(all_targets).mean(axis=0)

    print("Extracting optimal parameters and loading models...")
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
    Q, delta = 6, 3  # Add constraint limits

    scenarios = {
        "Baseline": {
            "cap_func": lambda t: BASE_R,
            "shift_func": lambda t: False
        },
        "Mid-Sim Breakdown": {
            "cap_func": lambda t: BASE_R - 1 if 400 <= t <= 600 else BASE_R,
            "shift_func": lambda t: False
        },
        "Mid-Sim Boost": {
            "cap_func": lambda t: BASE_R + 1 if 400 <= t <= 600 else BASE_R,
            "shift_func": lambda t: False
        },
        "Mid-Sim Dist. Shift": {
            "cap_func": lambda t: BASE_R,
            "shift_func": lambda t: True if 400 <= t <= 600 else False
        }
    }

    all_scenario_logs = {scenario_name: {} for scenario_name in scenarios.keys()}
    scenario_patient_counts = {}

    for scenario_name, config in scenarios.items():
        print(f"\n{'=' * 45}\nRunning Scenario: {scenario_name}\n{'=' * 45}")

        # 1. Establish isolated Random Number Generators for this scenario
        patient_rng = np.random.RandomState(42)
        noise_rng = np.random.RandomState(999)

        # Trackers specifically for the clinical baseline
        clinical_C = np.zeros(N)
        clinical_L = np.full(N, -np.inf)

        eval_model = NeuroICUSchedulingModel(N=N, T=T, R=BASE_R)
        oracle_patient_ref = []
        model_rolling_oracle = PerfectForesightOracle(oracle_patient_ref, T, device)

        wards = [
            WardEnvironment("MSE", RollingHorizonSimulator(model_mse, eval_model, N=N, T=T, R=BASE_R, device=device)),
            WardEnvironment("Huber",
                            RollingHorizonSimulator(model_huber, eval_model, N=N, T=T, R=BASE_R, device=device)),
            WardEnvironment("SPO+", RollingHorizonSimulator(model_spo, eval_model, N=N, T=T, R=BASE_R, device=device)),
            WardEnvironment("Classifier",
                            RollingHorizonSimulator(bridge_classifier, eval_model, N=N, T=T, R=BASE_R, device=device)),
            WardEnvironment("Rolling Oracle",
                            RollingHorizonSimulator(model_rolling_oracle, eval_model, N=N, T=T, R=BASE_R,
                                                    device=device)),
            WardEnvironment("Clinical Baseline", None, is_clinical_baseline=True)
        ]

        global_patient_trace = []
        patient_arrival_counter = 0

        # Draw indices using isolated patient RNG to guarantee identical sequences
        initial_indices = patient_rng.choice(len(test_patients), N, replace=False)
        raw_initial_pool = [copy.deepcopy(test_patients[idx]) for idx in initial_indices]

        current_shift_initial = config["shift_func"](0)
        initial_pool = []
        for p in raw_initial_pool:
            new_p = apply_distribution_shift(p, noise_std=0.15, noise_rng=noise_rng) if current_shift_initial else p

            # 1. Select a random start time within their length of stay
            start_t = patient_rng.randint(0, new_p.T)
            new_p.current_t = start_t

            # 2. Update their observation trackers relative to this new start time
            past_scans = np.where(new_p.scans[:start_t + 1] == 1)[0]
            if len(past_scans) > 0:
                last_scan_idx = past_scans[-1]
                new_p.last_observed_state = new_p.true_states[last_scan_idx]
                new_p.time_since_last_scan = start_t - last_scan_idx
            else:
                new_p.last_observed_state = new_p.features[0, 21]
                new_p.time_since_last_scan = start_t

            new_p.episode_id = f"{new_p.id}_0"

            global_patient_trace.append({
                "patient_id": new_p.episode_id,
                "arrival_hour": 0,
                "length_of_stay": new_p.T - start_t,  # Reflect remaining time
                "true_rewards": copy.deepcopy(new_p.rewards)
            })
            initial_pool.append(new_p)
            patient_arrival_counter += 1

        # FIX 1: Provide a pristine deepcopy of the initial pool to EVERY ward
        for w in wards:
            w.assign_initial_patients(copy.deepcopy(initial_pool))
            if w.name == "Rolling Oracle": oracle_patient_ref.extend(w.patients)

        for global_hour in range(SIM_HOURS):
            current_R = config["cap_func"](global_hour)
            current_shift = config["shift_func"](global_hour)

            eval_model.R = current_R
            for w in wards:
                if not w.is_clinical:
                    w.simulator.R = current_R
                    w.simulator.opt_model.R = current_R

            incoming_this_hour = {}
            for i in range(N):
                # We check wards[0] safely because LOS (T=3) guarantees synchronous discharges
                if not wards[0].patients[i].active:
                    idx = patient_rng.choice(len(test_patients))
                    raw_new_p = copy.deepcopy(test_patients[idx])

                    new_p = apply_distribution_shift(raw_new_p, noise_std=0.15,
                                                     noise_rng=noise_rng) if current_shift else raw_new_p

                    unique_episode_id = f"{new_p.id}_{global_hour}"
                    new_p.episode_id = unique_episode_id
                    incoming_this_hour[i] = new_p

                    # Track for the Global Oracle
                    global_patient_trace.append({
                        "patient_id": unique_episode_id,
                        "arrival_hour": global_hour,
                        "length_of_stay": new_p.T,
                        "true_rewards": copy.deepcopy(new_p.rewards)
                    })

            if incoming_this_hour:
                for w in wards:
                    # FIX 2: Provide a pristine deepcopy of the incoming patients to EVERY ward
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
                        if p.scans[p.current_t] == 1:
                            if clinical_C[i] < Q and (global_hour - clinical_L[i]) >= delta:
                                desired_scans.append(i)

                    if len(desired_scans) > current_R:
                        triage_data = [(i, w.patients[i].last_observed_state, w.patients[i].time_since_last_scan) for i
                                       in desired_scans]
                        triage_data.sort(key=lambda x: (x[1], x[2]), reverse=True)
                        scanned_indices = [data[0] for data in triage_data[:current_R]]
                    else:
                        scanned_indices = desired_scans

                    for i, p in enumerate(w.patients):
                        if i in scanned_indices:
                            p.update_after_scan(p.true_states[p.current_t])
                            # Update clinical trackers upon successful scan
                            clinical_C[i] += 1
                            clinical_L[i] = global_hour
                        else:
                            p.update_no_scan()
                else:
                    scanned_indices = w.simulator.step(global_hour, w.patients)

                for i in scanned_indices:
                    w.scan_log.append({'patient_id': w.patients[i].episode_id, 'hour': global_hour})

            if (global_hour + 1) % 250 == 0:
                print(f"[{scenario_name}] Completed Hour {global_hour + 1}/{SIM_HOURS} (Scanners: {current_R})")

        # Track how many total patients ran through the ward in this scenario
        scenario_patient_counts[scenario_name] = len(global_patient_trace)

        for w in wards:
            all_scenario_logs[scenario_name][w.name] = w.scan_log

        print(f"[{scenario_name}] Solving Global Static Oracle (Full Hindsight)...")
        global_log = solve_dynamic_global_oracle(global_patient_trace, config["cap_func"], SIM_HOURS=SIM_HOURS)
        all_scenario_logs[scenario_name]["Global Oracle"] = global_log

    # ==========================================
    # 7. SENSITIVITY EVALUATION & PLOTTING
    # ==========================================
    print("\nEvaluating Peak Detection Metrics Against Global Oracle...")
    plot_data = []

    for scenario_name, logs in all_scenario_logs.items():
        # GROUND TRUTH is now the perfect hindsight Global Oracle
        oracle_log = logs["Global Oracle"]
        total_scenario_patients = scenario_patient_counts[scenario_name]

        for model_name, model_log in logs.items():

            # Calculate and append Average Scans per Patient metric for all policies
            total_scans = len(model_log)
            avg_scans = total_scans / total_scenario_patients if total_scenario_patients > 0 else 0

            plot_data.append({
                'Scenario': scenario_name,
                'Policy': model_name,
                'Metric': 'Avg Scans/Patient',
                'Score': avg_scans
            })

            # Skip F1/Offset evaluation against itself for the Global Oracle
            if model_name == "Global Oracle": continue

            res = evaluate_peak_detection(oracle_log, model_log, tau=1)

            plot_data.extend([
                {'Scenario': scenario_name, 'Policy': model_name, 'Metric': 'F1-Score (%)',
                 'Score': res['F1-Score'] * 100},
                {'Scenario': scenario_name, 'Policy': model_name, 'Metric': 'Mean Offset (hrs)',
                 'Score': res['Mean Raw Offset (hrs)']}
            ])

    df_plot = pd.DataFrame(plot_data)

    # Establish specific, consistent colors mapped explicitly by policy
    custom_palette = {
        'MSE': 'mediumblue',
        'Huber': 'royalblue',
        'SPO+': 'darkblue',
        'Classifier': 'deepskyblue',
        'Rolling Oracle': 'steelblue',
        'Clinical Baseline': 'darkgray',
        'Global Oracle': '#c6dbef'
    }

    # --- Plot 1: F1-Score ---
    df_f1 = df_plot[df_plot['Metric'] == 'F1-Score (%)']
    plt.figure(figsize=(14, 7))
    ax1 = sns.barplot(data=df_f1, x='Scenario', y='Score', hue='Policy', palette=custom_palette, edgecolor=".2")
    plt.title('Policy Resilience vs. Global Oracle: F1-Score', fontsize=18, fontweight='bold', color='#08306b')
    plt.ylabel('F1-Score (%)', fontsize=14, fontweight='bold', color='#08306b')
    plt.xlabel('Simulation Environment', fontsize=14, fontweight='bold', color='#08306b')
    plt.ylim(0, 110)

    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=7, fontweight='bold', color='#08306b', xytext=(0, 4),
                         textcoords='offset points')

    plt.legend(title='Decision Policy', loc='upper right', bbox_to_anchor=(1.18, 1))
    plt.tight_layout()
    plt.savefig('global_sensitivity_f1.png', dpi=300)
    plt.close()

    # --- Plot 2: Mean Temporal Offset ---
    df_offset = df_plot[df_plot['Metric'] == 'Mean Offset (hrs)']
    plt.figure(figsize=(14, 7))
    ax2 = sns.barplot(data=df_offset, x='Scenario', y='Score', hue='Policy', palette=custom_palette, edgecolor=".2")
    plt.title('Policy Timing Accuracy vs. Global Oracle', fontsize=18, fontweight='bold', color='#08306b')
    plt.ylabel('Mean Offset (Hours)', fontsize=14, fontweight='bold', color='#08306b')
    plt.xlabel('Simulation Environment', fontsize=14, fontweight='bold', color='#08306b')

    max_abs_val = df_offset['Score'].abs().max()
    y_limit = max_abs_val * 1.2 if max_abs_val > 0 else 1
    plt.ylim(-y_limit, y_limit)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')

    for p in ax2.patches:
        height = p.get_height()
        if height != 0:
            y_offset = 4 if height > 0 else -12
            va_align = 'bottom' if height > 0 else 'top'
            ax2.annotate(f"{height:.2f}h", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va=va_align, fontsize=7, fontweight='bold', color='#08306b', xytext=(0, y_offset),
                         textcoords='offset points')

    plt.legend(title='Decision Policy', loc='upper right', bbox_to_anchor=(1.18, 1))
    plt.tight_layout()
    plt.savefig('global_sensitivity_offset.png', dpi=300)
    plt.close()

    # --- Plot 3: Average Scans Per Patient ---
    df_scans = df_plot[df_plot['Metric'] == 'Avg Scans/Patient']
    plt.figure(figsize=(14, 7))
    ax3 = sns.barplot(data=df_scans, x='Scenario', y='Score', hue='Policy', palette=custom_palette, edgecolor=".2")
    plt.title('Average Scans per Patient vs. Simulation Environment', fontsize=18, fontweight='bold', color='#08306b')
    plt.ylabel('Avg Scans / Patient', fontsize=14, fontweight='bold', color='#08306b')
    plt.xlabel('Simulation Environment', fontsize=14, fontweight='bold', color='#08306b')

    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=7, fontweight='bold', color='#08306b', xytext=(0, 4),
                         textcoords='offset points')

    plt.legend(title='Decision Policy', loc='upper right', bbox_to_anchor=(1.18, 1))
    plt.tight_layout()
    plt.savefig('global_sensitivity_avg_scans.png', dpi=300)
    plt.close()

    print(
        "\nSensitivity visualizations saved as 'global_sensitivity_f1.png', 'global_sensitivity_offset.png', and 'global_sensitivity_avg_scans.png'.")


if __name__ == "__main__":
    run_sensitivity_analysis()