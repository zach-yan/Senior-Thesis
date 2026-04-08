import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import optuna

# Import your architectures and data loaders
from network_architectures import DirectRewardNet

# Force matplotlib to not use any Xwindows backend for the cluster
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
# 1. SETUP & UTILS
# ==========================================
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dark_blue_colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
thesis_cmap = LinearSegmentedColormap.from_list("ThesisBlue", dark_blue_colors)

# ==========================================
# 2. HELPER FUNCTIONS & DATA LOGIC
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
    def __init__(self, patient_id, patient_rewards, delay_values, features, patient_scans, T):
        self.id = patient_id
        self.features = features
        self.T = T
        self.rewards = patient_rewards
        self.delay_values = delay_values
        self.scans = patient_scans

    def get_context(self, t, window=2):
        indices = np.arange(t - window + 1, t + 1)
        indices = np.maximum(indices, 0)
        return self.features[indices, :].flatten()

    def get_targets(self, t):
        if t + 3 > self.T:
            padding = np.zeros(3 - (self.T - t))
            rewards_window = np.concatenate([self.rewards[t:], padding])
        else:
            rewards_window = self.rewards[t:t + 3]

        delay_val = np.array([self.delay_values[t]])
        return np.concatenate([rewards_window, delay_val])


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
                        features=np.array(feature_list), patient_scans=scans, T=total_stay)
            patient_objects.append(p)
    return patient_objects


class ShapWrapper(nn.Module):
    """
    Wraps the model to output a single value (the reward for Action 0: 'Scan Now')
    from a 2D input tensor of shape (batch, 46).
    """

    def __init__(self, base_model):
        super(ShapWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # x shape: (batch_size, 46)
        out = self.base_model(x)
        # out shape: (batch_size, 4) -> [Scan t, Scan t+1, Scan t+2, Defer]

        # We extract index 0 ("Scan Now" at time t) and make it (batch_size, 1)
        return out[:, 0].unsqueeze(1)


# ==========================================
# 3. DATA PREPARATION FOR SHAP
# ==========================================
def prepare_tensors_for_shap(patients, sample_size=300):
    """
    Perfectly mimics 'sample_natural_hospital_shifts_pyepo' from the training script.
    Extracts a single 46-dimensional context vector per patient.
    """
    # Ensure patients have enough history
    valid_patients = [p for p in patients if p.T > 0]
    sampled = np.random.choice(valid_patients, min(sample_size, len(valid_patients)), replace=False)

    tensor_list = []
    for p in sampled:
        # Pick a random timestep in the patient's stay, just like training
        t = np.random.randint(0, p.T)
        context = p.get_context(t)  # This naturally returns the 46-length array
        tensor_list.append(context)

    feature_matrix = np.stack(tensor_list)
    return torch.tensor(feature_matrix, dtype=torch.float32).to(device)


# ==========================================
# 4. MAIN EXTRACTION FUNCTION
# ==========================================
def run_shap_extraction():
    print(f"Starting TRUE SHAP Extraction on {device}...")
    set_seeds(42)

    # 1. Load Data
    print("Loading datasets...")
    dfV = pd.read_csv("HELMET_Triangular_Targets_with_V.csv")
    patientArray = create_patient_objects(dfV)

    train_patients, temp_patients = train_test_split(patientArray, test_size=0.30, random_state=42)
    _, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)

    # 2. Build Tensors (Now standard 2D matrices: Batch x 46)
    background_tensor = prepare_tensors_for_shap(train_patients, sample_size=200)
    test_tensor = prepare_tensors_for_shap(test_patients, sample_size=500)

    # Feature Names Initialization
    feature_names = [f"Feature_{i}" for i in range(46)]

    # --- Timestep t-1 (Indices 0 to 22) ---
    feature_names[0:9] = [
        "admissionState_t-1", "mHat_t-1", "mHatLast_t-1", "mHat-mHatLast_t-1",
        "uncert_t-1", "prevUncert_t-1", "uncert_t-1 - prevUncert_t-1",
        "diverge_t-1", "time_t-1"
    ]

    feature_names[9:13] = [
        "Prob_State_1_t-1", "Prob_State_2t-1",
        "Prob_State_3_t-1", "Prob_State_4_t-1"
    ]
    feature_names[13:17] = [
        "prevProb_State_1_t-1", "prevProb_State_2_t-1",
        "prevProb_State_3_t-1", "prevProb_State_4_t-1"
    ]
    feature_names[17:21] = [
        "Prob_State_1_t-1 - prevProb_State_1_t-1", "Prob_State_2_t-1 - prevProb_State_2_t-1",
        "Prob_State_3_t-1 - prevProb_State_3_t-1", "prevProb_State_4_t-1 - prevProb_State_4_t-1",

    ]
    feature_names[21:23] = [
        "last_observed_state_t-1", "time_since_last_scan_t-1"
    ]

    # --- Timestep t (Indices 23 to 45) ---
    feature_names[23:32] = [
        "admissionState_t", "mHat_t", "mHatLast_t", "mHat-mHatLast_t",
        "uncert_t", "prevUncert_t", "uncert_t - prevUncert_t",
        "diverge_t", "time_t"
    ]
    feature_names[32:36] = [
        "t_Prob_State_1", "t_Prob_State_2",
        "t_Prob_State_3", "t_Prob_State_4"
    ]
    feature_names[36:40] = [
        "t_prevProb_State_1", "t_prevProb_State_2",
        "t_prevProb_State_3", "t_prevProb_State_4"
    ]
    feature_names[40:44] = [
        "Prob_State_1_t - prevProb_State_1_t", "Prob_State_2_t - prevProb_State_2_t",
        "Prob_State_3_t - prevProb_State_3_t", "prevProb_State_4_t - prevProb_State_4_t",

    ]
    feature_names[44:46] = [
        "last_observed_state_t", "time_since_last_scan_t"
    ]
    # 3. Load Model
    print("Loading SPO+ Model...")
    spo_params = optuna.load_study(study_name='new_spo_dfl_study',
                                   storage='sqlite:///optuna_new_spo_dfl.db').best_params
    model_spo = DirectRewardNet(input_dim=46, T=3, hidden_dim=spo_params['hidden_dim']).to(device)
    model_spo.load_state_dict(torch.load('final_spo.pth'))
    model_spo.eval()

    wrapped_model = ShapWrapper(model_spo).to(device)

    # ==========================================
    # 4. COMPUTE PERMUTATION FEATURE IMPORTANCE
    # ==========================================
    print("Calculating Permutation Feature Importance (Black-Box)...")

    # 1. Get baseline "Scan Now" predictions (No scrambling)
    baseline_output = wrapped_model(test_tensor).detach().cpu().numpy()

    importances = {}

    # 2. Iterate through all 46 features
    for feature_idx in range(46):
        # Clone the tensor so we don't permanently destroy the data
        shuffled_tensor = test_tensor.clone()

        # Scramble ONLY this specific feature across all patients in the batch
        perm = torch.randperm(shuffled_tensor.size(0))
        shuffled_tensor[:, feature_idx] = shuffled_tensor[perm, feature_idx]

        # 3. Get predictions with the scrambled feature
        shuffled_output = wrapped_model(shuffled_tensor).detach().cpu().numpy()

        # 4. Calculate the Mean Absolute Error (MAE) between baseline and scrambled
        # High MAE means the feature was critical to the model's decision
        mae_difference = np.mean(np.abs(baseline_output - shuffled_output))
        importances[feature_names[feature_idx]] = mae_difference

    # ==========================================
    # 5. GENERATE VISUALIZATIONS & EXPORTS
    # ==========================================
    # --- NEW CSV EXPORT CODE ---
    print("Saving all feature importances to CSV...")

    # Sort features by importance for the CSV (highest MAE impact at the top)
    sorted_importances_desc = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame and save to CSV
    importance_df = pd.DataFrame(sorted_importances_desc, columns=['Feature', 'Mean_Absolute_Change'])
    importance_df.to_csv('feature_importances_spo_plus.csv', index=False)
    print("Saved all feature importances to feature_importances_spo_plus.csv")
    # ---------------------------

    print("Generating Permutation Importance Plot...")

    # Sort features by importance ascending for the horizontal bar chart
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=False)

    # Keep the top 15 most important features to make the plot clean and readable
    top_features = sorted_importances[-15:]

    # FIX 1: Replace underscores with spaces so the cmr10 font doesn't break
    features_to_plot = [x[0].replace('_', ' ') for x in top_features]
    importance_scores = [x[1] for x in top_features]

    plt.figure(figsize=(10, 8))
    plt.barh(features_to_plot, importance_scores, color='#2171b5')
    # FIX 2: Remove fontweight='bold' because cmr10 does not have a bold weight
    plt.xlabel("Mean Absolute Change in `Scan Now' Expected Proxy Reward", fontsize=12)
    plt.title("SPO+ Policy: Permutation Feature Importance", fontsize=16, color='#08306b')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('permutation_importance_spo_plus.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Extraction complete. Saved plot as permutation_importance_spo_plus.png")


if __name__ == "__main__":
    run_shap_extraction()
