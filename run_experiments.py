"""
run_experiments.py
Complete pipeline for: 
- Single-cell network simulation
- Greedy SNR-based labeling
- Supervised learning (DT, RF, MLP)
- Baselines (Random, Greedy)
- Evaluation: imitation accuracy, throughput, fairness, outage, inference time
"""

import os
import tempfile
import numpy as np
import time
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# ==================== 1. Configuration ====================
CONFIG = {
    'num_snapshots': 50000,   # 增加数据量
    'num_users_min': 8,
    'num_users_max': 8,
    'num_channels_min': 8,
    'num_channels_max': 8,
    'cell_radius_m': 500,
    'tx_power_dbm': 30,
    'noise_psd_dbm_per_hz': -174,
    'bandwidth_hz': 1e6,
    'path_loss_exponent': 2.5,
    'ref_distance_m': 1.0,
    'ref_path_loss_db': 32.4,
    'carrier_freq_ghz': 2.4,
    'snr_threshold_db': 0,
    'test_ratio': 0.2,
    'random_seed': 42,
}


np.random.seed(CONFIG['random_seed'])

# ==================== 2. Simulation Helpers ====================
def generate_user_positions(num_users, radius):
    """Uniform random positions in a circle of given radius."""
    r = radius * np.sqrt(np.random.uniform(0, 1, num_users))
    theta = np.random.uniform(0, 2*np.pi, num_users)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def path_loss_db(distance_m, ref_dist=1.0, pl_exponent=3.5, ref_pl_db=32.4):
    """Log-distance path loss model."""
    return ref_pl_db + 10 * pl_exponent * np.log10(distance_m / ref_dist + 1e-9)

def rayleigh_fading(num_users, num_channels):
    """Generate Rayleigh fading coefficients for each user and channel.
       Returns a matrix of linear gains (mean power = 1)."""
    h = np.random.rayleigh(scale=np.sqrt(0.5), size=(num_users, num_channels)) * np.sqrt(2)
    # Alternative: complex Gaussian, but magnitude is Rayleigh.
    # For simplicity, we use real Rayleigh with power E[|h|^2]=1.
    return h

def compute_snr_matrix(x, y, tx_power_dbm, noise_psd_dbm_hz, bandwidth_hz,
                       ref_dist, pl_exponent, ref_pl_db, carrier_freq_ghz, num_channels):
    """
    Compute SNR (linear) for each user and each channel.
    Returns matrix shape (num_users, num_channels).
    """
    distance = np.sqrt(x**2 + y**2) + 1e-3  # avoid zero
    pl_db = path_loss_db(distance, ref_dist, pl_exponent, ref_pl_db)
    # Add small frequency-selective variation: each channel sees additional random loss
    # to make channels not identical. This is realistic and makes ML problem non-trivial.
    freq_variation_db = np.random.normal(0, 2, size=(len(x), num_channels))
    total_loss_db = pl_db[:, np.newaxis] + freq_variation_db
    
    # Convert to linear
    rx_power_linear = 10**((tx_power_dbm - total_loss_db) / 10)
    noise_power_linear = 10**((noise_psd_dbm_hz + 10*np.log10(bandwidth_hz)) / 10)
    snr_linear = rx_power_linear / noise_power_linear
    # Apply Rayleigh fading (multiplicative, per user-channel)
    fading_gain = rayleigh_fading(len(x), num_channels)
    snr_linear = snr_linear * (fading_gain ** 2)
    return snr_linear

def greedy_allocate(snr_matrix):
    U, K = snr_matrix.shape
    best_snr_per_user = np.max(snr_matrix, axis=1)
    sorted_users = np.argsort(best_snr_per_user)[::-1]
    assigned_channel = np.full(U, -1, dtype=int)
    used_channels = set()
    for u in sorted_users:
        available = [c for c in range(K) if c not in used_channels]
        if not available:
            break
        snr_available = snr_matrix[u, available]
        best_c = available[np.argmax(snr_available)]
        assigned_channel[u] = best_c
        used_channels.add(best_c)
    if not hasattr(greedy_allocate, 'printed'):
        print(f"Unassigned users: {np.sum(assigned_channel == -1)}")
        greedy_allocate.printed = True
    return assigned_channel

def random_allocate(num_users, num_channels):
    """Random allocation without conflict resolution (will be resolved later)."""
    return np.random.randint(0, num_channels, size=num_users)

def resolve_conflicts(assignment, snr_matrix):
    """
    Post-process an assignment that may have conflicts.
    Keeps the user with highest SNR on each channel, reallocates others greedily.
    Returns conflict-free assignment.
    """
    U, K = snr_matrix.shape
    assignment = assignment.copy()
    # Build mapping channel -> list of users
    channel_to_users = {}
    for u, ch in enumerate(assignment):
        if ch != -1:
            channel_to_users.setdefault(ch, []).append(u)
    # Find conflicts: channels with more than one user
    conflict_channels = [ch for ch, users in channel_to_users.items() if len(users) > 1]
    # For each conflicted channel, keep only the user with highest SNR on that channel
    to_reallocate = []
    for ch in conflict_channels:
        users_on_ch = channel_to_users[ch]
        # Find user with highest SNR on this channel
        snr_on_ch = snr_matrix[users_on_ch, ch]
        best_idx = np.argmax(snr_on_ch)
        keep_user = users_on_ch[best_idx]
        # Others need reallocation
        for u in users_on_ch:
            if u != keep_user:
                to_reallocate.append(u)
        assignment[keep_user] = ch
        # Mark others as unassigned temporarily
        for u in to_reallocate:
            assignment[u] = -1
    # Now reallocate pending users greedily
    # Sort by their best SNR
    to_reallocate = list(set(to_reallocate))
    # Recompute best SNR for each pending user (only on free channels)
    used_channels = set(assignment[assignment != -1])
    free_channels = [c for c in range(K) if c not in used_channels]
    # If not enough free channels, assign best available
    pending_sorted = sorted(to_reallocate, key=lambda u: np.max(snr_matrix[u]), reverse=True)
    for u in pending_sorted:
        # Find best free channel for this user
        if not free_channels:
            # No free channels left; assign best among all (conflict allowed? we avoid)
            # In this case, we can reuse the best channel, but better to break.
            # For simplicity, we assign the channel with highest SNR even if used.
            best_ch = np.argmax(snr_matrix[u])
        else:
            snr_free = snr_matrix[u, free_channels]
            best_ch = free_channels[np.argmax(snr_free)]
            free_channels.remove(best_ch)
        assignment[u] = best_ch
    return assignment

# ==================== 3. Dataset Generation ====================
def generate_dataset(config):
    """
    Generate X (features) and y (greedy labels) for all snapshots.
    Also store additional info needed for evaluation (original SNR matrices).
    """
    num_snapshots = config['num_snapshots']
    # We will fix number of users and channels per snapshot (random within range)
    # But for simplicity and to have consistent feature dimension, we fix them per experiment.
    # Here we choose a fixed number for the whole dataset. You can vary, but then features need padding.
    # According to proposal: U in [10,20], K in [5,10]. We'll fix U=15, K=8 for simplicity.
    # However, to show generalization, you can vary later. For now, fixed.
    U = config['num_users_min']
    K = config['num_channels_min']
    config['fixed_num_users'] = U
    config['fixed_num_channels'] = K
    
    X_list = []      # features: per snapshot, flattened [snr_u1, snr_u2, ..., x1, y1, ...]
    y_list = []      # greedy allocation per snapshot
    snr_matrices = []  # store for later evaluation
    
    for _ in range(num_snapshots):
        x, y_pos = generate_user_positions(U, config['cell_radius_m'])
        snr_mat = compute_snr_matrix(
            x, y_pos,
            config['tx_power_dbm'], config['noise_psd_dbm_per_hz'], config['bandwidth_hz'],
            config['ref_distance_m'], config['path_loss_exponent'],
            config['ref_path_loss_db'], config['carrier_freq_ghz'],
            K
        )
        # Greedy label
        alloc = greedy_allocate(snr_mat)
        # Features: concatenate SNRs (flattened) and user positions
        features = np.concatenate([snr_mat.flatten(), x, y_pos])
        X_list.append(features)
        y_list.append(alloc)
        snr_matrices.append(snr_mat)
    
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, snr_matrices, (U, K)

# ==================== 4. Training and Evaluation ====================
def train_models(X_train, y_train):
    """Train DT, RF, MLP using MultiOutputClassifier."""
    models = {}
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=15, random_state=CONFIG['random_seed'])
    models['DT'] = MultiOutputClassifier(dt, n_jobs=-1)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=CONFIG['random_seed'])
    models['RF'] = MultiOutputClassifier(rf, n_jobs=-1)
    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', 
                        early_stopping=True, validation_fraction=0.1,
                        max_iter=200, random_state=CONFIG['random_seed'])
    models['MLP'] = MultiOutputClassifier(mlp, n_jobs=-1)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_allocations(allocations, snr_matrices, ground_truth_labels=None, method_name=""):
    """
    allocations: list of arrays, each shape (U,) predicted assignment
    snr_matrices: list of matrices (U,K) original SNR
    ground_truth_labels: optional list of greedy allocations for imitation accuracy
    Returns dict of metrics.
    """
    num_snapshots = len(snr_matrices)
    U, K = snr_matrices[0].shape
    throughputs = []
    fairness_indices = []
    outage_rates = []
    imitation_accuracies = [] if ground_truth_labels is not None else None

    snr_threshold_linear = 10**(CONFIG['snr_threshold_db']/10)

    for i in range(num_snapshots):
        snr = snr_matrices[i]
        alloc = allocations[i]

        # 计算每个用户的吞吐量和中断状态
        throughput_user = np.zeros(U)
        outage_user = np.zeros(U, dtype=bool)

        for u in range(U):
            if alloc[u] == -1:                     # 未分配用户
                throughput_user[u] = 0.0
                outage_user[u] = True              # 视为中断
            else:
                snr_val = snr[u, alloc[u]]
                throughput_user[u] = np.log2(1 + snr_val)   # bps/Hz
                outage_user[u] = (snr_val < snr_threshold_linear)

        total_throughput = np.sum(throughput_user)
        throughputs.append(total_throughput)

        # Jain公平性指数（未分配用户贡献0）
        sum_t = total_throughput
        sum_t2 = np.sum(throughput_user**2)
        fairness = (sum_t**2) / (U * sum_t2 + 1e-9)
        fairness_indices.append(fairness)

        outage_rate = np.mean(outage_user)
        outage_rates.append(outage_rate)

        # 模仿准确率（若提供真实标签）
        if ground_truth_labels is not None:
            gt = ground_truth_labels[i]
            acc = np.mean(alloc == gt)
            imitation_accuracies.append(acc)

    metrics = {
        'avg_throughput': np.mean(throughputs),
        'avg_fairness': np.mean(fairness_indices),
        'avg_outage': np.mean(outage_rates),
    }
    if imitation_accuracies is not None:
        metrics['imitation_accuracy'] = np.mean(imitation_accuracies)
    return metrics

def main():
    # 定义需要依次运行的实验配置
    configs = [
        {**CONFIG, 'num_users_min': 8,  'num_users_max': 8,  'num_channels_min': 8,  'num_channels_max': 8},
        {**CONFIG, 'num_users_min': 10, 'num_users_max': 10, 'num_channels_min': 8,  'num_channels_max': 8},
        {**CONFIG, 'num_users_min': 16, 'num_users_max': 16, 'num_channels_min': 8,  'num_channels_max': 8},
        {**CONFIG, 'num_users_min': 12, 'num_users_max': 12, 'num_channels_min': 4,  'num_channels_max': 4},
        {**CONFIG, 'num_users_min': 12, 'num_users_max': 12, 'num_channels_min': 12, 'num_channels_max': 12},
    ]
    
    results_dfs = []
    
    for conf in configs:
        print(f"Running experiment with U={conf['num_users_min']}, K={conf['num_channels_min']}")
        
        print("Generating dataset...")
        X, y_greedy, snr_matrices, (U, K) = generate_dataset(conf)
        print(f"Dataset shape: X={X.shape}, y={y_greedy.shape}, U={U}, K={K}")
        
        # Split data
        X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
            X, y_greedy, snr_matrices, test_size=conf['test_ratio'], random_state=conf['random_seed']
        )
        
        # Train supervised models
        print("Training models...")
        models = train_models(X_train, y_train)
        
        # Baselines: Random and Greedy (on test set)
        print("Evaluating baselines...")
        greedy_allocations = y_test
        random_raw = [random_allocate(U, K) for _ in range(len(X_test))]
        random_resolved = [resolve_conflicts(alloc, snr_test[i]) for i, alloc in enumerate(random_raw)]
        
        greedy_metrics = evaluate_allocations(greedy_allocations, snr_test, method_name="Greedy")
        greedy_metrics['imitation_accuracy'] = 1.0
        
        random_metrics = evaluate_allocations(random_resolved, snr_test, ground_truth_labels=y_test)
        
        # Evaluate supervised models
        results = {}
        for name, model in models.items():
            print(f"Evaluating {name}...")
            start_time = time.perf_counter()
            pred_raw = model.predict(X_test)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) / len(X_test) * 1000
            pred_resolved = []
            for i in range(len(pred_raw)):
                resolved = resolve_conflicts(pred_raw[i], snr_test[i])
                pred_resolved.append(resolved)
            metrics = evaluate_allocations(pred_resolved, snr_test, ground_truth_labels=y_test)
            metrics['inference_time_ms'] = inference_time_ms
            results[name] = metrics
        
        # Measure inference times for Greedy and Random
        print("Measuring Greedy inference time...")
        start_time = time.perf_counter()
        for i, snr in enumerate(snr_test):
            _ = greedy_allocate(snr)
        end_time = time.perf_counter()
        greedy_metrics['inference_time_ms'] = (end_time - start_time) / len(snr_test) * 1000
        
        start_time = time.perf_counter()
        for i, snr in enumerate(snr_test):
            rand_alloc = random_allocate(U, K)
            _ = resolve_conflicts(rand_alloc, snr)
        end_time = time.perf_counter()
        random_metrics['inference_time_ms'] = (end_time - start_time) / len(snr_test) * 1000
        
        # Print summary (optional)
        print("\n" + "="*70)
        print(f"Experiment Configuration: U={U}, K={K}, snapshots={conf['num_snapshots']}")
        print("="*70)
        print(f"{'Method':<10} {'Imitation Acc':<14} {'Throughput':<12} {'Fairness':<10} {'Outage':<10} {'Inference (ms)':<14}")
        print("-"*70)
        print(f"{'Greedy':<10} {'1.0000':<14} {greedy_metrics['avg_throughput']:<12.4f} {greedy_metrics['avg_fairness']:<10.4f} {greedy_metrics['avg_outage']:<10.4f} {greedy_metrics['inference_time_ms']:<14.4f}")
        print(f"{'Random':<10} {random_metrics['imitation_accuracy']:<14.4f} {random_metrics['avg_throughput']:<12.4f} {random_metrics['avg_fairness']:<10.4f} {random_metrics['avg_outage']:<10.4f} {random_metrics['inference_time_ms']:<14.4f}")
        for name, metrics in results.items():
            print(f"{name:<10} {metrics['imitation_accuracy']:<14.4f} {metrics['avg_throughput']:<12.4f} {metrics['avg_fairness']:<10.4f} {metrics['avg_outage']:<10.4f} {metrics['inference_time_ms']:<14.4f}")
        print("="*70)
        
        # 收集结果到 DataFrame
        data = []
        for name, metrics in results.items():
            data.append([name, U, K, metrics['imitation_accuracy'], metrics['avg_throughput'], metrics['avg_fairness'], metrics['avg_outage'], metrics['inference_time_ms']])
        data.append(['Greedy', U, K, 1.0, greedy_metrics['avg_throughput'], greedy_metrics['avg_fairness'], greedy_metrics['avg_outage'], greedy_metrics['inference_time_ms']])
        data.append(['Random', U, K, random_metrics['imitation_accuracy'], random_metrics['avg_throughput'], random_metrics['avg_fairness'], random_metrics['avg_outage'], random_metrics['inference_time_ms']])
        
        df = pd.DataFrame(data, columns=['Method', 'U', 'K', 'ImitationAcc', 'Throughput', 'Fairness', 'Outage', 'Inference_ms'])
        results_dfs.append(df)
    
    # 写入 Excel 文件，分隔每个实验组
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, 'experiment_results.xlsx')

    def write_excel(path):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            start_row = 0
            for df in results_dfs:
                df.to_excel(writer, sheet_name='Results', startrow=start_row, index=False)
                start_row += len(df) + 2  # 空 2 行分隔

    try:
        write_excel(excel_path)
        print(f"所有实验结果已保存到: {excel_path}")
    except PermissionError:
        fallback_path = os.path.join(os.path.expanduser('~'), 'experiment_results.xlsx')
        try:
            write_excel(fallback_path)
            print(f"文件被占用，改为保存到: {fallback_path}")
        except PermissionError:
            temp_path = os.path.join(tempfile.gettempdir(), 'experiment_results.xlsx')
            write_excel(temp_path)
            print(f"文件被占用，改为临时目录保存到: {temp_path}")
def max_snr_random_resolve(snr_matrix):
    """
    Each user picks its best channel (may conflict).
    Then resolve conflicts randomly.
    """
    U, K = snr_matrix.shape
    # Step 1: each user picks best channel
    best_ch = np.argmax(snr_matrix, axis=1)
    assignment = best_ch.copy()
    # Step 2: conflict resolution
    # We'll do: for each channel, keep one random user, others get random free channels
    used_channels = set()
    # First, decide which users keep their chosen channel (random per channel)
    channel_to_users = {}
    for u, ch in enumerate(assignment):
        channel_to_users.setdefault(ch, []).append(u)
    new_assignment = np.full(U, -1, dtype=int)
    for ch, users in channel_to_users.items():
        if len(users) == 1:
            new_assignment[users[0]] = ch
            used_channels.add(ch)
        else:
            # randomly select one user to keep this channel
            keep = np.random.choice(users)
            new_assignment[keep] = ch
            used_channels.add(ch)
            # others will be reassigned later
            for u in users:
                if u != keep:
                    new_assignment[u] = -1
    # Reassign unassigned users (those with -1)
    unassigned = [u for u in range(U) if new_assignment[u] == -1]
    free_channels = [c for c in range(K) if c not in used_channels]
    # If not enough free channels, we may need to reuse, but assume U <= K for fairness
    # For U > K, we can assign best available free channel (or reuse)
    for u in unassigned:
        if free_channels:
            # choose random free channel
            ch = np.random.choice(free_channels)
            free_channels.remove(ch)
        else:
            # fallback: choose best channel (could cause conflict but we accept)
            ch = np.argmax(snr_matrix[u])
        new_assignment[u] = ch
    return new_assignment
if __name__ == "__main__":
