"""
SFE Conversation Geometry Pipeline
====================================
Parses raw txt conversation exports, extracts turn-by-turn structure,
computes joint observer state space (user, AI, correlation),
and runs PCA to map the geometric trajectory of each conversation.

Usage:
    python sfe_conversation_geometry.py --folder ./conversations

Output:
    - Per-file geometry report (e1, e2, e3, shape, phase transitions)
    - Cross-file comparison table
    - Two static plots saved as PNG
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# ─── Language detection (minimal, no external deps) ──────────────────────────

SPANISH_MARKERS = [
    'que', 'de', 'en', 'el', 'la', 'los', 'las', 'un', 'una',
    'es', 'son', 'por', 'con', 'para', 'del', 'al', 'se', 'no',
    'una', 'como', 'pero', 'más', 'si', 'ya', 'lo', 'le', 'su',
    'dijo', 'dijiste', 'también', 'esto', 'ese', 'eso', 'hay',
    'porque', 'cuando', 'donde', 'tiene', 'puede', 'hacer'
]

def detect_language(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 'unknown'
    spanish_hits = sum(1 for w in words if w in SPANISH_MARKERS)
    ratio = spanish_hits / len(words)
    if ratio > 0.08:
        return 'es'
    return 'en'

# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_conversation(filepath):
    """
    Parse a txt conversation file into a list of turns.
    Each turn: {'role': 'user'|'ai', 'text': str, 'lang': str}

    Handles:
    - Missing 'Dijiste:' on first user message
    - Mixed language
    - Empty lines between turns
    - Both 'ChatGPT dijo:' and 'ChatGPT said:' markers
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Normalize line endings
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')

    # Split on role markers
    # Patterns: "ChatGPT dijo:", "ChatGPT said:", "Dijiste:", "You said:"
    ai_pattern   = r'(?:ChatGPT\s+dijo:|ChatGPT\s+said:|Assistant:|Claude:)'
    user_pattern = r'(?:Dijiste:|You\s+said:|Usuario:)'

    # Split into segments with their markers
    splitter = re.compile(
        r'(' + ai_pattern + r'|' + user_pattern + r')',
        re.IGNORECASE
    )

    parts = splitter.split(raw)

    turns = []

    # First segment before any marker = first user message (no "Dijiste:" prefix)
    if parts and parts[0].strip():
        text = parts[0].strip()
        if text:
            turns.append({
                'role': 'user',
                'text': text,
                'lang': detect_language(text)
            })

    # Process remaining segments in pairs: (marker, content)
    i = 1
    while i < len(parts) - 1:
        marker  = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ''

        if re.match(ai_pattern, marker, re.IGNORECASE):
            role = 'ai'
        else:
            role = 'user'

        if content:
            turns.append({
                'role': role,
                'text': content,
                'lang': detect_language(content)
            })
        i += 2

    return turns

# ─── Feature extraction ───────────────────────────────────────────────────────

def message_features(text):
    """
    Extract numerical features from a single message.
    Returns a feature vector.
    """
    words   = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    length          = len(text)
    word_count      = len(words)
    sentence_count  = max(len(sentences), 1)
    avg_word_len    = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len    = word_count / sentence_count
    question_count  = text.count('?')
    has_code        = 1.0 if ('```' in text or 'def ' in text or 'import ' in text) else 0.0
    has_numbers     = 1.0 if re.search(r'\d+\.\d+|\d{3,}', text) else 0.0
    caps_ratio      = sum(1 for c in text if c.isupper()) / max(length, 1)
    lang_es         = 1.0 if detect_language(text) == 'es' else 0.0

    return np.array([
        length,
        word_count,
        avg_word_len,
        avg_sent_len,
        question_count,
        has_code,
        has_numbers,
        caps_ratio,
        lang_es
    ], dtype=float)

def windowed_correlation(r_a, r_b, W=5):
    """
    Windowed Pearson correlation between two feature series.
    Same logic as SFE — W cycles, return correlation at each point.
    """
    n = len(r_a)
    corr = np.zeros(n)
    for i in range(n):
        start = max(0, i - W + 1)
        a_win = r_a[start:i+1]
        b_win = r_b[start:i+1]
        if len(a_win) < 2:
            corr[i] = 0.0
            continue
        if np.std(a_win) < 1e-9 or np.std(b_win) < 1e-9:
            corr[i] = 0.0
            continue
        c = np.corrcoef(a_win, b_win)[0, 1]
        corr[i] = 0.0 if np.isnan(c) else c
    return corr

def build_joint_state(turns, W=5):
    """
    Build the joint observer state space from conversation turns.

    Observer A = user messages
    Observer B = AI messages

    At each exchange (cycle), we record:
      - scalar feature of user message (norm of feature vector)
      - scalar feature of AI message
      - windowed cross-correlation of their feature norms

    Returns: array of shape (n_cycles, 3)
    """
    # Pair up turns into exchange cycles
    cycles = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]['role'] == 'user' and turns[i+1]['role'] == 'ai':
            user_feat = message_features(turns[i]['text'])
            ai_feat   = message_features(turns[i+1]['text'])
            cycles.append((user_feat, ai_feat))
            i += 2
        else:
            i += 1

    if len(cycles) < 3:
        return None, cycles

    # Scalar representation: L2 norm of feature vectors
    r_user = np.array([np.linalg.norm(c[0]) for c in cycles])
    r_ai   = np.array([np.linalg.norm(c[1]) for c in cycles])

    # Normalize to [0,1] range for correlation stability
    def safe_norm(x):
        rng = x.max() - x.min()
        if rng < 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    r_user_n = safe_norm(r_user)
    r_ai_n   = safe_norm(r_ai)

    rho = windowed_correlation(r_user_n, r_ai_n, W=W)

    joint = np.column_stack([r_user_n, r_ai_n, rho])
    return joint, cycles

# ─── Geometry classification ──────────────────────────────────────────────────

def classify_shape(e1):
    if e1 < 0.50:
        return 'VOLUMETRIC'
    elif e1 < 0.70:
        return 'PLANAR'
    else:
        return 'LINEAR'

def run_pca(joint):
    """
    Run PCA on joint state cloud.
    Returns eigenvalues, components, shape label.
    """
    if len(joint) < 4:
        return None

    scaler = StandardScaler()
    normed = scaler.fit_transform(joint)

    pca = PCA(n_components=min(3, joint.shape[1]))
    pca.fit(normed)

    ev = pca.explained_variance_ratio_
    # Pad to 3 if needed
    while len(ev) < 3:
        ev = np.append(ev, 0.0)

    components = pca.components_
    w_rho = abs(components[0][2]) if components.shape[1] > 2 else 0.0

    return {
        'e1': ev[0],
        'e2': ev[1],
        'e3': ev[2],
        'w_rho': w_rho,
        'e2_e3_ratio': ev[1] / max(ev[2], 1e-9),
        'shape': classify_shape(ev[0])
    }

# ─── Phase transition detection ───────────────────────────────────────────────

def detect_transitions(joint, window=5):
    """
    Slide a window through the conversation and compute geometry at each point.
    Returns trajectory of (position, e1, shape) to show how geometry evolves.
    """
    n = len(joint)
    trajectory = []

    for i in range(window, n + 1):
        chunk = joint[i-window:i]
        result = run_pca(chunk)
        if result:
            trajectory.append({
                'position': i,
                'e1': result['e1'],
                'shape': result['shape'],
                'w_rho': result['w_rho']
            })

    return trajectory

# ─── Per-file analysis ────────────────────────────────────────────────────────

def analyze_file(filepath, W=5, window=8, verbose=True):
    filename = os.path.basename(filepath)

    turns = parse_conversation(filepath)
    if not turns:
        return None

    joint, cycles = build_joint_state(turns, W=W)
    if joint is None or len(joint) < 4:
        if verbose:
            print(f"  {filename}: too short ({len(turns)} turns), skipping")
        return None

    # Global geometry
    global_pca = run_pca(joint)

    # Language distribution
    user_turns = [t for t in turns if t['role'] == 'user']
    ai_turns   = [t for t in turns if t['role'] == 'ai']
    es_ratio   = sum(1 for t in turns if t['lang'] == 'es') / max(len(turns), 1)

    # Phase trajectory
    trajectory = detect_transitions(joint, window=window)

    # Find transitions in trajectory
    transitions = []
    if trajectory:
        prev_shape = trajectory[0]['shape']
        for point in trajectory[1:]:
            if point['shape'] != prev_shape:
                transitions.append({
                    'from': prev_shape,
                    'to': point['shape'],
                    'at_cycle': point['position']
                })
                prev_shape = point['shape']

    result = {
        'filename': filename,
        'n_turns': len(turns),
        'n_cycles': len(cycles),
        'n_user': len(user_turns),
        'n_ai': len(ai_turns),
        'es_ratio': es_ratio,
        'global': global_pca,
        'trajectory': trajectory,
        'transitions': transitions,
        'joint': joint
    }

    if verbose:
        g = global_pca
        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"  Turns: {len(turns)}  Cycles: {len(cycles)}  "
              f"Spanish: {es_ratio:.0%}")
        print(f"  Global geometry:")
        print(f"    e1={g['e1']:.3f}  e2={g['e2']:.3f}  e3={g['e3']:.3f}")
        print(f"    |w_rho|={g['w_rho']:.3f}  e2/e3={g['e2_e3_ratio']:.2f}")
        print(f"    Shape: {g['shape']}")
        if transitions:
            print(f"  Transitions:")
            for t in transitions:
                print(f"    {t['from']} → {t['to']} at cycle {t['at_cycle']}")
        else:
            print(f"  No shape transitions detected")

    return result

# ─── Cross-file comparison plot ───────────────────────────────────────────────

def plot_comparison(results, output_path='sfe_conversation_geometry.png'):
    valid = [r for r in results if r and r['global']]
    if not valid:
        print("No valid results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                              facecolor='#07080f')

    for ax in axes.flat:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#aaa')
        ax.xaxis.label.set_color('#aaa')
        ax.yaxis.label.set_color('#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    names     = [r['filename'].replace('.txt', '')[:20] for r in valid]
    e1_vals   = [r['global']['e1'] for r in valid]
    e2_vals   = [r['global']['e2'] for r in valid]
    e3_vals   = [r['global']['e3'] for r in valid]
    wrho_vals = [r['global']['w_rho'] for r in valid]
    ratio_vals= [r['global']['e2_e3_ratio'] for r in valid]
    es_vals   = [r['es_ratio'] for r in valid]
    colors_shape = ['#ff6b35' if r['global']['shape'] == 'LINEAR'
                    else '#00b4d8' if r['global']['shape'] == 'PLANAR'
                    else '#90e0ef' for r in valid]

    x = np.arange(len(valid))

    # Plot 1: e1 per file (shape indicator)
    ax = axes[0, 0]
    bars = ax.bar(x, e1_vals, color=colors_shape, alpha=0.85, width=0.6)
    ax.axhline(0.70, color='red',    linestyle='--', alpha=0.6, label='LINEAR (0.70)')
    ax.axhline(0.50, color='yellow', linestyle='--', alpha=0.6, label='PLANAR (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('e1 (dominant eigenvalue)', color='#aaa')
    ax.set_title('Dominant Eigenvalue per Conversation', color='white')
    ax.legend(fontsize=7, facecolor='#0a0a0a', labelcolor='white')
    ax.set_ylim(0, 1.05)

    # Plot 2: |w_rho| per file
    ax = axes[0, 1]
    ax.bar(x, wrho_vals, color='#ffd60a', alpha=0.85, width=0.6)
    ax.axhline(0.90, color='white', linestyle='--', alpha=0.5, label='dominance (0.90)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('|ρ weight in PC1|', color='#aaa')
    ax.set_title('Correlation Axis Weight per Conversation', color='white')
    ax.legend(fontsize=7, facecolor='#0a0a0a', labelcolor='white')
    ax.set_ylim(0, 1.05)

    # Plot 3: e2/e3 ratio (asymmetry)
    ax = axes[1, 0]
    ax.bar(x, ratio_vals, color='#c77dff', alpha=0.85, width=0.6)
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.6, label='asymmetry (2.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('e2/e3 ratio', color='#aaa')
    ax.set_title('Secondary Eigenvalue Asymmetry', color='white')
    ax.legend(fontsize=7, facecolor='#0a0a0a', labelcolor='white')

    # Plot 4: Spanish ratio vs e1 scatter
    ax = axes[1, 1]
    sc = ax.scatter(es_vals, e1_vals, c=wrho_vals, cmap='plasma',
                    s=80, alpha=0.85, vmin=0, vmax=1)
    ax.axhline(0.70, color='red',    linestyle='--', alpha=0.4)
    ax.axhline(0.50, color='yellow', linestyle='--', alpha=0.4)
    ax.set_xlabel('Spanish ratio', color='#aaa')
    ax.set_ylabel('e1 (dominant eigenvalue)', color='#aaa')
    ax.set_title('Language Mix vs Geometry\n(color = |w_rho|)', color='white')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(colors='#aaa')
    cbar.set_label('|w_rho|', color='#aaa')

    fig.suptitle('SFE · Conversation Geometry Map', color='white',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#07080f')
    plt.close()
    print(f"\nSaved: {output_path}")

def plot_trajectory(result, output_path=None):
    """Plot the geometric trajectory of a single conversation."""
    if not result or not result['trajectory']:
        return

    traj = result['trajectory']
    positions = [t['position'] for t in traj]
    e1_vals   = [t['e1'] for t in traj]
    wrho_vals = [t['w_rho'] for t in traj]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#07080f')
    ax.set_facecolor('#0a0a0a')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    ax.plot(positions, e1_vals,   color='#ff6b35', linewidth=2,
            label='e1 (dominant)', marker='o', markersize=4)
    ax.plot(positions, wrho_vals, color='#ffd60a', linewidth=2,
            linestyle='--', label='|w_rho|', marker='o', markersize=4)

    ax.axhline(0.70, color='red',    linestyle=':', alpha=0.5, label='LINEAR')
    ax.axhline(0.50, color='yellow', linestyle=':', alpha=0.5, label='PLANAR')

    # Mark transitions
    for t in result['transitions']:
        ax.axvline(t['at_cycle'], color='white', alpha=0.3, linestyle='--')
        ax.text(t['at_cycle'], 0.95,
                f"{t['from'][:3]}→{t['to'][:3]}",
                color='white', fontsize=7, ha='center', va='top')

    ax.set_xlabel('Conversation cycle', color='#aaa')
    ax.set_ylabel('Value', color='#aaa')
    ax.set_title(f"Geometric Trajectory — {result['filename']}",
                 color='white')
    ax.legend(fontsize=8, facecolor='#0a0a0a', labelcolor='white')
    ax.set_ylim(-0.1, 1.1)

    if output_path is None:
        output_path = result['filename'].replace('.txt', '_trajectory.png')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#07080f')
    plt.close()
    print(f"Saved: {output_path}")

# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(results):
    valid = [r for r in results if r and r['global']]
    if not valid:
        print("No valid results.")
        return

    print(f"\n{'='*80}")
    print("SFE · Conversation Geometry Summary")
    print(f"{'='*80}")
    print(f"{'File':<25} {'Cycles':>6} {'e1':>6} {'e2':>6} {'e3':>6} "
          f"{'|wρ|':>6} {'e2/e3':>6} {'ES%':>5} {'Shape'}")
    print(f"{'-'*80}")

    for r in sorted(valid, key=lambda x: x['global']['e1'], reverse=True):
        g = r['global']
        print(f"{r['filename'][:24]:<25} {r['n_cycles']:>6} "
              f"{g['e1']:>6.3f} {g['e2']:>6.3f} {g['e3']:>6.3f} "
              f"{g['w_rho']:>6.3f} {g['e2_e3_ratio']:>6.2f} "
              f"{r['es_ratio']:>4.0%} {g['shape']}")

    print(f"{'-'*80}")
    shapes = [r['global']['shape'] for r in valid]
    print(f"\nShape distribution:")
    for s in ['VOLUMETRIC', 'PLANAR', 'LINEAR']:
        count = shapes.count(s)
        print(f"  {s}: {count} ({count/len(valid):.0%})")
    print(f"{'='*80}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SFE Conversation Geometry Pipeline'
    )
    parser.add_argument('--folder', default='.',
                        help='Folder containing txt conversation files')
    parser.add_argument('--W',      type=int, default=5,
                        help='Correlation window size (default: 5)')
    parser.add_argument('--window', type=int, default=8,
                        help='PCA sliding window for trajectory (default: 8)')
    parser.add_argument('--output', default='sfe_conversation_geometry.png',
                        help='Output filename for comparison plot')
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    txt_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith('.txt')
    ])

    if not txt_files:
        print(f"No .txt files found in {folder}")
        sys.exit(1)

    print(f"Found {len(txt_files)} txt files in {folder}")

    results = []
    for filepath in txt_files:
        result = analyze_file(filepath, W=args.W, window=args.window)
        results.append(result)

    print_summary(results)
    plot_comparison(results, output_path=args.output)

    # Trajectory plot for the most interesting file
    # (highest e1 = most collapsed = most convergent conversation)
    valid = [r for r in results if r and r['global']]
    if valid:
        most_linear = max(valid, key=lambda r: r['global']['e1'])
        traj_path = most_linear['filename'].replace('.txt', '_trajectory.png')
        plot_trajectory(most_linear, output_path=traj_path)
        print(f"\nTrajectory plot for most convergent conversation: "
              f"{most_linear['filename']}")

if __name__ == '__main__':
    main()
