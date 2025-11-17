"""
utils.py
Enhanced logging, visualization, and analysis utilities.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def analyze_bandit_log(log_path="logs/bandit_history.jsonl", patch_space=None):
    """Load and visualize bandit decision history."""
    if not Path(log_path).exists():
        print(f"⚠️ Log file not found: {log_path}")
        return None
    
    records = []
    with open(log_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    
    if not records:
        print("⚠️ No records found in log")
        return None
    
    df = pd.json_normalize(records)
    
    # Add human-readable arm names
    if patch_space:
        df["arm_name"] = df["arm"].apply(
            lambda x: f"Arm{x}: {patch_space[x]['retriever_type'][:3]}_k{patch_space[x]['k']}"
        )
    else:
        df["arm_name"] = df["arm"].astype(str)
    
    # Plot 1: Arm selection frequency
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="arm_name", order=df["arm_name"].value_counts().index)
    plt.xticks(rotation=45, ha="right")
    plt.title("Bandit Arm Selection Frequency")
    plt.xlabel("Patch Configuration")
    plt.ylabel("Selection Count")
    plt.tight_layout()
    plt.savefig("logs/bandit_arm_selection.png", dpi=150)
    print("✅ Saved: logs/bandit_arm_selection.png")
    
    # Plot 2: Reward vs Cost scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="cost", y="reward", hue="arm_name", palette="Set2", s=80, alpha=0.7)
    plt.title("Reward vs Cost per Patch")
    plt.xlabel("Normalized Cost")
    plt.ylabel("Reward (0-1)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig("logs/reward_vs_cost.png", dpi=150)
    print("✅ Saved: logs/reward_vs_cost.png")
    
    # Summary statistics
    summary = df.groupby("arm_name").agg({
        "reward": ["mean", "std", "count"],
        "cost": ["mean", "std"]
    }).round(3)
    
    print("📊 Bandit Performance Summary:")
    print(summary)
    
    return df


def plot_cost_breakdown(log_path="logs/cost_breakdown.jsonl"):
    """Visualize per-component cost breakdown."""
    if not Path(log_path).exists():
        print(f"⚠️ Cost log not found: {log_path}")
        return
    
    records = []
    with open(log_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    
    if not records:
        print("⚠️ No cost records found")
        return
    
    # Aggregate by component
    component_totals = {}
    for record in records:
        for component, time_spent in record["breakdown"].items():
            component_totals[component] = component_totals.get(component, 0.0) + time_spent
    
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        component_totals.values(),
        labels=component_totals.keys(),
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Total Time Spent per Component")
    plt.tight_layout()
    plt.savefig("logs/cost_breakdown_pie.png", dpi=150)
    print("✅ Saved: logs/cost_breakdown_pie.png")
    
    # Plot stacked bar chart (per-sample breakdown)
    df_breakdown = pd.DataFrame([
        {**{"sample_idx": r["sample_idx"]}, **r["breakdown"]}
        for r in records
    ])
    
    df_breakdown.set_index("sample_idx").plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="tab20"
    )
    plt.title("Per-Sample Cost Breakdown (Stacked)")
    plt.xlabel("Sample Index")
    plt.ylabel("Time (seconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("logs/cost_breakdown_stacked.png", dpi=150)
    print("✅ Saved: logs/cost_breakdown_stacked.png")


def save_trial_results(results, output_path="logs/trial_results.jsonl"):
    """Save trial results to JSONL."""
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"✅ Trial results saved to {output_path}")
