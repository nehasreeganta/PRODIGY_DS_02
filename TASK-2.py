import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load CSV (change path to train.csv if you want survival column)
df = pd.read_csv(r"C:\MINI PROJECT\PRODIGY\TASK2\test.csv")
print(df.head(), "\nColumns:", df.columns.tolist())

# Helper function to plot & save
def plot_and_save(func, filename, **kwargs):
    plt.figure(figsize=kwargs.pop("figsize", (6, 4)))
    func(**kwargs)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

# Create output folder
os.makedirs("plots", exist_ok=True)

# 1. Missing values heatmap
plot_and_save(
    sns.heatmap,
    "plots/missing_values.png",
    data=df.isnull().astype(int),
    cbar=False,
    cmap="viridis",
    figsize=(8, 5)
)

# 2. Sex distribution
if "Sex" in df.columns:
    plot_and_save(
        sns.countplot,
        "plots/sex_distribution.png",
        x="Sex",
        hue="Sex",
        data=df,
        palette="pastel",
        legend=False
    )

# 3. Pclass distribution
if "Pclass" in df.columns:
    plot_and_save(
        sns.countplot,
        "plots/pclass_distribution.png",
        x="Pclass",
        hue="Pclass",
        data=df,
        palette="muted",
        legend=False
    )

# 4. Age distribution
if "Age" in df.columns:
    plot_and_save(
        sns.histplot,
        "plots/age_distribution.png",
        data=df,
        x="Age",
        kde=True,
        color="skyblue",
        bins=30,
        figsize=(8, 5)
    )

# 5. Survival count (only if 'Survived' exists)
if "Survived" in df.columns:
    plot_and_save(
        sns.countplot,
        "plots/survival_count.png",
        x="Survived",
        hue="Survived",
        data=df,
        palette="coolwarm",
        legend=False
    )
else:
    print("'Survived' column not found — skipping survival plot.")

# 6. Correlation heatmap (numeric columns only)
num_df = df.select_dtypes(include="number")
if not num_df.empty:
    plot_and_save(
        sns.heatmap,
        "plots/correlation_heatmap.png",
        data=num_df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        figsize=(8, 6)
    )
else:
    print("No numeric columns for correlation heatmap.")
