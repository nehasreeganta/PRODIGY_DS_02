import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preview data
df = pd.read_csv(r"C:\PRODIGY\TASK-2\test.csv")
print(df.head(), "\nColumns:", df.columns.tolist())

# Helper to plot & save
def plot_and_save(func, filename, **kwargs):
    plt.figure(figsize=kwargs.pop("figsize", (6, 4)))
    func(**kwargs)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Plots
plot_and_save(sns.heatmap, "missing_values.png", data=df.isnull(), cbar=False, cmap='viridis', figsize=(8, 5))
plot_and_save(sns.countplot, "sex_distribution.png", x='Sex', data=df, palette='pastel')
plot_and_save(sns.countplot, "pclass_distribution.png", x='Pclass', data=df, palette='muted')
plot_and_save(sns.histplot, "age_distribution.png", data=df, x='Age', kde=True, color='skyblue', bins=30, figsize=(8, 5))
plot_and_save(sns.countplot, "survival_count.png", x='Survived', data=df, palette='coolwarm')

# Correlation heatmap for numeric columns
num_df = df.select_dtypes(include='number')
if not num_df.empty:
    plot_and_save(sns.heatmap, "correlation_heatmap.png", data=num_df.corr(), annot=True, cmap='coolwarm', figsize=(8, 6))
else:
    print("No numeric columns for correlation heatmap.")