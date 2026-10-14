"""eda.py - creates the EDA plots and saves them in images/. Run: python eda.py"""
import os
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_dataset, clean_features, TARGET, ID_COLUMNS

os.makedirs("images", exist_ok=True)
sns.set_theme(style="whitegrid")
ORDER = ["None", "Insomnia", "Sleep Apnea"]
PALETTE = {"None": "#4C9F70", "Insomnia": "#E9A23B", "Sleep Apnea": "#D1495B"}


def save(name):
    plt.tight_layout()
    plt.savefig(f"images/{name}.png", dpi=150)
    plt.close()


def make_eda_plots(path="data/dataset.csv"):
    df = load_dataset(path).drop(columns=ID_COLUMNS)
    df = df.drop_duplicates()            # same de-duplicated data used for training
    df = clean_features(df)  # splits blood pressure, merges BMI labels

    # 1-7: single-variable distributions
    sns.countplot(data=df, x=TARGET, order=ORDER, hue=TARGET, palette=PALETTE, legend=False)
    plt.title("Sleep Disorder Class Distribution"); save("01_class_distribution")
    for i, (col, title) in enumerate([
        ("Age", "Age Distribution"), ("Sleep Duration", "Sleep Duration Distribution"),
        ("Quality of Sleep", "Quality of Sleep Distribution"), ("Stress Level", "Stress Level Distribution"),
        ("Physical Activity Level", "Physical Activity Level Distribution")], start=2):
        sns.histplot(df[col], bins=15, kde=True, color="#3B6EA8")
        plt.title(title); save(f"{i:02d}_{col.lower().replace(' ', '_')}_distribution")
    sns.countplot(data=df, x="BMI Category", color="#3B6EA8")
    plt.title("BMI Category Distribution"); save("07_bmi_distribution")

    # 8-10: numeric feature vs target
    for i, col in enumerate(["Sleep Duration", "Stress Level", "Quality of Sleep"], start=8):
        sns.boxplot(data=df, x=TARGET, y=col, order=ORDER, hue=TARGET, palette=PALETTE, legend=False)
        plt.title(f"{col} vs Sleep Disorder"); save(f"{i:02d}_{col.lower().replace(' ', '_')}_vs_disorder")
    # 11: BMI vs target
    sns.countplot(data=df, x="BMI Category", hue=TARGET, hue_order=ORDER, palette=PALETTE)
    plt.title("BMI Category vs Sleep Disorder"); save("11_bmi_vs_disorder")
    # 12: correlation heatmap
    num = df.select_dtypes(include="number")
    plt.figure(figsize=(9, 7))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numerical Features)"); save("12_correlation_heatmap")
    print("EDA plots saved in images/")


if __name__ == "__main__":
    make_eda_plots()
