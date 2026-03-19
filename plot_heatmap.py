import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

OUTPUT_DIR = "results/heatmap"

def save_heatmap(pivot, title, cbar_label, color, vmin, vmax,filename):
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=color, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Num Exemples (K)")
    plt.ylabel("Data Base")
    plt.tight_layout()

    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path)
    plt.show()
def heatmap(csv_file):
    df = pd.read_csv(csv_file)

    #mitjana de les 3s repeticions. 
    avg = df.groupby(['Database', 'K']).agg(
        Accuracy=('Accuracy', 'mean'),
        Avg_Input_Tokens=('Avg_Input_Tokens', 'mean')
    ).reset_index()

    #calcul eficiencia
    avg['Efficiency'] = avg['Accuracy'] / avg['Avg_Input_Tokens'] *1000


    pivot = avg.pivot(index='Database', columns='K', values='Accuracy')
    pivot_efficiency = avg.pivot(index='Database', columns='K', values='Efficiency')

    #dibuixar heatmap
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_heatmap(pivot,"Few-Shot Accuracy %","Accuracy (%)", "YlGnBu", 0, 100, f"heatmap_accuracy_{now}.png")
    save_heatmap(pivot_efficiency, "Few-Shot Eficiencia / 1K tokens","Accuracy / 1000 tokens", "YlOrRd", None, None, f"heatmap_efficiency_{now}.png" )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    heatmap("results/csv/results_experiment_20260228_172918.csv")