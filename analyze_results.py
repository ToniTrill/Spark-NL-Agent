import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json 
from pathlib import Path
from datetime import datetime


#Calcula precisió, recall i F1 comparant un conjunt d'UDFs gold (correctes) vs predits (predicció del model)
def calculate_udf_metircs(gold, pred):
    gold_set = set(gold)
    pred_set = set(pred)

    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

#llegir json i per cada pregunta calcular les metriques i guardar-les en un dataframe, després resumir per dificultat i mostrar quadrivula
def analyze_json(file_path, output_dir='results_analysis', manual_desc="Aquest gràfic mostra els resultats de les preguntes del benchmark UDFBench, agrupats per dificultat. La columna 'Functional' indica la precisió funcional global (verd per correcte, vermell per incorrecte), mentre que 'UDF_F1' mostra la mitjana de la puntuació F1 de les UDFs associades a cada pregunta. Les columnes 'In_Tokens' i 'Out_Tokens' representen la mitjana de tokens d'entrada i sortida respectivament. Aquest anàlisi permet identificar tendències en el rendiment del model segons la dificultat de les preguntes i la complexitat de les UDFs involucrades."):
    
    #crear capreta de soritda si no existeix
    output_path = Path(output_dir)
    output_path.mkdir( exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir= output_path / f"analysis_{timestamp}"
    session_dir.mkdir(exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []

    for item in data:
        p, r, f1 = calculate_udf_metircs(item['gold_udfs'], item['predicted_udfs'])

        results.append({
            "ID": item["question_id"],
            "Difficulty": item["difficulty"],
            "Query_Accuracy": item['is_functional_correct'],
            "UDF_Precision": p,
            "UDF_Recall": r,
            "UDF_F1": f1,
            "In_Tokens": item['tokens']['input'],
            "Out_Tokens": item['tokens']['output']
        })

    df = pd.DataFrame(results)

    summary = df.groupby('Difficulty').agg({
        'Query_Accuracy': 'mean',
        'UDF_F1': 'mean',
        'In_Tokens': 'mean',
        'Out_Tokens': 'mean'
    }).reset_index()

    summary_styled = summary.copy()
    summary_styled['Query_Accuracy'] = (summary_styled['Query_Accuracy'] * 100).round(2).astype(str) + '%'
    summary_styled['UDF_F1'] = summary_styled['UDF_F1'].round(3)

    print("Resum per dificultat")
    print(summary_styled)

    #guardar csv
    summary_styled.to_csv(session_dir / 'summary_by_difficulty.csv', index=False)

    #Precisio golobal
    global_accuracy = (df['Query_Accuracy'].mean() * 100)
    footer_text = f"Precisió global: {global_accuracy:.2f}% | Data analitzada: {timestamp}"
    print(f"\n{footer_text}")
    display_tables_interactive(df, summary_styled, global_accuracy, session_dir)
    #guardar grafic

    image_name = f"heatmap_{timestamp}.png"
    desc_for_plot = f"{manual_desc}\n{footer_text}"
    plot_grid(df, session_dir / image_name, desc_for_plot)

    df.to_csv(session_dir / "detailed_results.csv", index=False)

    return df
    
#  Crea un mapa de calor (heatmap) mostrant l'estat de cada pregunta (correcte/incorrecte) per dificultat i ID de pregunta
def plot_grid(df, save_path, description="la meva descicion manual per el grafic"):
    plt.figure(figsize=(14, 7))
    
    # Convertim Query_Accuracy a int per al mapa de colors
    df_plot = df.copy()
    df_plot['Query_Accuracy'] = df_plot['Query_Accuracy'].astype(int)

    # Pivotar dades
    grid_data = df_plot.pivot(index="Difficulty", columns="ID", values="Query_Accuracy")

    # Crear heatmap: Verd (1) Correcte, Vermell (0) Incorrecte
    sns.heatmap(grid_data, annot=True, cmap="RdYlGn", cbar=False, 
                linewidths=1, linecolor='white', fmt=".0f", annot_kws={"size": 12})

    plt.title('Resultats per Pregunta i Dificultat', fontsize=16, pad=20)
    plt.xlabel('ID de la Pregunta (UDFBench)', fontsize=12)
    plt.ylabel('Dificultat', fontsize=12)
    
    # Afegir descripció manual a la part inferior de la imatge
    if description:
        plt.figtext(0.5, 0.01, description, wrap=True, horizontalalignment='center', 
                    fontsize=10, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"Gràfic guardat a: {save_path}")
    #plt.show()


def display_tables_interactive(df, summary_styled, global_accuracy, session_dir):
    """Mostra taules interactives al terminal i guarda HTML"""
    
    print("\n" + "="*80)
    print("📊 RESULTATS GLOBALS".center(80))
    print("="*80)
    print(f"Precisió Global: {global_accuracy:.2f}%")
    print(f"Total preguntes: {len(df)}")
    print(f"✅ Correctes: {df['Query_Accuracy'].sum()}")
    print(f"❌ Incorrectes: {len(df) - df['Query_Accuracy'].sum()}")
    
    print("\n" + "="*80)
    print("📈 RESUME PER DIFICULTAT".center(80))
    print("="*80)
    print(summary_styled.to_string(index=False))
    
    # Guardar HTML simple (sense gradient)
    html_path = session_dir / 'taula_resultats.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""<html>
<head><title>Resultats UDFBench</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background-color: #4472C4; color: white; }}
    tr:nth-child(even) {{ background-color: #f2f2f2; }}
    tr:hover {{ background-color: #f5f5f5; }}
    .correcte {{ color: green; font-weight: bold; }}
    .incorrecte {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
    <h1>📊 Resultats UDFBench</h1>
    <p><strong>Precisió Global:</strong> {global_accuracy:.2f}%</p>
    <p><strong>Total preguntes:</strong> {len(df)} | 
    <span style='color:green'>✅ Correctes: {df['Query_Accuracy'].sum()}</span> | 
    <span style='color:red'>❌ Incorrectes: {len(df) - df['Query_Accuracy'].sum()}</span></p>
    {summary_styled.to_html(index=False)}
</body>
</html>""")
    
    print(f"\n✅ Taula HTML guardada: {html_path}")

def analyze_all_jsons(results_folder='results', output_dir='results_analysis'):
    """
    Analitza tots els fitxers JSON de la carpeta results
    """
    # Buscar tots els fitxers JSON a la carpeta
    json_files = list(Path(results_folder).glob('*.json'))
    
    if not json_files:
        print(f"❌ No s'han trobat fitxers JSON a la carpeta '{results_folder}'")
        return
    
    print(f"\n🔍 S'han trobat {len(json_files)} fitxers JSON per analitzar:")
    for f in json_files:
        print(f"   - {f.name}")
    
    print(f"\n{'='*80}")
    print("🚀 COMENÇANT ANÀLISI MÚLTIPLE")
    print(f"{'='*80}")
    
    # Analitzar cada fitxer
    results_summary = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n📄 [{i}/{len(json_files)}] Processant: {json_file.name}")
        try:
            df, session_dir = analyze_json(json_file, output_dir)
            
            # Guardar resultat per l'anàlisi global
            results_summary.append({
                'fitxer': json_file.stem,
                'precisio_global': (df['Query_Accuracy'].mean() * 100),
                'total_preguntes': len(df),
                'correctes': df['Query_Accuracy'].sum(),
                'incorrectes': len(df) - df['Query_Accuracy'].sum(),
                'carpeta_sortida': str(session_dir)
            })
            
        except Exception as e:
            print(f"❌ Error processant {json_file.name}: {str(e)}")
    
    # Crear un resum global de tots els fitxers analitzats
    if results_summary:
        print(f"\n{'='*80}")
        print("📊 RESUM GLOBAL DE TOTS ELS FITXERS")
        print(f"{'='*80}")
        
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
        
        # Guardar resum global
        global_summary_path = Path(output_dir) / 'resum_global_tots_fitxers.csv'
        summary_df.to_csv(global_summary_path, index=False)
        print(f"\n✅ Resum global guardat a: {global_summary_path}")
        
        # Crear gràfic comparatiu
        plot_global_comparison(summary_df, Path(output_dir))
    
    print(f"\n{'='*80}")
    print("✅ ANÀLISI COMPLETADA!")
    print(f"{'='*80}")

def plot_global_comparison(summary_df, output_dir):
    """Crea un gràfic comparant la precisió global de tots els fitxers"""
    
    plt.figure(figsize=(12, 6))
    
    # Ordenar per precisió
    summary_df = summary_df.sort_values('precisio_global', ascending=True)
    
    colors = ['red' if x < 30 else 'orange' if x < 60 else 'green' 
              for x in summary_df['precisio_global']]
    
    bars = plt.barh(summary_df['fitxer'], summary_df['precisio_global'], color=colors)
    
    # Afegir etiquetes
    for i, (bar, precisio) in enumerate(zip(bars, summary_df['precisio_global'])):
        plt.text(precisio + 1, bar.get_y() + bar.get_height()/2, 
                f'{precisio:.1f}%', va='center', fontsize=10)
    
    plt.xlabel('Precisió Global (%)', fontsize=12)
    plt.ylabel('Fitxer', fontsize=12)
    plt.title('Comparativa de Precisió Global per Fitxer', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Guardar gràfic
    comparison_path = output_dir / 'comparativa_global.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gràfic comparatiu guardat a: {comparison_path}")
    plt.close()

if __name__ == "__main__":
    #df = analyze_json('results\\udf_results_experiment_20260424_183650_v4.json')
    analyze_all_jsons(results_folder='results', output_dir='results_analysis')
