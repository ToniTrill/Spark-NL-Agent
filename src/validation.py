import json

def normalize(res):
    if not res: 
        return []
    
    flat_list = []
    for row in res:
        # converitr fila en llista de vlaors
        values = list(row) if isinstance(row, (tuple, list)) else row.asDict().values()
        
        clean_values = []
        for v in values:
            if v is None:
                clean_values.append("None")
            elif isinstance(v, (float, int, str)):
                try:
                    #Aixi 163, "163" i 163.00001 es tornen "163.0" conveirtr a float i arodonir
                    num = round(float(v), 2)
                    clean_values.append(str(num))
                except ValueError:
                    #si no es nuemro ho deixem com a text en minuscules
                    clean_values.append(str(v).strip().lower())
            else:
                clean_values.append(str(v))
        
        flat_list.append(tuple(clean_values))        
        return sorted(flat_list) #ordenar per si l'ordre importes.

def validate(spark, nl_query, model_result, json_path="db/bird-1/dev.json"):
    # Comprova si el resultat de la IA es igual al dels resultats dev.json
    print("-" * 60)
    print("Valiadnt els resultats \n")

    #buscar la pregunta i resposta, golen sql, al dev.json
    golden_sql = None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item["question"].strip().lower() == nl_query.strip().lower():
            golden_sql = item["SQL"]
            break
    
    if not golden_sql: return False

    try:
        #Executar el SQL original per aseugrar nos que el resultat es correcte
        print(f"Executant Golden SQL \n")
        golden_result_df = spark.sql(golden_sql)
        golden_data = golden_result_df.collect() #llista de files

        #Comparar els resultats
        return normalize(model_result) == normalize(golden_data)
    except Exception as e:
        return False