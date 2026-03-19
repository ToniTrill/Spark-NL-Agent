import json
import ast

#convertir cada valor en string i ordenar (per si venen en ordre diferent les respostes)
# (163, "163", 163.0) -> "163.0", text tot a minuscules i nulls ("None")
#Permet comapra restlats idnepemdenmtent del formati i ordre de la resposta
def normalize(res):
    if not res: 
        return []

    if isinstance(res, str):
        try:
            res = ast.literal_eval(res)
        except (ValueError, SyntaxError):
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
                    num = round(float(v), 4)
                    clean_values.append(str(num))
                except ValueError:
                    #si no es nuemro ho deixem com a text en minuscules
                    clean_values.append(str(v).strip().lower())
            else:
                clean_values.append(str(v))
        
        flat_list.append(tuple(clean_values))        
    return sorted(flat_list) #ordenar per si l'ordre importes.

#Compara valors normalitzats String amb toelrancia
#Si es poden convertir a numero acepta tolerancia. SI es text comparacio exacta.
def equals_value(v1, v2, tolerance=0.0001):
    try:
        return abs(float(v1) - float(v2)) <= tolerance
    except ValueError:
        return v1 == v2

#Mira si el resultat de la IA es igual a golden query(al fitxer de respostes ja doant). 
def validate(spark, nl_query, model_result, json_path):
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

        norm_model = normalize(model_result)
        norm_golden = normalize(golden_data)

        if len(norm_model) != len(norm_golden):
            return False
        for row_model, row_golden in zip(norm_model, norm_golden):
            if len(row_model) != len(row_golden):
                return False
            if not all(equals_value(v1, v2) for v1, v2 in zip(row_model, row_golden)):
                return False
        return True
    except Exception as e:
        return False