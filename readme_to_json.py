import json
import re
import os

#Generar json amb les metadates de les funcions udf i les preguntes extretes de un readme
def parse_readme_to_json(readme_path, output_json_path):
    if not os.path.exists(readme_path):
        print(f"Error: No s'ha trobat el fitxer {readme_path}")
        return

    with open(readme_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    udfs = []
    queries =[]
    current_query = {}

    for line in lines:
        line_str = line.strip()

        q_match = re.match(r'<summary>(Q\d+)\s*-\s*(.*?)</summary>', line_str)
        if q_match:
            current_query = {
                "id": q_match.group(1).strip(),
                "name": q_match.group(2).strip()
            }

        # exrtreure deswcripcio, input i output
        if "- **Description**:" in line_str and "id" in current_query:
            current_query["description"] = line_str.split("**Description**:")[1].strip()

        if "- **Input**:" in line_str and "id" in current_query:
            current_query["input"] = line_str.split("**Input**:")[1].strip()

        if "- **Output**:" in line_str and "id" in current_query:
            current_query["output"] = line_str.split("**Output**:")[1].strip()
            # quan tenim l'output, donem la query per acabada i la guardem
            queries.append(current_query)
            current_query = {} # Resetejem per a la seguent

        # buscar linies que comencin per "| U" i tinguin suficients columnes
        if line_str.startswith('| U') and line_str.count('|') >= 8:
            parts = [p.strip() for p in line_str.split('|')]
            #  [0]="", [1]=ID, [2]=Name, [3]=Desc, [4]=Type, [5]=#In, [6]=InType, [7]=#Out, [8]=OutType
            udf_id = parts[1]
            if udf_id.startswith('U') and udf_id[1:].isdigit():
                udfs.append({
                    "id": udf_id,
                    "name": parts[2],
                    "description": parts[3],
                    "input_type": parts[6],
                    "output_type": parts[8]
                })

    unique_udfs = list({udf["id"]: udf for udf in udfs}.values())

    final_data = {
        "queries": queries,
        "udfs": unique_udfs
    }

    #crearla carpeta si no existeix
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)

    print(f" Extracció completada amb èxit!")
    print(f"   - S'han trobat {len(queries)} Queries.")
    print(f"   - S'han trobat {len(unique_udfs)} UDFs.")
    print(f"   - Fitxer guardat a: {output_json_path}")


if __name__ == "__main__":
    PATH_AL_README = "db/udfbench/README.md"
    PATH_OUTPUT = "db/udfbench/udf_catalog.json"
    
    parse_readme_to_json(PATH_AL_README, PATH_OUTPUT)