import csv
import vobject
import re
from itertools import zip_longest
import pandas as pd



def table_Variant(dic, string, alt):

    dic_aux = {}

    for val in dic['Variant']:
        dic_aux[val[2]] = []
        index = re.finditer(val[0], string)
        if index:
            for match in index:
                index_fin = string.find(val[1], match.end())
                dic_aux[val[2]].append(string[match.end():index_fin])
        
    dic_aux["Alt"] = alt
    dic['Variant'] = dic_aux
    return dic


def table_Annotation(dic, string):
    val = dic['Annotation'][0]
    dic_aux = {}
    list= ["Allele", "Impact", "Consequence"]
    index = re.finditer(val, string)
    if index:
        for match in index:
            index_fin = match.end()
            for i in list:
                dic_aux[i] = []
                index_aux = index_fin
                index_fin = string.find("|", index_aux)
                dic_aux[i].append(string[index_aux:index_fin])

    dic['Annotation'] = dic_aux
    return dic

def table_ChromosomeSequence(dic):
    val = dic['ChromosomeSequence']
    dic_aux = {"chromosome": val}
    dic['ChromosomeSequence'] = dic_aux
    return dic

def table_PositionInfo(dic):
    val = dic['PositionInfo']
    dic_aux = {"pos": val[0], "ref": val[1]}
    dic['PositionInfo'] = dic_aux
    return dic

def comillas(c):
    return f'"{c}"'


def table_HGVSExpression(dic, string):
    dic_aux = {}
    val = dic["HGVSExpression"][0]
    dic_aux[dic["HGVSExpression"][1]] = []
    index = re.finditer(val, string)
    if index:
        for match in index:
            index_fin = string.find(":", match.end())
            dic_aux[dic["HGVSExpression"][1]].append(string[match.end():index_fin])

    #dic_aux["variant_id"] = dic["Variant"]["variant_rs_id"]
            
        
    dic["HGVSExpression"] = dic_aux
    return dic

def table_Interpretation(dic, string):
    dic_aux = {}
    for ref in dic['Interpretation']:
        for val in ref[0]:
            dic_aux[ref[1]] = []
            index = re.finditer(val, string)
            if index:
                for match in index:
                    index_fin = string.find(";", match.end())
                    dic_aux[ref[1]].append(string[match.end():index_fin])
    
    dic_or = { 0:"unknown", 1:"germline", 2:"somatic", 4:"inherited", 8:"paternal", 16:"maternal", 32: "de-novo", 64:"bioparental", 
                  128:"uniparental", 256:"not-tested", 512:"tested-inconclusive", 1073741824:"other" }
    
    list=[]
    for orig in dic_aux["variant_origin"]:
        list.append(dic_or.get(orig))
    dic_aux["variant_origin"] = list
    #dic_aux['variant_rs_id'] =dic['Variant']['variant_rs_id']

    dic['Interpretation'] = dic_aux

    return dic

def table_Disease(dic, string):
    dic_aux = {}
    dic_aux[dic['Disease'][1]] = []
    for val in dic['Disease'][0]:
        index = re.finditer(val, string)
        if index:
            for match in index:
                index_fin = string.find(";", match.end())
                dic_aux[dic['Disease'][1]].append(string[match.end():index_fin])

    dic['Disease'] = dic_aux
    return dic

def table_Database(dic, string):
    dic_aux = {}
    dic_aux[dic['Database'][1]] = []
    for val in dic['Database'][0]:
        index = re.finditer(val, string)
        if index:
            for match in index:
                index_fin = string.find(";", match.end())
                dic_aux[dic['Database'][1]].append(string[match.end():index_fin])

    dic['Database'] = dic_aux
    return dic


def create_csv(dic):
    for key, val in dic.items():
        name_csv= f'{key}.csv'
        data = dic[key]

        max_length = max(len(v) for v in data.values())

        for key in data:
            data[key] += [''] * (max_length - len(data[key]))
        df = pd.DataFrame(data)

        df.to_csv(name_csv, index=False)


def clasificar(string):
    c= "./."
    words = string.split()  # Dividir la cadena en palabras
    index=-1
    chromosome, pos, ref, alt=[], [], [], []

    while True:
        try:
            index = words.index(c, index + 1)  # Encontrar el índice de la palabra buscada
            if index + 1 < len(words):
                chromosome.append(words[index + 1])  # Capturar la siguiente palabra
                pos.append(words[index + 2])
                ref.append(words[index + 4])
                alt.append(words[index + 5])

        except ValueError:
            break  # Salir del bucle cuando no se encuentren más ocurrencias

    
    dic = {"Variant": [["RS=", "\t", "variant_rs_id"], ["CLNVC=", ";", "variant_type"]], "Annotation": ["ANN="], 
           "ChromosomeSequence": chromosome, "PositionInfo": [pos,ref],  "HGVSExpression": ("CLNHGVS=", "hgvs"), 
           "Interpretation": [(["CLNREVSTAT=", " ONCREVSTAT=", "SCIREVSTAT="],"review_status"), (["ORIGIN="],"variant_origin"), 
                              (["SCIINCL=", "SCI="],"clinical_significance")],
           "Disease": (["SCIDNINCL=", "CLNDNINCL=", "ONCDNINCL=", "CLNDN=", "ONCDN=", "SCIDN="], 'preferred_name'), 
           "Database": (["ONCDISDB=", "ONCDISDBINCL=", "SCIDISDB=", "CLNDISDB=", "CLNDISDBINCL=", "SCIDISDBINCL="], "name")}

    
    dic = table_Variant(dic,string, alt)
    dic = table_Annotation(dic, string)
    dic = table_ChromosomeSequence(dic)
    dic = table_PositionInfo(dic)
    dic = table_HGVSExpression(dic, string)
    dic = table_Interpretation(dic, string)
    dic = table_Disease(dic, string)
    dic = table_Database(dic, string)
    print(dic)

    create_csv(dic)



    

def vcf_to_string(file_path):
    with open(file_path, 'r') as file:
        vcf_lines = [line for line in file.readlines() if not line.startswith('#')]
        string = ''.join(vcf_lines)
        string = './. '+ string
        clasificar(string)



# Ruta del archivo de entrada y salida
input_file = 'genes.vcf'
output_file = 'genes.csv'

# Convertir VCF a CSV
vcf_to_string(input_file)


