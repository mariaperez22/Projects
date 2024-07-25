import subprocess


#download SnpEff
""" def download_program(url, destino):
    try:
        # Ejecutar el comando wget en la terminal para descargar el archivo
        subprocess.run(["wget", url, "-O", destino])
        print("Descarga completa.")
    except Exception as e:
        print("Error al descargar el archivo:", e)

def unzip(archivo_zip):
    try:
        # Ejecutar el comando unzip en la terminal
        subprocess.run(["unzip", archivo_zip])
        print("Archivo descomprimido exitosamente.")
    except Exception as e:
        print(f"Error al descomprimir el archivo: {e}") """

def annotate_sift(vcf1,vcf2,vcf_res,snpS):
    try:
        # Ejecutar el comando join en la terminal
        java_command = "java"
        java_args = ["-jar", snpS, "annotate", vcf1, vcf2, ">", vcf_res]
        print(java_args)
        java_args_str = " ".join(java_args)
       
        process = subprocess.Popen(f"{java_command} {java_args_str}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
       
        if stderr:
           print("Error:", stderr.decode("utf-8"))
        else:
           print("Process completed.")
           
        print("Document annotated.")
    except Exception as e:
        print(f"Error annotating the vcf: {e}")

def annotate_eff(snpE, ref, vcf_or, vcf_res):
    try:
        java_command = "java"
        java_args = ["-jar", snpE, ref, vcf_or, ">", vcf_res]
        print(java_args)
        java_args_str = " ".join(java_args)
       
        process = subprocess.Popen(f"{java_command} {java_args_str}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
       
        if stderr:
           print("Error:", stderr.decode("utf-8"))
        else:
           print("Process completed")
           
        print("Document converted")
    except Exception as e:
        print(f"Error in document conversion: {e}")



#Input
data_path = "clinvar.vcf"
snpE_path = "snpEff/snpEff.jar"
snpS_path = "snpEff/SnpSift.jar"

test_path = "test.vcf"
reference_gene = "GRCh37.75"

#Output
eff_path = "eff.vcf"
annotate_path= "genes.vcf"



annotate_eff(snpE_path, reference_gene, test_path, eff_path)
annotate_sift(data_path, eff_path, annotate_path, snpS_path)








