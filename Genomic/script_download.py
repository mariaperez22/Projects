import os
import requests
from github import Github

# URL of the VCF
url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz"

# local path for the doc
file_path = "clinvar.vcf"

# GET Request
response = requests.get(url)

# Comprobar si la solicitud fue exitosa
# 
if response.status_code == 200:
    #decodify bytes with UTF-8
    vcf_content = response.content.decode("utf-8", errors="ignore")
    
    # Save VCF
    with open("archivo.vcf", "w", encoding="utf-8") as vcf_file:
        vcf_file.write(vcf_content)
    print(" VCF downloaded")
else:
    print("Error downloading the VCF:", response.status_code)
