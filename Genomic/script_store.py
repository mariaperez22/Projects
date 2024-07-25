import mysql.connector
import glob
import csv
import os

def to_database():

    # Directorio actual
    csv_directory = os.getcwd()

    # Lista de archivos CSV en el directorio
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    cursor = conn.cursor()

    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0]  # Nombre de la tabla será el nombre del archivo CSV sin la extensión

        with open(os.path.join(csv_directory, csv_file), 'r') as file:
            reader = csv.reader(file)

            # Obtiene los nombres de las columnas del archivo CSV
            headers = next(reader)
            print(headers)

            # Verifica si la tabla existe en la base de datos
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            result = cursor.fetchone()

            if result:
                # Inserta los datos en la tabla existente
                insert_query = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({', '.join(['%s']*len(headers))})"
                for row in reader:
                    if any(field.strip() for field in row):
                        cursor.execute(insert_query, row)
            else:
                print(f"La tabla '{table_name}' no existe en la base de datos.")

    # Guarda los cambios y cierra la conexión
    conn.commit()
    cursor.close()

# Configura la conexión a tu base de datos en Clever Cloud
config = {
    'user': 'uympqfivyyyzv6ha',
    'password': 'Ykmu0QN5X1sRLPng0Zw9',
    'host': 'bmffhzqmxqjc6r2tvsfl-mysql.services.clever-cloud.com',
    'database': 'bmffhzqmxqjc6r2tvsfl',
    'port': 3306
}

# Conéctate a la base de datos
conn = mysql.connector.connect(**config)
cursor = conn.cursor()

# Define el comando SQL para crear la tabla
create_table_query = """

CREATE TABLE IF NOT EXISTS ChromosomeSequence (
    chromosome VARCHAR(10),
    assembly VARCHAR(50),
    PRIMARY KEY (chromosome, assembly)
);

CREATE TABLE IF NOT EXISTS Variant (
    variant_rs_id INT PRIMARY KEY,
    alt VARCHAR(50),
    variant_type VARCHAR(50),
    chromosome VARCHAR(10),
    assembly VARCHAR(50),
    FOREIGN KEY (chromosome, assembly) REFERENCES ChromosomeSequence(chromosome, assembly)
);

CREATE TABLE IF NOT EXISTS HGVSExpression (
    hgvs VARCHAR(50),
    variant_rs_id INT,
    PRIMARY KEY (hgvs, variant_rs_id),
    FOREIGN KEY (variant_rs_id) REFERENCES Variant(variant_rs_id)
);

CREATE TABLE IF NOT EXISTS Annotation (
    impact VARCHAR(50),
    consequence VARCHAR(50),
    allele VARCHAR(50)
);


CREATE TABLE IF NOT EXISTS LocationInfo (
    pos INT,
    ref VARCHAR(50),
    chromosome VARCHAR(50),
    assembly VARCHAR(50),
    FOREIGN KEY (chromosome, assembly) REFERENCES ChromosomeSequence(chromosome, assembly)

);

CREATE TABLE IF NOT EXISTS DatabaseInfo (
    name VARCHAR(50) PRIMARY KEY,
    URL VARCHAR(50)
);


CREATE TABLE IF NOT EXISTS Disease (
    preferred_name VARCHAR(50) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Interpretation (
    id_interpretation INT AUTO_INCREMENT PRIMARY KEY,
    clinical_significance VARCHAR(30),
    method VARCHAR(50),
    variant_origin VARCHAR(50),
    review_status VARCHAR(50),
    submitter VARCHAR(50),
    variant_rs_id INT,
    database_id VARCHAR(50),  -- Cambiar el tipo de datos si es necesario
    FOREIGN KEY (database_id) REFERENCES DatabaseInfo(name),
    FOREIGN KEY (variant_rs_id) REFERENCES Variant(variant_rs_id)
);



CREATE TABLE IF NOT EXISTS Interpretation_Disease (
    id_interpretation INT,
    preferred_name VARCHAR(100),
    PRIMARY KEY (id_interpretation, preferred_name),
    FOREIGN KEY (id_interpretation) REFERENCES Interpretation(id_interpretation),
    FOREIGN KEY (preferred_name) REFERENCES Disease(preferred_name)
)

"""

# Ejecuta el comando para crear la tabla
for query in create_table_query.split(";"):
    cursor.execute(query)

# Guarda los cambios y cierra la conexión
conn.commit()
to_database()

conn.close()

print("Tabla creada exitosamente.")

archivos_csv = glob.glob('./')


def comprobar():
    import mysql.connector

    # Configura la conexión a tu base de datos en Clever Cloud
    config = {
        'user': 'uympqfivyyyzv6ha',
        'password': 'Ykmu0QN5X1sRLPng0Zw9',
        'host': 'bmffhzqmxqjc6r2tvsfl-mysql.services.clever-cloud.com',
        'database': 'bmffhzqmxqjc6r2tvsfl',
        'port': 3306
    }

    # Conéctate a la base de datos
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    # Ejecuta una consulta para obtener la información del esquema de cada tabla
    cursor.execute("SHOW TABLES")

    # Recupera los nombres de las tablas
    tables = cursor.fetchall()

    # Para cada tabla, obtenemos su información de esquema
    for table in tables:
        table_name = table[0]
        print(f"Tabla: {table_name}")
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        for column in columns:
            print(f"Nombre de columna: {column[0]}, Tipo: {column[1]}")

    # Cierra el cursor y la conexión
    cursor.close()
    conn.close()

#comprobar()


def eliminar_tabla(n):
    # Configura la conexión a tu base de datos en Clever Cloud
    config = {
        'user': 'uympqfivyyyzv6ha',
        'password': 'Ykmu0QN5X1sRLPng0Zw9',
        'host': 'bmffhzqmxqjc6r2tvsfl-mysql.services.clever-cloud.com',
        'database': 'bmffhzqmxqjc6r2tvsfl',
        'port': 3306
    }

    # Conéctate a la base de datos
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    # Ejecuta una consulta para eliminar la tabla
    table_name = n
    drop_query = f"DROP TABLE IF EXISTS {table_name}"
    cursor.execute(drop_query)

    # Guarda los cambios y cierra la conexión
    conn.commit()
    conn.close()

    print("Tabla eliminada exitosamente.")

#eliminar_tabla("")
#eliminar_tabla("Variant")


