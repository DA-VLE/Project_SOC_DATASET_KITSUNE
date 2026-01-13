from pyspark.sql import SparkSession, Row  #demarrage du moteur spark pour pouvoir distribuer les calculs
from pyspark.sql.types import StructType, StructField, DoubleType #ici, on importe les structures de données  
from pyspark.sql import functions as F #nous importons notre boite de outils pour if else, etc

def main():
    # 1) Créer la session Spark
    spark = SparkSession.builder \ 
        .appName("KitsuneSparkExample") \
        .getOrCreate() #recupere un builder spark deja disponible sinon on cree un new avec kitsunesparkExample

    sc = spark.sparkContext # c'est la méthode ancienne pour pouvoir utiliser rdd

    # 2) Lire le CSV Kitsune comme RDD de lignes texte
    dataset_path = "/opt/project/data/MitM_dataset.csv"  
    rdd_lines = sc.textFile(dataset_path)

    # 3) Transformer chaque ligne CSV en liste de floats
    def parse_line(line): #line=donnée brute # parsing = netoyage ligne par ligne
        str_values = line.split(",")# on separe les valeurs  a chaque virgule
        floats = [float(x) for x in str_values]# on convertir du texte en en nombre decimal
        return Row(**{f"f{i}": floats[i] for i in range(len(floats))}) #On transforme cette liste de nombres en un objet Ligne structuré, et on donne des noms provisoires aux colonnes (f0, f1, f2...).

    rdd_rows = rdd_lines.map(parse_line) # principe du MapReduce, cette fonction magique applique la fonction perse_line /
    #sur le milliard de lignes de fichier, et fais le en parallèle sur tous les processeurs

    # 4) Construire le schéma automatiquement à partir d'un exemple
    sample = rdd_rows.first()
    fields = [StructField(name, DoubleType(), nullable=False)
              for name in sample.asDict().keys()]
    schema = StructType(fields)

    # 5) Convertir l'RDD en DataFrame
    df = spark.createDataFrame(rdd_rows, schema)

    print("➡️ DataFrame initial :")
    df.show(5, truncate=False)

    # 6) Ajout regle soc (rule_alert)
    # si f1 > 1500, on met alerte = 1
    df_rules = df.withColumn(
        "rule_alert",
        F.when(F.col("f1") > 1500.0, 1).otherwise(0)
    )

    print("➡️ Avec la colonne rule_alert :")
    df_rules.select("f0", "f1", "rule_alert").show(20, truncate=False)

    # 7) Quelques stats : combien d'alertes ?
    total = df_rules.count()
    alerted = df_rules.filter(F.col("rule_alert") == 1).count()
    print(f"📊 Nombre total de lignes : {total}")
    print(f"🚨 Nombre de lignes avec alerte rule_alert=1 : {alerted}")

    spark.stop()

if __name__ == "__main__":
    main()
