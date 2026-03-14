<div align="center">

# 🛡️ Projet SOC Big Data  
### Détection d’attaques en streaming avec Kafka, Spark, ELK & ML

![Kafka](https://img.shields.io/badge/Kafka-Streaming-black?logo=apachekafka)
![Spark](https://img.shields.io/badge/Apache%20Spark-Structured%20Streaming-orange?logo=apachespark)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-Analytics-005571?logo=elasticsearch)
![Kibana](https://img.shields.io/badge/Kibana-Visualization-005571?logo=kibana)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-Data%20Pipeline-3776AB?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-brightgreen)

</div>

---

## 📌 Présentation

Ce projet consiste à concevoir une **architecture SOC temps réel** capable d’**ingérer**, **enrichir**, **analyser** et **visualiser** des événements de sécurité à partir du dataset **Kitsune**.

> **Objectif :** reproduire une chaîne de traitement proche d’un SOC moderne en combinant  
> **streaming**, **machine learning** et **visualisation de données de sécurité**.

---

## 🏗️ Architecture du pipeline

```text
Producer CSV → Kafka → Spark Structured Streaming → Kafka enrichi → Logstash → Elasticsearch → Kibana
```

## Mini Description: 
Développement d’un pipeline SOC temps réel à partir du dataset Kitsune : Producer CSV → Kafka → Spark Structured Streaming → Kafka enrichi → Logstash → Elasticsearch → Kibana

Création d’un dataset global homogène à partir de plusieurs datasets d’attaques, avec harmonisation des colonnes, ajout des labels et traitement en chunks pour éviter les saturations RAM

Entraînement d’un pipeline ML Spark MLlib en offline, incluant assemblage des features, normalisation et classification Random Forest multi-classes

Déploiement d’une étape d’inférence en streaming permettant d’enrichir chaque événement avec une attaque prédite, un niveau de confiance, un score d’anomalie et des règles d’alerte SOC

Indexation et visualisation des événements enrichis dans Elasticsearch et Kibana pour analyse, investigation et suivi des volumes d’alertes

Traitement de problèmes réels de fiabilité et d’exploitation : configuration Kafka, healthchecks Docker, document_id unique côté Logstash, replay Kafka, nettoyage de volumes Docker et gestion de la volumétrie Elasticsearch

📄 [Consulter le rapport complet du projet](Rapport_Project_SOC.pdf)
