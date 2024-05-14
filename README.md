# Flight Data Analysis and Prediction Project

## Overview
This project focuses on analyzing flight data, performing data engineering tasks, and developing a machine learning model to predict flight delays using Google Cloud Platform (GCP) services. The workflow encompasses data loading, transformation, exploratory data analysis (EDA), feature engineering, model development, deployment, and pipeline orchestration.

## Workflow
1. **Data Loading:**
   - Raw flight data in CSV format is uploaded to Google Cloud Storage (GCS) bucket.

2. **Data Transformation & Cleaning:**
   - Data is ingested from GCS into BigQuery for transformation and cleaning.
   - Data cleaning tasks include handling missing values, standardizing formats, and removing duplicates.

3. **Exploratory Data Analysis (EDA):**
   - EDA is performed directly in BigQuery to understand data characteristics, identify patterns, and derive insights.

4. **Dashboard Creation:**
   - Data from BigQuery is used to create interactive dashboards using Google Data Studio.
   - Dashboards are served to stakeholders and end-users for visualization and analysis.

5. **Feature Engineering:**
   - Cloud Dataprep is utilized for feature engineering tasks.
   - Relevant features are extracted and transformed to enhance model performance.

6. **Model Development:**
   - Google Cloud AI Platform is used to develop machine learning models based on the engineered features.
   - Models are trained and tuned for predicting flight delays.

7. **Model Deployment:**
   - Trained models are saved to a GCS bucket for deployment.
   - Deployment is done on Google Kubernetes Engine (GKE) for serving predictions to end-users.

8. **Pipeline Orchestration:**
   - Cloud Composer (Apache Airflow) orchestrates the entire pipeline on GKE.
   - Tasks for data processing, model training, deployment, and monitoring are defined using Airflow's Directed Acyclic Graphs (DAGs).

9. **Monitoring and Logging:**
   - Cloud Monitoring and Cloud Logging are used for pipeline monitoring, resource utilization tracking, and logging.
   - Custom dashboards and alerts are set up to ensure pipeline reliability and performance.

## Repository Structure
- **data:** Contains sample flight data files in CSV format.
- **notebooks:** Jupyter notebooks for exploratory data analysis and model development.
- **scripts:** Python scripts for data preprocessing, feature engineering, model training, and deployment.
- **dashboards:** Data Studio dashboard configurations and visualizations.
- **pipeline:** DAG definitions for Cloud Composer orchestration.
- **docs:** Additional documentation files.

## Usage
1. **Data Loading and Cleaning:**
   - Run the data loading script to upload raw data to GCS.
   - Execute the data preprocessing script to ingest and clean data in BigQuery.

2. **EDA and Dashboard Creation:**
   - Perform EDA in BigQuery using provided SQL queries.
   - Use Data Studio to import dashboard configurations and visualize insights.

3. **Feature Engineering:**
   - Utilize Cloud Dataprep for feature engineering tasks.

4. **Model Development and Deployment:**
   - Develop machine learning models using provided notebooks on AI Platform Notebooks.
   - Train and save models to GCS for deployment.
   - Deploy models on GKE for serving predictions.

5. **Pipeline Orchestration and Monitoring:**
   - Define DAGs in Cloud Composer for orchestrating the pipeline.
   - Set up monitoring and logging using Cloud Monitoring and Cloud Logging.

## Requirements
- Google Cloud Platform account with appropriate permissions.
- Python environment with necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, stats models, etc...).
- Google Cloud SDK installed for command-line access to GCP services.

## References
- [Google Cloud Platform Documentation](https://cloud.google.com/docs)
- [Google Data Studio Documentation](https://developers.google.com/datastudio)
- [Cloud Composer Documentation](https://cloud.google.com/composer/docs)
- [Google Cloud AI Platform Documentation](https://cloud.google.com/ai-platform/docs)

## License
This project is licensed under the [MIT License](LICENSE).
