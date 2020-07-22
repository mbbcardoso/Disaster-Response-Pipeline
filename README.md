# Disaster-Response-Pipeline

## Table of Contents
1. [Requirements](#requirements)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Requirements <a name="requirements"></a>
- pandas 0.23.3
- numpy 1.12.1
- sqlalchemy 1.1.13
- nltk 3.2.5
- scikit-learn 0.19.1

## Project Motivation <a name="motivation"></a>

In this project, I completed an app that classifies messages sent during disasters into different categories in order to help the job of disaster response organizations

## File Descriptions <a name="files"></a>

- data/process_data.py: A data processing pipeline that prepares the data and stores it in a SQLite database
- models/train_classifier.py: A machine learning pipeline that does preprocessing of the data and then trains, tunes and stores the classifier
- ETL Pipeline Preparation.ipynb: A jupyter notebook used to develop the data processing pipeline
- ML Pipeline Preparation.ipynb: A jupyter notebook used to develop the machine learning pipeline

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Credit to Figure Eight for the dataset and Udacity for the project outline, instructions and templates
