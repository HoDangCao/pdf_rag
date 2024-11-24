# BUILDING A MULTI PDF RAG CHATBOT

## Table of Contents
- [What have been done in this project?](#1-what-have-been-done-in-this-project)
- [Demo](#2-demo)
- [How to run app](#3-how-to-run-app)

## 1. **What have been done in this project?**
Build a websites that allows
- Uploading pdf files (even scanned files).
- Returning information corresponding to questions from users.

For more comprehensive explaination, please check `Multi_PDF_RAG.ipynb` file.

## 2. **Demo**

https://github.com/user-attachments/assets/e8e125ac-d157-4dd4-8e81-1d3394769093

- The user will upload files at the side bar, then click `Submit & Process`.
- The user can type some questions at the input cell and press `Enter`, then the website will return the corresponding information.

## 3. **How to run app**
- step 0: open cmd/terminal at anywhere you wanna place the app.
- step 1: clone this repository `git clone https://github.com/HoDangCao/pdf_rag.git`
- step 2: run `cd ./pdf_rag`

*The 1st way*: manually install
- step 3: run `pip install -r requiments.txt` 
- step 4: run `streamlit run app.py`

*The 2st way*: use Docker
- step 3: run `docker build -t pdf_rag .` 
- step 4: run `docker run -it pdf_rag`

- step 5: Let's experience our website!!!
