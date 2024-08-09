# Projeto exemplo: Um Tutorial para a Construção com Dados Próprios e Modelos Abertos

Este projeto é uma aplicação de chatbot que utiliza o modelo `LLaMA` (LLaMA:8B) integrado com Streamlit para a interface web e `LangChain` para a manipulação de cadeias de linguagem e para o embedding estamos utilizando o modelo `nomic-embed-text`, o projeto suporta o uso de 3 modelos, sendo eles:

* Modelo I - Usando um modelo pré-treinado diretamente
* Modelo II - Modelo com contexto (RAG)
* Modelo III - Empregando o Fine-Tunning

## DEMO
![gif](assets/demo.gif)

## Instalação
`pip install -r requirements.txt`

## Execução
`ollama run llama3:8b`

`streamlit run src/app.py`

## Autores
- Fabio Martiniano Sato
- Julio Eduardo Silva
- Rogério de Oliveira