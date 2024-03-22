FROM python:3.7.9

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos necessários para o contêiner (por exemplo, requirements.txt)
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte da aplicação para o contêiner
COPY . .

# Define o comando padrão a ser executado quando o contêiner for iniciado
CMD ["python", "app.py"]