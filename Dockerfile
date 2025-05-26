# Imagem base
FROM python:3.10-slim

# Diretório da aplicação
WORKDIR /app

# Copia arquivos
COPY . .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta da API
EXPOSE 5000

# Comando para iniciar a API
CMD ["python", "app.py"]
