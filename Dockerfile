# Imagem base
FROM python:3.9-slim-buster

# Variáveis de ambiente
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV DJANGO_SETTINGS_MODULE nubank.settings

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get -y install libpq-dev gcc && \
    apt-get install wget -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean 

# Diretorio de trabalho
WORKDIR /app

COPY . /app

# Instalar dependências do Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
# Configurar acesso ao S3

# Rodar migrações do banco de dados
RUN python3 manage.py migrate

#Expor a porta 8081
EXPOSE 8081
# Executar o servidor de desenvolvimento
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8081"]

# Iniciar Gunicorn para produção (Ainda falta ajustar)
#CMD ["gunicorn", "public.wsgi:application", "-b", "0.0.0.0:8000"]