FROM python:latest

WORKDIR /app

RUN mkdir /app/images

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY templates templates
COPY app.py app.py
COPY search.py search.py
COPY config.docker.yml config.yml

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]