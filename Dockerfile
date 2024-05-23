FROM python:3.8

WORKDIR /app

RUN apt-get update -qq && apt-get install -y libgl1-mesa-glx

RUN mkdir /app/images
RUN mkdir /app/logs

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

COPY templates templates
COPY *.py .
COPY config.docker.yml config.yml

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]