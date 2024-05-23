# testing NL search with elastic and iNat data and CLIP

0. download a mediapipe face detector model from google: https://developers.google.com/mediapipe/solutions/vision/face_detector#models - I'm using the BlazeFace (short-range) since that's the only model available at the time of this writing
1. install elasticsearch and run it. I'm using docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
2. make a python venv and activate it
3. `pip install -r requirements.txt`
4. copy `config.yml.sample` to `config.yml` and edit it
5. `flask reindex`  # this will take a while, mostly due to photo downloading
6. `flask run`
7. visit localhost:5001 in your browser

# Docker version

1. build docker image
`docker build . -t inaturalist/inatnls:latest`

2. run ES and Flask with docker compose
`docker compose up -d`

3. copy sample into the docker container
`docker cp complete_1k_obs_sample.csv inatnls:/app/complete_1k_obs_sample.csv`

4. index sample data
`docker exec -ti inatnls flask reindex complete_1k_obs_sample.csv`

5. visit `localhost:5000` in your browser
