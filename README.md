# testing NL search with elastic and iNat data and CLIP

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

2. create .htpasswd file
`htpasswd -c .htpasswd inat`

3. run ES and Flask with docker compose
`docker compose up -d`

4. copy sample into the docker container
`docker cp complete_1k_obs_sample.csv inatnls:/app/complete_1k_obs_sample.csv`

5. index sample data
`docker exec -ti inatnls flask reindex complete_1k_obs_sample.csv`

6. visit `localhost:5000` in your browser
