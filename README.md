Local Deployment
1. docker pull docker.io/kadambarikajal12/model-api:latest
2. docker run -p 5000:5030 docker.io/kadambarikajal12/model-api:latest

How to Test:
curl -X POST -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' http://127.0.0.1:5010/predict 

Results :

{"prediction":[0]}
