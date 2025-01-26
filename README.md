Local Training 

1. git clone git@github.com:kadambarikajal/mlops_1.git
2. dvc pull 
3. python3 model_training/model.py

Local Deployment & Testing

1. docker pull docker.io/kadambarikajal12/model-api:latest
2. docker run -p 6000:6000 docker.io/kadambarikajal12/model-api:latest

How to Test:
curl -X POST -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' http://127.0.0.1:6000/predict 

Results :

{"prediction":[0]}



DVC change data version 

1. change md5 in data/iris.data.dvc (other md5sum with last row repeated ae244df326bd230685a3e47e0521ad56) 
2. run training again .. 
