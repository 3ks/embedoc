FROM python:3.10 as npy

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt &&\
    python3 main.py &&\
    python3 count.py


FROM python:3.10 as npy

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt &&\
    python3 main.py &&\
    python3 count.py

pip install --no-cache-dir  --use-deprecated=legacy-resolver -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
