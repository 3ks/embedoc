FROM python:3.10 as langchain-serve-img

RUN pip3 install langchain-serve api

WORKDIR /app

COPY . .

CMD [ "lc-serve", "deploy", "local", "api" ]

FROM python:3.10 as pdf-gpt-img

WORKDIR /app

COPY . .

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD [ "python3", "app.py" ]
