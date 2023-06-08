FROM env:0.1 as langchain-serve-img

WORKDIR /app

COPY . .

CMD [ "lc-serve", "deploy", "local", "api" ]

FROM env:0.1 as pdf-gpt-img

WORKDIR /app

COPY . .

CMD [ "python3", "app.py" ]
