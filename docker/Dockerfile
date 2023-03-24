FROM nikolaik/python-nodejs:python3.8-nodejs18-bullseye
ARG PYPI=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /meerkat
COPY meerkat/ ./meerkat
COPY scratch/ ./scratch
COPY scripts/ ./scripts
COPY tests/ ./tests
COPY setup.py ./
COPY FairytaleQA_Dataset ./FairytaleQA_Dataset

RUN pip install -e ".[embeddings-linux,text]" --no-cache-dir -i ${PYPI}

WORKDIR /meerkat/meerkat/interactive/app
RUN npm i

RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free \ndeb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-updates main contrib non-free \ndeb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-backports main contrib non-free \ndeb https://mirrors.tuna.tsinghua.edu.cn/debian-security bullseye-security main contrib non-free" > /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y vim

WORKDIR /
#COPY data/datasets/ /home/.meerkat/datasets
#COPY data/dataframes/ /home/.meerkat/dataframes