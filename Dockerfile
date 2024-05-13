FROM python:3.8
WORKDIR /opt/deepcoro/

COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
COPY params.json /

VOLUME ["/dcm_input", "/results"]

CMD ["/bin/sh", "runall.sh"]