FROM python:3.8
WORKDIR /opt/deepcoro/

COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs -y
RUN git lfs install

RUN git clone https://huggingface.co/heartwise/DeepCoro /opt/deepcoro/models_tmp

RUN mv /opt/deepcoro/models_tmp/models /opt/deepcoro/models
RUN rm -R /opt/deepcoro/models_tmp

RUN pip install -r requirements.txt

VOLUME ["/dcm_input", "/results"]

CMD ["/bin/sh", "runall.sh"]