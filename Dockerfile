FROM python:3.8
WORKDIR /opt/deepcoro/

COPY . .
RUN pip install --upgrade pip
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs -y
RUN git lfs install
RUN git clone https://huggingface.co/heartwise/DeepCoro /opt/deepcoro/models
RUN rm -rf /opt/deepcoro/models/.git
RUN pip install -r requirements.txt

VOLUME ["/dcm_input", "/results"]

CMD ["/bin/sh", "runall.sh"]