FROM python:3.8
WORKDIR /opt/deepcoro/

COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
COPY params.json /

# Copy the .env file
COPY .env /opt/deepcoro/.env

# Run the setup script to download all models
RUN python setup.py

# Mount Input/Output Volumes
VOLUME ["/dcm_input", "/results"]

# Run DeepCoro
CMD ["/bin/sh", "runall.sh"]