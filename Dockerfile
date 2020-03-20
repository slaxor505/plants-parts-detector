FROM python:3.7-buster

#MAINTAINER Slava Pisarevskiy "slava@plantbook.io"


#preinstalling CPU only pytorch to reduce image footprint as we don't use GPU here
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev && \
    pip install --upgrade pip && \
	pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

#ENTRYPOINT [ "python" ]
#CMD [ "plants-parts-detector.py","production" ]
#'gunicorn plants-parts-detector:app --workers=4 -b 0.0.0.0:5000'

CMD [ "gunicorn", "plants-parts-detector:app", "--workers=1", "-b", "0.0.0.0:5000" ]

#to run container
#docker run --name ppd -it --detach --publish 5001:5000 --restart=always -m 512m plants-parts-detector:0.11 gunicorn plants-parts-detector:app --workers=1 -b 0.0.0.0:5000