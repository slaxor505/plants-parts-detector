FROM python:3.7-buster

#MAINTAINER Slava Pisarevskiy "slava@plantbook.io"
#to build
#docker build -t plants-parts-detector:latest .

#to run container
#docker run --name ppd -it --detach --publish 5001:5000 --restart=always -m 512m plants-parts-detector:latest gunicorn plants-parts-detector:app --workers=1 -b 0.0.0.0:5000
#with binded img_pool
#docker run --name ppd -it --detach --publish 5001:5000 --restart=always -m 512m --mount type=bind,source=ppd_img_pool.docker,target=/app/static/img_pool plants-parts-detector gunicorn plants-parts-detector:app --workers=1 -b 0.0.0.0:5000
#with binded entire app folder for dynamic development and updates. Need to restart container once app updated.
#docker run --name ppd -it --detach --publish 5001:5000 --restart=always -m 512m --mount type=bind,source=`pwd`,target=/app plants-parts-detector:latest gunicorn plants-parts-detector:app --workers=1 -b 0.0.0.0:5000

#preinstalling CPU only pytorch to reduce image footprint as we don't use GPU here
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev && \
    pip install --upgrade pip && \
	pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

# to run as Flask app - not suitable for production
#ENTRYPOINT [ "python" ]
#CMD [ "plants-parts-detector.py","production" ]

# Using production WSGI server 'gunicorn plants-parts-detector:app --workers=4 -b 0.0.0.0:5000'
CMD [ "gunicorn", "plants-parts-detector:app", "--workers=1", "-b", "0.0.0.0:5000" ]


