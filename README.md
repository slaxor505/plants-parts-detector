# Plantbook Plants Parts Detector 
This app detects part of plants from photos using a Convolutional Neural Net (CNN). 

## Version 1.01
A simple 5-Class ResNet-50 classifier.

# Usage:

## Command line

  python ./plants-parts-detector.py 
  
## Docker
The app can be run in Docker container.

Steps:

1. Clone repository.
2. Inside app directory run:

  docker build -t plants-parts-detector .

3. Run container:

  docker run --name ppd -it --publish 5001:5000 plants-parts-detector
  
  
