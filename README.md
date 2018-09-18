# AICamp CV Final Project - Face Recognition System

## Prework

Download the [model](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) of facenet, and put all the files into `recognize/facenet_model`.

```
cp /path/to/unzip/dir/* recognize/facenet_model
```

## Run
First, install the python dependencies.

```
pip install -r requirements.txt
```

Then, run the webserver.

```
python webface.py
```

## To calculate score only
```
python test.py
```

## To test facial Detection with MTCNN
```
cd test
python one_image_test.py
```
