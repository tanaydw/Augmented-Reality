# Carla Simulator

This directory provides various scripts which can be executed on Carla Simulator for training CenterTrack. For running one of these scripts, it is necessary to first download the Carla Simulator, which can be done by following steps on this URL - https://github.com/carla-simulator/carla.

Place these scripts in directory PythonAPI/examples and run one of these scripts after starting the Carla Server. Record the data and train CenterTrack on this data. This can be successfully done using Google Colab.

For example for running 3D Object Detection script, run *client_bounding_boxes.py* and press CTRL+R for start recording the data. The data is then stored in *_out* directory and this can be used for training the model.

If you have GPU and want to train locally, you can do this by integrating CenterTrack in one of these scripts. Instead of recording the data, the data can be provided directly to CenterTrack. However this method of online learning is compute intensive, since Carla as well as training of CenterTrack requires GPU.
