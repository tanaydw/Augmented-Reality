pip install cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd /content/CenterTrack
pip install motmetrics==1.1.3
pip install -r requirements.txt
cd /content/CenterTrack/src/lib/model/networks/DCNv2
./make.sh
cd /content/CenterTrack