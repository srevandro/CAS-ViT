#Required application versions for DAP and CAS
#Change dual boot order: https://www.youtube.com/watch?v=gVw1OMB-D5A&t=32s
   
# #Need to install Visual Studio Tool Box 
conda create -n cas-dap python=3.8
#conda create -n cas-dap python=3.9
conda activate cas-dap
#conda create -n cas-dap_v3.9 python=3.9
#conda activate cas-dap_v3.9
#conda deactivate
#remove -n ENV_NAME --all

#CAS - Install in this order
pip install mmcv-full==1.5.3 
pip install timm==0.5.4  # pip install "timm<0.7"
pip install mmdet==2.24 
pip install mmsegmentation==0.24 
pip install tensorboardX 
pip install einops 
#pip install torch==1.8.0
#pip install torchvision==0.9.1
pip install tabulate
pip install fvcore
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install tensorboard

#DAP ADDITIONAL
pip install tensorflow
pip install tensorflow_datasets
pip install tensorrt

#https://pytorch.org/get-started/locally/
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#conda install pytorch torchvision  torchaudio  cudatoolkit -c pytorch
#conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
#conda install -c conda-forge cudnn  

### Evandro: IMAGENET
#https://stackoverflow.com/questions/64714119/valid-url-for-downloading-imagenet-dataset
#wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
#wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
#Script to extract the ILSVRC datase
#https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
#. ./extract_ILSVRC.sh


#DAP - NÃO PRECISA PARA SOMENTE CAS
# pip install tensorflow==2.5.0
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# conda install -c conda-forge cudnn

# pip install git+https://github.com/facebookresearch/fvcore.git
# pip install numpy==1.23.4
# pip install tensorflow_datasets
# pip install scipy
# pip install ml-collections
# pip install -U --force-reinstall charset-normalizer

#MAY HAVE PROBLEMS IN THE ALGO COMPILATIONS
#ModuleNotFoundError: No module named 'cv2'
#python -m pip install opencv-python

#Solution for python 3.8 no CUDA FOUND - pip install mmcv-full==1.5.3
#https://vinesmsuic.github.io/linux-conda-cudahome/index.html
#export CUDA_HOME=$CONDA_PREFIX
#echo $CUDA_HOME

#Solution for python 3.8 - Failed building wheel for mmcv-full
#conda install gcc gxx_linux-64