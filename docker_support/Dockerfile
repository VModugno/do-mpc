FROM ubuntu:20.04
#FROM nvidia/cuda:11.2.2-base-ubuntu20.04
ENV HOME /root
COPY README /
RUN mkdir /workdir && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get install -y python3-pip imagemagick python3-tk curl wget git nano libjpeg-dev \
    cmake libopenmpi-dev python3-dev libglib2.0-0 libxext6 libsm6 zlib1g-dev xvfb libglu1-mesa ffmpeg freeglut3-dev mesa-utils
    #install conda
ENV CONDA_DIR $HOME/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"
RUN conda init bash 
RUN conda info \
    && conda create --name dompc_dev python=3.6 --yes
#&& conda create --name dompc_dev --yes
#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
#RUN pip3 install jupyterlab 
#This conda env is for the "dev" version of dompc: in this env we install all the dependencies and when the image is run, the repository is installed (from a host repo or a remote one) 
SHELL ["conda", "run", "-n", "dompc_dev", "/bin/bash", "-c"]
RUN conda install pip \
    && pip3 install jupyterlab \
    && pip3 install numpy \
    && pip3 install matplotlib \
    && pip3 install casadi \
    && pip3 install scipy \
    && pip3 install -U pytest \
    && pip3 install pyyaml \
    && mkdir /workdir/amr_repo \
    && git clone https://github.com/VModugno/do-mpc.git /workdir/amr_repo/do-mpc \
    && pip3 install git+https://github.com/hill-a/stable-baselines
ENV DEFAULT_CONDA_ENV=dompc_dev
RUN echo "conda activate \${DEFAULT_CONDA_ENV}" >> /root/.bashrc

#to enable a container run with nvidadocker to connect to the x server of the host if the host has nvidia driver installed
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

WORKDIR /workdir
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

