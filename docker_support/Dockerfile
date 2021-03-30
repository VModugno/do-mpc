FROM ubuntu:20.04
ENV HOME /root
COPY README /
RUN mkdir /workdir && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get install -y python3-pip imagemagick python3-tk curl wget git nano
    #install conda
ENV CONDA_DIR $HOME/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"
RUN conda init bash 
RUN conda info \
    && conda create --name dompc_dev --yes
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
    && mkdir /workdir/amr_dompc \
    && git clone https://github.com/VModugno/do-mpc.git /workdir/amr_dompc/do-mpc
ENV DEFAULT_CONDA_ENV=dompc_dev
RUN echo "conda activate \${DEFAULT_CONDA_ENV}" >> /root/.bashrc
WORKDIR /workdir
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

