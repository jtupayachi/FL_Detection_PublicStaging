# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.15.0-gpu
RUN apt-get update && apt-get install -y git curl
RUN pip3 install flwr pandas numpy matplotlib scikit-learn
RUN yes |  apt update



RUN curl https://install.zerotier.com/ | bash 
# RUN /usr/sbin/zerotier-one -d 
#RUN /usr/sbin/zerotier-cli join c7c8172af153068f

# RUN pip3 install flwr[simulation]
# RUN pip3 install ray==1.11.1
# RUN apt-get install  python3-psutil


CMD ["cat", "/etc/os-release"]

RUN yes |  apt install iproute2
