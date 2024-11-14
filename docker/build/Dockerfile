FROM python:3.11-alpine
MAINTAINER Xander Wilcke "w.x.wilcke@vu.nl"

ARG VERSION_HYPODISC=1.0.0

# update repositories
#RUN apt-get update && apt-get -y upgrade && apt-get -y install curl bash py3-pip && apt-get -y clean
RUN apk update && apk add --no-cache gcc g++ gfortran lapack-dev libffi-dev libressl-dev musl-dev patch curl bash && rm -rf /var/cache/apk/*

# add new user to install applications under
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home /home/datalegend \
    --ingroup users \
    --shell /bin/bash \
    datalegend

RUN echo "datalegend:datalegend" | chpasswd

USER datalegend
WORKDIR /home/datalegend

RUN python3 -m venv python_venv
WORKDIR /home/datalegend/python_venv

# clone COW release
RUN mkdir -p $PWD/hypodisc
RUN curl -L "https://github.com/wxwilcke/hypodisc/archive/refs/tags/$VERSION_HYPODISC.tar.gz" | tar xz -C $PWD/hypodisc --strip-components=1
WORKDIR /home/datalegend/python_venv/hypodisc
RUN ../bin/pip3 install . && ../bin/pip3 cache purge

# set correct server address
COPY browse.diff ./
RUN patch -i browse.diff hypodisc/browse.py
RUN rm browse.diff

WORKDIR /home/datalegend

RUN echo "[[ -f ~/.bashrc ]] && . ~/.bashrc" > .bash_profile
RUN echo "PS1='\[\e[0;32m\][\[\e[0;37m\]\u\[\e[0;32m\]] \[\e[0;33m\]\W \[\e[0;37m\]\$ '" > .bashrc
RUN echo "alias python='/home/datalegend/python_venv/bin/python'" >> .bashrc

USER root

# Expose port 5000 for the Flask application
EXPOSE 5000

RUN mkdir -p /mnt/shared/
RUN chown -R datalegend:users /mnt/shared/

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
