FROM gcr.io/tensorflow/tensorflow

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# pandoc prerequisites
RUN set -ex \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    pandoc \
    texlive-latex-recommended \
    texlive-xetex \
    texlive-luatex \
    pandoc-citeproc \
 && rm -rf /var/lib/apt/lists/*

# pip install requirements
COPY requirements.txt .
RUN set -ex \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && rm requirements.txt
