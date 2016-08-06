FROM rkawajiri/dockerfile4datascience

COPY requirements.txt .
RUN set -ex \
 && pip install -r requirements.txt \
 && rm requirements.txt

COPY src $HOME

USER $USER
EXPOSE 8888
VOLUME ["$HOME"]
WORKDIR $HOME
CMD ["sh", "-c", "jupyter notebook --ip=*"]
