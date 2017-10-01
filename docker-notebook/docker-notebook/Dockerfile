FROM jupyter/tensorflow-notebook:ae885c0a6226
MAINTAINER Elze Kool

USER root

# Install Natural Language Toolkit with popular part of the dataset
RUN conda install --quiet --yes \
    'nltk=3.2*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    python -m nltk.downloader -d /usr/local/share/nltk_data popular

USER $NB_USER




