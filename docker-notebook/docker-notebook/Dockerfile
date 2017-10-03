FROM jupyter/tensorflow-notebook:ae885c0a6226
MAINTAINER Elze Kool

USER root

# Install Natural Language Toolkit with popular part of the dataset
RUN conda install --quiet --yes \
    'nltk=3.2*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    python -m nltk.downloader -d /usr/local/share/nltk_data popular

# Install statsmodels
RUN conda install --quiet --yes \
    'statsmodels=0.8.0*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR

# Install pydotplus
RUN apt-get update && apt-get install -y graphviz &&\
    conda install --quiet --yes \
    'pydotplus=2.*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR

USER $NB_USER




