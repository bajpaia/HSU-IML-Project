FROM continuumio/miniconda3
WORKDIR /home/mbti-demo-app
COPY . ./
RUN ls 
RUN chmod +x boot.sh
RUN conda create -n env
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/your-environment-name/bin:$PATH
RUN pip install gunicorn
RUN conda install keras
RUN conda install nltk
RUN conda install flask
RUN conda install numpy
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]