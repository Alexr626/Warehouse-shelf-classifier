FROM python:3.10-slim
# TODO: Use the appropriate base image for your model/service

# TODO: Install all your dependencies here
#RUN pip install jupyterlab
#RUN pip install notebook
RUN pip install torch
RUN pip install python-abc
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install wandb
RUN pip install pandas
RUN pip install matplotlib
RUN pip install opencv-python-headless
RUN pip install Pillow
RUN pip install scipy


WORKDIR /workspace
COPY . /workspace

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Expose port for JupyterLab
#EXPOSE 8888

# TODO: Start your service here