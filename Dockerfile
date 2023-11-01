# Use a customized pytorch 1.4 with cuda 9.2 for compatibility with DGX's cuda-10.0
FROM anibali/pytorch:cuda-9.2

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update
RUN apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev -y

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
ENV NAME World

ENV LD_LIBRARY_PATH /usr/local/lib

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]