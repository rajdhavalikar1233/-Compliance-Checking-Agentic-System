FROM debian:bullseye

# Install system dependencies for building Python
RUN apt-get clean && apt-get update --fix-missing && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    git

# Set working directory
WORKDIR /usr/src

# Download and build Python 3.13.2
RUN wget https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tgz && \
    tar xvf Python-3.13.2.tgz && \
    cd Python-3.13.2 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Set up work directory for your app
WORKDIR /app

# Copy your app files into the container
COPY . /app

# Create and activate virtual environment using Python 3.13
RUN python3.13 -m venv venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r requirements.txt

# Use the venv path
ENV PATH="/app/venv/bin:$PATH"

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app4.py"]
