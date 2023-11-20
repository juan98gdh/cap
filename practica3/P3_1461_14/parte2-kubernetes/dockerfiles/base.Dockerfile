FROM ubuntu:22.04

# 1. Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y default-jdk default-jre curl

# 2. define spark and hadoop versions
ENV SPARK_VERSION=3.3.1
ENV HADOOP_VERSION=3.3.4

# 3. Download and extract spark
RUN mkdir -p /opt && \
    cd /opt && \
    curl http://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz | \
        tar -zx && \
    ln -s spark-${SPARK_VERSION}-bin-hadoop3 spark

# 4. Download and extract hadoop
RUN mkdir -p /opt && \
    cd /opt && \
    curl http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | \
        tar -zx hadoop-${HADOOP_VERSION}/lib/native && \
    ln -s hadoop-${HADOOP_VERSION} hadoop

# 5. Add spark and hadoop to PATH
ENV PATH $PATH:/opt/spark/bin