FROM public.ecr.aws/lambda/python:3.10


RUN yum update -y && \
    yum install -y gcc gcc-c++ make && \
    yum clean all

RUN pip install keras-image-helper

RUN python -m pip install --upgrade pip && \
    pip uninstall -y numpy || true && \
    pip install --no-cache-dir \
        "numpy==1.26.4" \
        "pillow==10.4.0" \
        "requests" && \
    pip install --no-cache-dir \
        https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl && \
    pip install --force-reinstall --no-cache-dir "numpy==1.26.4"


COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

WORKDIR ${LAMBDA_TASK_ROOT}

CMD ["src.inference.lambda_handler"]

