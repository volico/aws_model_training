FROM python:3.7


COPY train.py /opt/ml/train.py
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/opt/ml/train.py"]