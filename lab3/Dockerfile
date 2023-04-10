FROM python:3.10
EXPOSE 8501
WORKDIR /app
COPY lab3/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run toxic_classification.py
