from flask import Flask
from housing.logger import logging
from housing.exception import HousingException
import sys

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
        raise Exception("testing custom exception")
    except Exception as e:
        housing = HousingException(e,sys)   
        logging.info(housing.error_message)
        logging.info("Testing custom exception")
    return "Hello"


if __name__=='__main__':
    app.run(debug=True)