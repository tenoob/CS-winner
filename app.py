from flask import Flask

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return "Hello"


if __name__=='__main__':
    app.run(debug=True)

"""
e8d5bbb3-388d-433b-b8b1-3299413f6227
cs-winner-v2"""