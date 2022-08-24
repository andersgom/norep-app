from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/')
def title():
    return "NoRep app - Homepage"



if __name__ == "__main__":
    app.run(debug=True, port=3430)