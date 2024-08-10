from flask import Flask, request, render_template
from input import ask_question  # Import the function from your existing script

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            result = ask_question(query)
            return render_template('index.html', query=query, result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
