from flask import Flask, render_template, request
from search_core import SearchEngine

app = Flask(__name__)

# Initialize search engine on startup
search_engine = SearchEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "")
    results = []
    if query:
        results = search_engine.search(query)
    return render_template("results.html", query=query, results=results)


if __name__ == "__main__":
    app.run(debug=True)
