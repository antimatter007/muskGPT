# Import necessary modules from Flask
from flask import Flask, render_template, jsonify, request

# Import additional modules for configuration and interacting with OpenAI
import config
import openai
import aiapi
import notebook

# Define a custom error handler for 404 (Page Not Found) errors
def page_not_found(e):
    # When a 404 error occurs, render a '404.html' template
    return render_template('404.html'), 404

# Create an instance of the Flask application
app = Flask(__name__)

# Load the configuration settings for the Flask application from the 'development' configuration in 'config.py'
app.config.from_object(config.config['development'])

# Register the custom error handler for 404 errors with the Flask application
app.register_error_handler(404, page_not_found)


# Define the route for the root URL of the application
@app.route('/', methods=['POST', 'GET'])
def index():

    # If the request is a POST request
    if request.method == 'POST':
        # Extract the 'prompt' data from the form in the request
        prompt = request.form['prompt']

        # Initialize a dictionary to hold the response
        res = {}

        # Generate a chat response based on the given prompt using the 'generateChatResponse' function from 'aiapi.py'
        res['answer'] = aiapi.generateChatResponse(prompt)

        # Return the generated response as JSON with a 200 OK status
        return jsonify(res), 200

    # If the request is a GET request, render and return the 'index.html' template
    return render_template('index.html', **locals())

# This block ensures that the Flask application runs only if this script is executed directly (not imported elsewhere)
if __name__ == '__main__':
    # Run the Flask application on host '0.0.0.0' (accessible from any IP address) and port '8888', with debugging enabled
    app.run(host='0.0.0.0', port='8888', debug=True)
