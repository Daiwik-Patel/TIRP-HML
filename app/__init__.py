from flask import Flask

def create_app():
    app = Flask(__name__)

    # Initialize any required services, blueprints, etc.
    # Example: register routes, models, etc.
    from .routes import main
    app.register_blueprint(main)

    return app
