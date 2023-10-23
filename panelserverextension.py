from subprocess import Popen

def load_jupyter_server_extension(nbapp):
    """serve the app.py with the panel server"""
    Popen(["panel", "serve", "images_registration.ipynb", "--allow-websocket-origin=*"])