from app import create_demo

demo = create_demo()

def app(environ, start_response):
    return demo(environ, start_response)
