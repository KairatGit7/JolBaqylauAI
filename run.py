import os
from dotenv import load_dotenv

load_dotenv()
from jol import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '5000')),
            debug=False, threaded=True, use_reloader=False)
