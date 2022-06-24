# Dependencies

First, there are a few Python dependencies you need to install. You can do it in a venv if you like, but I'll just provide instructions for a global install:

* `$ python -m pip install flask`

* `$ python -m pip install flask-cors`

* `$ python -m pip install python-dotenv`


`flask` is the module for the backend itself.

`flask-cors` is the an extension for Flask allowing the Flask and React servers to communicate.

`python-dotenv` just allows us to easily configure certain environment variables needed for Flask.

# Adding a proxy

In order for React to communicate with Flask, we need to set up a proxy to handle cross-origin issues (I don't think we necessarily need flask-cors if we do this, but I don't really want to take the risk in trying to sort it all out).

To do this, you should have some `package.json` file somewhere, usually in the directory you call `npm start` from, but it might be somewhere else. If you can't find it, let me know and I'll try and figure out another solution.

Then, add a `"proxy"` key to the file:

    {
      "proxy": "http://localhost:5000",

      ... the rest of the options ...
    }

# Using the API

The API is all written in the `api/api.py` file. The only function you need to care about is the `send_advert()` function on line 61. To use it, make a POST request from the frontend. I've included an example of how this might work on lines 64-83, but let me know if you want me to explain anything.

# Starting the backend

First, start the frontend as usual. For me this is `yarn start` but I think you said yours was with `npm start`.

Now, start the backend in another terminal by navigating to the `api/` folder and running `$ flask run`. The terminal running Flask should also provide some debugging information in case anything goes wrong.

That's it! Hopefully the backend and frontend will now be running and be able to communicate, but let me know if there's something that's not working.



