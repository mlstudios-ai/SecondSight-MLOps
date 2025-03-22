# API Endpoint
Endpoint for model inferencing using FastAPI.

## Environment Setup

1. Create a new conda environment (or use an existing one):

    ```sh
    conda create -p venv/ python==3.11
    conda activate venv/
    ```

2. Install the required dependencies:

    ```sh
    pip3 install -r requirements.txt
    ```

## Running the Server

To start the server, run the following command:

```sh
uvicorn app.main:app --reload
```