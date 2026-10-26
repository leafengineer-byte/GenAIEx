# GAN Face Generator — Flask + VS Code

This project wraps the DCGAN Generator from `Genaiex6.ipynb` in a small Flask web app.

## Project files

- `app.py` — Flask routes and the same Generator architecture as the notebook
- `generator.pth` — trained weights from Colab
- `templates/index.html` — web interface
- `requirements.txt` — Python dependencies

## Run in VS Code (Windows)

1. Extract this ZIP into a folder and open that folder in VS Code.
2. Make sure `generator.pth` is in the same folder as `app.py`.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt or run:

   ```cmd
   .venv\Scripts\activate.bat
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   PyTorch installation can take a while. If installation fails, use the official PyTorch selector for the right command for your Windows/Python setup.

6. Start the app:

   ```powershell
   python app.py
   ```

7. Open `http://127.0.0.1:5000` in your browser.

## Important

- Do **not** run the Kaggle download or model-training cells for this Flask app. You already have the trained `generator.pth` file.
- The model architecture and latent size (`nz=100`) must match the notebook that produced the weights.
- If `generator.pth` is missing, put your downloaded copy beside `app.py`.
- For this notebook's trained model, generation runs on CPU by default if CUDA is unavailable; generating a few images may be slower on CPU.
- This is a local demo. `debug=True` is suitable for local development only, not public deployment.
