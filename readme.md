# Development:

1. Create venv directory
```bash
python3 -m venv .venv
```

2. Activate `venv`
- In Powershell/Windows:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
.venv\Scripts\Activate.ps1
```
- In Linux:
```bash
source .venv/Scripts/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```
- Update dependency list
```bash
pip freeze > requirements.txt
```

4. Run the server:
```python
cd app
uvicorn main:app --reload
```

Then you can go to the interactive API docs for documentation and testing:
`[local-host-address:port]/docs` e.g. `http://127.0.0.1:8000/docs`

# Additional Documents:

Why FastAPI?
- https://betterprogramming.pub/fastapi-for-better-ml-in-production-358e555d9ca3


Guides in MLOps:
- https://towardsdatascience.com/image-classification-api-with-tensorflow-and-fastapi-fc85dc6d39e8