# ODDS ASSASSIN X — Free mobile prototype

Read-only Kalshi public market scanner. No account login, private API key, order placement, or real-money trading.

## Deploy free from an iPhone

1. Create a free GitHub account at https://github.com.
2. In Safari, create a new public repository named `odds-assassin-x`.
3. Upload `app.py`, `requirements.txt`, and `render.yaml` from this project.
4. Create a free Render account at https://render.com.
5. In Render choose **New → Web Service**, connect the GitHub repository, and choose the **Free** instance.
6. Render should detect the Python app. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:app`.
7. Deploy, then open the generated `onrender.com` URL on your iPhone.

Free Render services sleep after 15 minutes without inbound traffic and may take about a minute to wake up. Do not use this prototype for automated or live trading.

## Local development

```bash
pip install -r requirements.txt
python app.py
```
