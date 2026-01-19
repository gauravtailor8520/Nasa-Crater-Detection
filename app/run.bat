@echo off
echo Installing requirements...
pip install -r d:\datashare\app\requirements.txt

echo Starting NASA Crater Detection App...
python d:\datashare\app\app.py
pause
