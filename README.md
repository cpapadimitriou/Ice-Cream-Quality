# Ice-Cream-Quality
In this project I built a classifier that distinguishes between good and bad ice cream recipes.

*Modeling_Notebook.ipynb* gives an overview of the feature engineering and methodology.

To run the web app open two terminal windows and run the below commands: 

#### Terminal 1 - UI: 
```
cd ui
npm install -g serve
npm run build
serve -s build -l 3000
```

#### Terminal 2 - Service: 
```
cd service 
virtualenv -p Python3 .
source bin/activate
pip install -r requirements.txt
FLASK_APP=app.py flask run
```