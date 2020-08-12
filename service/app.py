from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import sys
import pandas as pd
from src.util import clean_text, lemmatize_text, tfidf

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Ice Cream Recipe Classifier", 
		  description = "Predict the quality of an ice cream recipe")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'cookingSteps': fields.String(required = True, 
				  							   description="Cooking Steps", 
    					  				 	   help="Cooking Steps cannot be blank"),
				  'ingredients': fields.String(required = True, 
				  							description="Ingredients", 
    					  				 	help="Ingredients cannot be blank")})

classifier = joblib.load('classifier.joblib')
vectorizer = joblib.load('vectorizer.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			ingredients = [x.strip() for x in formData['ingredients'].split(',')]
			X_count = vectorizer.transform(ingredients)
			countvector = X_count.toarray().astype(int)
			x_ingredients = np.array([sum(x) for x in zip(*countvector)])
			cooking_steps = lemmatize_text(clean_text(formData['cookingSteps']))
			x_cooking = tfidf_vectorizer.transform([cooking_steps]).toarray()[0]
			x = np.concatenate([x_ingredients, x_cooking])
			prediction = classifier.predict(x.reshape(1, -1))
			types = { 0: "Bad Recipe", 1: "Good Recipe"}
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Recipe Quality Prediction: " + types[prediction[0]]
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})