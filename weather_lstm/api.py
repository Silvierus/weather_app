from flask import Flask
from flask_restx import Api, Resource, fields
from lstm import train_data as td


app = Flask(__name__)
api = Api(app, version='1.0', title='TodoMVC API',
    description='A simple TodoMVC API',
)

ns = api.namespace('weather', description='TODO operations')

train_data = api.model('weather_train_loc', {
    'lat': fields.Float(),
    'lon': fields.Float(),
    'model_type': fields.Integer()
})


@ns.route('/train')
class TodoList(Resource):
    @ns.doc('train_weather_location')
    @ns.expect(train_data)
    def post(self):
        lat, lon = api.payload['lat'], api.payload['lon']

        reason = ""
        image = ""
        predicted = 0
        try:
            image, predicted = td(round(float(lat), 3), round(float(lon), 3), int(api.payload['model_type']))
        except Exception as e:
            reason = str(e)

        print(predicted)

        response_body = {
            "reason": reason,
            "resultImage": image,
            "weather_predicted_tmw": str(predicted)
        }

        return response_body, 200 if reason == "" else 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)