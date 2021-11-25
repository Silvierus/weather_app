﻿using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace WeatherApp.Logic
{
    public class WeatherRequest
    {
        [JsonProperty("lat")] public double Lat { get; set; }
        [JsonProperty("lon")] public double Lon { get; set; }

    }

    public class WeatherResponse
    {
        [JsonProperty("resultImage")] public string ResultImage { get; set; }
        [JsonProperty("weather_predicted_tmw")] public float WeatherPredictedTomorrow { get; set; }

    }

    public class WeatherPythonApi
    {
        private readonly string _baseApiPath;
        private readonly HttpClient _client;
        public WeatherPythonApi()
        {
            _baseApiPath = "http://localhost:5000/weather";
            _client = new HttpClient();
        }


        public async Task<WeatherResponse> SendTrainLocation(WeatherRequest request)
        {
            var trainEndpoint = $"{_baseApiPath}/train";

            var content = new StringContent(JsonConvert.SerializeObject(request), Encoding.UTF8, "application/json");
            var result = await _client.PostAsync(trainEndpoint, content);

            return JsonConvert.DeserializeObject<WeatherResponse>(await result.Content.ReadAsStringAsync());
        }

    }
}
