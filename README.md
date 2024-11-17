# Flood Guard

Flood Guard is a predictive modeling project designed to assess the risk of flooding in Kerala based on current rainfall conditions. By utilizing historical weather data and machine learning techniques, this project aims to forecast potential flooding events, providing essential insights for disaster management and response.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [License](#license)

## Features
- *Data Analysis*: Analyze historical rainfall and weather data to identify patterns and trends.
- *Predictive Modeling*: Implement machine learning algorithms, specifically XGBoost, to predict the likelihood of flooding based on real-time weather conditions.
- *Visualization*: Create visual representations of data to communicate findings effectively.

## Demo of the Project
Here's a demo of how **Flood Guard** works:

![Flood Guard Demo](demo.gif)

## Dataset

The *Flood Guard* project leverages a diverse weather dataset, which includes detailed meteorological data from *U.S. states*. The dataset consists of the following key features:

- *Maximum Temperature (Tmax)*: The highest temperature recorded for each day.
- *Minimum Temperature (Tmin)*: The lowest temperature recorded for each day.
- *Average Temperature (Tavg)*: The mean temperature recorded for each day.
- *Wind Speed (Wspd)*: The average wind speed during the day, measured in meters per second.
- *Air Pressure (Pres)*: Atmospheric pressure, which can influence weather systems and help predict certain weather patterns.

These features are critical for understanding the environmental conditions during potential flooding events, as extreme weather conditions like high temperatures and wind speed can exacerbate flooding risks.

### Severity Prediction

The severity of a disaster, such as flooding, is predicted using machine learning algorithms based on these weather features. For example:

- *High temperatures* can increase the rate of snowmelt or soil evaporation, potentially leading to higher runoff and increased flood risk.
- *Wind speed* plays an important role in cases of hurricanes or storms, worsening flooding due to heavy rainfall.
- *Air pressure* is crucial in identifying storm systems that could lead to extreme weather, including floods.

By analyzing this historical data, the model uses *XGBoost* to predict the likelihood and severity of flooding events, providing valuable insights for disaster preparedness and response.

## Technologies Used
- *Python*: The primary programming language for data analysis and modeling.
- *Pandas*: Data manipulation and analysis library.
- *NumPy*: Library for numerical computing.
- *Scikit-learn*: Machine learning library for predictive modeling.
- *XGBoost*: A gradient boosting framework for efficient and effective predictive modeling.
- *Flask*: Web framework for deploying the machine learning model as a RESTful API.
- *Matplotlib / Seaborn*: Libraries for data visualization.
- *Meteostat API*: For fetching historical weather data.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
