The ACM WSDM WebTour 2021 Challenge focuses on a multi destinations trip planning problem, which is a popular scenario in the travel domain.
The goal of this challenge is to make the best recommendation of an additional in-trip destination. Booking.com provides a unique dataset based on millions of real anonymized bookings to encourage the research on sequential recommendation problems.
Many travelers go on trips which include more than one destination. Our mission at Booking.com is to make it easier for everyone to experience the world, and we can help to do that by providing real-time recommendations for what their next in-trip destination will be. By making accurate predictions, we help deliver a frictionless trip planning experience.
Teams are encouraged to compete and submit their trip predictions before January 28th 2021 to qualify for WSDM WebTour challenge.

The goal of this challenge is to use a dataset based on millions of real anonymized accommodation reservations to come up with a strategy for making the best recommendation for their next destination in real-time.


## Dataset
-------------------
The training dataset consists of over a million of anonymized hotel reservations, based on real data, with the following features:
- user_id - User ID
- checkin - Reservation check-in date
- checkout - Reservation check-out date- created_date - Date when the reservation was made
- affiliate_id - An anonymized ID of affiliate channels where the booker came from (e.g. direct, some third party referrals, paid search engine, etc.)
- device_class - desktop/mobile
- booker_country - Country from which the reservation was made (anonymized)
- hotel_country - Country of the hotel (anonymized)
- city_id - city_id of the hotel's city (anonymized)
- utrip_id - Unique identification of user's trip (a group of multi-destinations bookings within the same trip).

Each reservation is a part of a customer's trip (identified by utrip_id) which includes consecutive reservations. 
The evaluation dataset is constructed similarly, however the city_id (and the country) of the final reservation of each trip is concealed and requires a prediction.


## Evaluation Criteria
----------------------
The goal of the challenge is to predict (and recommend) the final city (city_id) of each trip (utrip_id). We will evaluate the quality of the predictions based on the top four recommended cities for each trip by using Accuracy@Top 4 metric (4 representing the four suggestion slots at Booking.com website). When the true city is one of the top 4 suggestions (regardless of the order), it is considered correct.


Attachments
- booking_train_set.csv - train dataset
- sample_test_set.csv - a sample of the test set data (including concealed last destination)
- sample_truth.csv - a sample of true values of the test set
- submission.csv - an example submission for the sample test data
- evaluation_sample.ipynb - a jupyter notebook exampling train set loading, submission generation for test set and evaluation function