import pandas as pd


def evaluate_accuracy_at_4(submission, ground_truth):
    '''checks if the true city is within the four recommended cities'''
    data_to_eval = submission.join(ground_truth, on='utrip_id')
    hits = data_to_eval.apply(
        lambda row: row['city_id'] in (
            row[['city_id_1', 'city_id_2', 'city_id_3', 'city_id_4']].values
        ), axis=1)
    return hits.mean()


if __name__ == '__main__':
    train_set = pd.read_csv('../input/booking_train_set.csv',
                            usecols=[
                                'user_id',
                                'checkin',
                                'checkout',
                                'city_id',
                                'device_class',
                                'affiliate_id',
                                'booker_country',
                                'hotel_country',
                                'utrip_id'
                            ]).sort_values(by=['utrip_id', 'checkin'])
    test_set = pd.read_csv('../input/sample_test_set.csv',
                           usecols=[
                               'user_id',
                               'checkin',
                               'checkout',
                               'device_class',
                               'affiliate_id',
                               'booker_country',
                               'utrip_id',
                               'row_num',       # test only
                               'total_rows',    # test only
                               'city_id',
                               'hotel_country'
                           ]).sort_values(by=['utrip_id', 'checkin'])
    print(train_set.shape, test_set.shape)

    topcities = train_set.city_id.value_counts().index[:4]
    print(topcities)

    test_trips = (test_set[['utrip_id']].drop_duplicates()).reset_index().drop('index', axis=1)
    cities_prediction = pd.DataFrame([topcities] * test_trips.shape[0],
                                     columns=['city_id_1', 'city_id_2', 'city_id_3', 'city_id_4'])

    submission = pd.concat([test_trips, cities_prediction], axis=1)
    submission.to_csv('../input/submission.csv', index=False)
    submission = pd.read_csv('../input/submission.csv', index_col=[0])
    ground_truth = pd.read_csv('../input/sample_truth.csv', index_col=[0])
    print(evaluate_accuracy_at_4(submission, ground_truth))
