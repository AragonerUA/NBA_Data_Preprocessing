import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def clean_data(file_path):
    df = pd.read_csv(file_path)
    df["b_day"] = pd.to_datetime(df["b_day"], format="%m/%d/%y")
    df["draft_year"] = ["01/01/"+str(x) for x in df["draft_year"]]
    df["draft_year"] = pd.to_datetime(df["draft_year"], format="%m/%d/%Y")
    df["team"].fillna("No Team", inplace=True)
    df["height"] = df.height.apply(lambda x: float(str(x).split("/")[1].lstrip()))
    df["weight"] = df.weight.apply(lambda x: float(str(x).split("/")[1].split(" ")[1].lstrip()))
    df["salary"] = df.salary.apply(lambda x: float(x[1:]))
    df["country"] = [x if x == "USA" else "Not-USA" for x in df["country"]]
    df["draft_round"].replace(to_replace="Undrafted", value="0", inplace=True)
    return df


def feature_data(cleaned_dataframe):
    cleaned_dataframe["version"] = ["12/31/20"+str(x)[-2:] for x in cleaned_dataframe["version"]]
    cleaned_dataframe["version"] = pd.to_datetime(cleaned_dataframe["version"], format="%m/%d/%Y")
    cleaned_dataframe["age"] = [int(int(str(cleaned_dataframe.version[i] - cleaned_dataframe.b_day[i]).split()[0])//365.2425) for i in range(len(cleaned_dataframe.version))]
    cleaned_dataframe["experience"] = [int(int(str(cleaned_dataframe.version[i] - cleaned_dataframe.draft_year[i]).split()[0])//365.2425) for i in range(len(cleaned_dataframe.version))]
    cleaned_dataframe["bmi"] = [cleaned_dataframe.weight[i] / (cleaned_dataframe.height[i]**2) for i in range(len(cleaned_dataframe.weight))]
    cleaned_dataframe.drop(columns=["version", "b_day", "draft_year", "weight", "height"], inplace=True)
    cleaned_dataframe.drop(columns=["college", "draft_peak", "jersey", "full_name"], inplace=True)
    # print(cleaned_dataframe)
    return cleaned_dataframe


def multicol_data(df):
    # print(df.corr())
    df.drop(columns="age", inplace=True)
    return df


def transform_data(df):
    num_feat_df = df.select_dtypes('number')  # numerical features
    cat_feat_df = df.select_dtypes('object')  # categorical features

    scaler = StandardScaler()
    sc = scaler.fit_transform(num_feat_df.drop(columns="salary"))
    scaler_df = pd.DataFrame(sc)
    scaler_df.columns = ["rating", "experience", "bmi"]

    encoder = OneHotEncoder(sparse=False)
    enc = encoder.fit_transform(cat_feat_df)
    encoder_df = pd.DataFrame(enc)
    enc_columns = list()
    for col_cat in encoder.categories_:
        for column in col_cat:
            enc_columns.append(column)
    encoder_df.columns = enc_columns

    normalized_features = pd.concat((scaler_df, encoder_df), axis=1)
    X = normalized_features
    y = df.salary
    return X, y


if __name__ == "__main__":
    pd.set_option('display.max_columns', 15)
    # Checking ../Data directory presence
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'nba2k-full.csv' not in os.listdir('../Data'):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/nba2k-full.csv', 'wb').write(r.content)
        print('Loaded.')

    data_path = "../Data/nba2k-full.csv"

    # write your code here
    # df_cleaned = clean_data(data_path)
    # df_featured = feature_data(df_cleaned)
    # df = multicol_data(df_featured)
    # X, y = transform_data(df)
    #
    # answer = {
    #     'shape': [X.shape, y.shape],
    #     'features': list(X.columns),
    # }
    # print(answer)